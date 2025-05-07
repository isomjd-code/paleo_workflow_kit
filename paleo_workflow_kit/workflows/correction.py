# paleo_workflow_kit/workflows/correction.py

import logging
import time
import os
import json
import re
import shutil
from typing import Dict, Any, Optional, List, Tuple

# Import base class and components
from .base_workflow import BaseWorkflow
from paleo_workflow_kit.transkribus_api import TranskribusAPI
from paleo_workflow_kit.page_xml_handler import PageXMLHandler, PageXMLParseError
from paleo_workflow_kit.image_handler import ImageHandler, ImageProcessingError
from paleo_workflow_kit.llm_clients.base_llm import BaseLLMClient
from paleo_workflow_kit.llm_clients.anthropic_client import AnthropicClient # Specific import for batch methods
from paleo_workflow_kit.llm_clients.gemini_client import GeminiClient # Specific import if needed
from paleo_workflow_kit.utils.text_utils import calculate_cer
from paleo_workflow_kit.utils.cache_manager import load_llm_cache, save_llm_cache, load_batch_cache, save_batch_cache, remove_from_batch_cache
from paleo_workflow_kit.config import (
    TEMP_XML_FILENAME_BASE, OUTPUT_IMAGE_GEMINI_BASE, OUTPUT_IMAGE_CLAUDE_BASE,
    TEMP_CHUNK_IMAGE_DIR, CER_THRESHOLD, METADATA_TAG_VALUE, SCRIPT_VERSION,
    CLEANUP_TEMP_FILES, CLEANUP_SEGMENTED_IMAGES, DOC_TITLE_REGEX, AALT_INDEX_URL,
    BATCH_CACHE_FILE
)
# Import custom exceptions if defined
# from paleo_workflow_kit.exceptions import WorkflowConfigurationError, LLMGenerationError

# Get logger for this module
logger = logging.getLogger(__name__)

class CorrectionWorkflow(BaseWorkflow):
    """
    Implements the combined Gemini + Claude transcription correction workflow.

    Steps:
    1. Filters documents (optional, checks if already corrected).
    2. For each eligible document's first page:
        a. Downloads original XML and image.
        b. Fetches external entities (e.g., WAALT).
        c. Prepares image for Gemini (rotate, draw lines).
        d. Calls Gemini LLM twice.
        e. Prepares image for Claude (draw lines on original, segment).
        f. Calls Claude LLM (batch API) twice.
        g. Compares the 4 LLM transcriptions using CER.
        h. If consensus (max pairwise CER < threshold):
            i. Modifies original XML with chosen transcription (e.g., Gemini A)
               and adds combined processing metadata.
            j. Uploads modified XML to Transkribus.
        i. Cleans up temporary files.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the CorrectionWorkflow."""
        super().__init__(*args, **kwargs)

        # Validate required components
        if not isinstance(self.transkribus_api, TranskribusAPI):
            raise TypeError("CorrectionWorkflow requires an initialized TranskribusAPI instance.")
        if not issubclass(self.page_xml_handler_cls, PageXMLHandler):
             raise TypeError("CorrectionWorkflow requires a PageXMLHandler class.")
        if not isinstance(self.image_handler, ImageHandler):
            raise TypeError("CorrectionWorkflow requires an initialized ImageHandler instance.")
        if not self.llm_clients or not isinstance(self.llm_clients, dict):
             raise ValueError("CorrectionWorkflow requires a dictionary of initialized LLM clients.")
        if "gemini" not in self.llm_clients or not isinstance(self.llm_clients["gemini"], BaseLLMClient):
             raise ValueError("CorrectionWorkflow requires an initialized 'gemini' LLM client.")
        if "anthropic" not in self.llm_clients or not isinstance(self.llm_clients["anthropic"], AnthropicClient): # Need specific type for batch
             raise ValueError("CorrectionWorkflow requires an initialized 'anthropic' LLM client (AnthropicClient type).")

        # Load caches
        self.llm_cache = load_llm_cache(self.config.get("LLM_CACHE_FILE", "llm_correction_cache.json"))
        self.batch_cache = load_batch_cache(self.config.get("BATCH_CACHE_FILE", "anthropic_batch_cache.json"))
        self.batch_cache_file_path = self.config.get("BATCH_CACHE_FILE", "anthropic_batch_cache.json")

        # Workflow specific state initialization
        self.results = {
            "documents_checked_filter": 0,
            "documents_skipped_filter": 0,
            "documents_processed_loop": 0,
            "pages_processed": 0,
            "successful_updates": 0,
            "skipped_cer_threshold": 0,
            "llm_failures": 0,
            "other_failures": 0,
        }

    def _cleanup_iteration_files(self, files_to_remove: List[Optional[str]], dirs_to_remove: List[Optional[str]]):
        """Cleans up temporary files and directories for a single iteration."""
        for f_path in files_to_remove:
            if f_path and os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    self.logger.debug(f"Cleaned up temp file: {f_path}")
                except OSError as e:
                    self.logger.warning(f"Could not remove temp file {f_path}: {e}")

        if self.config.get('CLEANUP_SEGMENTED_IMAGES', True):
            for dir_path in dirs_to_remove:
                if dir_path and os.path.exists(dir_path) and os.path.isdir(dir_path):
                    try:
                        shutil.rmtree(dir_path)
                        self.logger.debug(f"Cleaned up temp directory: {dir_path}")
                    except OSError as e:
                        self.logger.warning(f"Could not remove temp directory {dir_path}: {e}")

    def run(self, collection_id: int, process_all: bool = False) -> Dict[str, Any]:
        """
        Runs the correction workflow.

        Args:
            collection_id: The Transkribus collection ID to process.
            process_all: If True, attempts to process all documents, ignoring
                         the 'already corrected' check (useful for reprocessing).

        Returns:
            A dictionary summarizing the results.
        """
        self._log_start()
        self.results["collection_id"] = collection_id
        self.results["process_all_flag"] = process_all

        gemini_client = self.get_llm_client("gemini")
        anthropic_client = self.get_llm_client("anthropic") # Type checked in __init__

        # Ensure clients are available (check again just in case)
        if not gemini_client or not anthropic_client:
             self._add_error("Required LLM clients (Gemini, Anthropic) are not available.", {})
             self._log_end()
             return self.get_summary()

        try:
            # 1. Get Document List
            all_docs = self.transkribus_api.list_documents(collection_id)
            total_docs_in_collection = len(all_docs)
            self.results["total_documents_in_collection"] = total_docs_in_collection
            if not all_docs:
                self.logger.warning("No documents found in collection.")
                return self.get_summary()

            # 2. Filter Documents (Check if already corrected by this script)
            docs_to_process = []
            metadata_tag = self.config.get('METADATA_TAG_VALUE', f"CombinedCorrected_v{self.config.get('SCRIPT_VERSION', 'unknown')}")
            self.logger.info(f"Filtering documents. Checking for status tag: '{metadata_tag}' (process_all={process_all})")

            for doc_info in all_docs:
                self.results["documents_checked_filter"] += 1
                doc_id = doc_info['docId']
                doc_title = doc_info['title']
                self.logger.debug(f"Checking filter status for Doc {doc_id} ('{doc_title}')")

                if process_all:
                    self.logger.debug(f"process_all=True, adding Doc {doc_id} regardless of status.")
                    # Still need page_nr for processing
                    page_nr_filter, _, _ = self.transkribus_api.get_page_details(collection_id, doc_id, 0)
                    if page_nr_filter is not None:
                        docs_to_process.append({'docId': doc_id, 'title': doc_title, 'pageNr': page_nr_filter, 'page_index': 0})
                    else:
                        self._add_error(f"Filter Check: Could not get page details for Doc {doc_id}", {"doc_id": doc_id})
                        self.results["other_failures"] += 1 # Count as other failure
                    continue

                # Get page details for the first page (index 0)
                page_nr_filter, _, _ = self.transkribus_api.get_page_details(collection_id, doc_id, 0)
                if page_nr_filter is None:
                    self._add_error(f"Filter Check: Could not get page details for Doc {doc_id}", {"doc_id": doc_id})
                    self.results["other_failures"] += 1
                    continue

                # Check the latest transcript metadata for the specific tag
                # Assuming check_xml_for_combined_corrected uses the API client correctly
                already_corrected = self.transkribus_api.check_xml_for_combined_corrected(
                    collection_id, doc_id, page_nr_filter, metadata_tag
                ) # Pass session implicitly via self.transkribus_api

                if already_corrected is None:
                    self._add_error(f"Filter Check: Error checking correction status for Doc {doc_id}", {"doc_id": doc_id})
                    self.results["other_failures"] += 1
                elif already_corrected is True:
                    self.logger.info(f"Filter Check: Doc {doc_id} already has status '{metadata_tag}'. Skipping.")
                    self.results["documents_skipped_filter"] += 1
                else: # False
                    self.logger.info(f"Filter Check: Doc {doc_id} does not have status '{metadata_tag}'. Adding to process list.")
                    docs_to_process.append({'docId': doc_id, 'title': doc_title, 'pageNr': page_nr_filter, 'page_index': 0})

            total_to_process = len(docs_to_process)
            self.logger.info(f"Filtering complete. {total_to_process} documents to process.")
            if total_to_process == 0:
                self.logger.warning("No documents eligible for processing after filtering.")
                return self.get_summary()

            # 3. Processing Loop
            for idx, doc_data in enumerate(docs_to_process):
                current_doc_id = doc_data['docId']
                current_doc_title = doc_data['title']
                current_page_nr = doc_data['pageNr'] # Use pageNr obtained during filtering
                current_page_index = doc_data['page_index'] # Should be 0
                self.results["documents_processed_loop"] += 1

                # --- Initialize variables for this iteration ---
                page_id = None; img_url = None; xml_data = None; image_bytes = None;
                rotated_image_bytes_gemini = None; updated_xml_data_gemini = None
                numbered_rotated_img_path_actual = None; numbered_original_img_path_actual = None
                htr_lines_from_xml = []; original_lines_data = []
                lines_Gemini_A, lines_Gemini_B = None, None
                lines_Claude_A, lines_Claude_B = None, None
                modified_xml_bytes = None; iteration_success = False
                temp_chunk_image_paths_for_doc = []
                named_entity_list = "Not fetched yet."
                iteration_status = "pending"

                # --- Generate unique file paths ---
                safe_title_part = re.sub(r'[\\/*?:"<>|]', "_", current_doc_title).replace('.JPG', '').replace('.jpg', '')
                temp_xml_path = self.config.get("TEMP_DIR") / f"{self.config.get('TEMP_XML_FILENAME_BASE', 'temp_page')}_{safe_title_part}.xml"
                numbered_rotated_image_path_gemini = self.config.get("TEMP_DIR") / f"{self.config.get('OUTPUT_IMAGE_GEMINI_BASE', 'numbered_rotated_page_gemini')}_{safe_title_part}.jpg"
                numbered_original_image_path_claude = self.config.get("TEMP_DIR") / f"{self.config.get('OUTPUT_IMAGE_CLAUDE_BASE', 'numbered_original_page_claude')}_{safe_title_part}.jpg"
                segmentation_output_subdir = self.config.get("TEMP_CHUNK_IMAGE_DIR") / safe_title_part

                files_to_remove_finally = [str(temp_xml_path), str(numbered_rotated_image_path_gemini), str(numbered_original_image_path_claude)]
                dirs_to_remove_finally = [str(segmentation_output_subdir)]

                self.logger.info(f"\n======= PROCESSING Doc {idx + 1}/{total_to_process} (Title: {current_doc_title}, ID: {current_doc_id}) =======")

                try:
                    # 3.1 Get Page Details (mainly for img_url)
                    _, page_id, img_url = self.transkribus_api.get_page_details(collection_id, current_doc_id, current_page_index)
                    if img_url is None: raise ValueError("Could not retrieve image URL.")

                    # 3.2 Download XML
                    if not self.transkribus_api.download_page_xml_file(collection_id, current_doc_id, current_page_nr, str(temp_xml_path)):
                        raise ValueError("Failed to download PAGE XML.")

                    # 3.3 Parse XML
                    page_handler = self.create_page_xml_handler(temp_xml_path.read_bytes())
                    if not page_handler: raise ValueError("Failed to parse PAGE XML.") # Error logged in helper
                    xml_data = { # Reconstruct data needed by image handler etc.
                        'lines': page_handler.get_lines_data(),
                        'htr_full_text_lines': page_handler.get_text_lines(),
                        'image_width': page_handler.get_page_dimensions()[0] if page_handler.get_page_dimensions() else 0,
                        'image_height': page_handler.get_page_dimensions()[1] if page_handler.get_page_dimensions() else 0,
                    }
                    htr_lines_from_xml = xml_data['htr_full_text_lines']
                    original_lines_data = xml_data['lines']
                    num_original_lines = len(htr_lines_from_xml)
                    if num_original_lines < 1: raise ValueError("XML contains 0 text lines.")

                    # 3.4 Download Image
                    image_bytes = self.image_handler.download_image(img_url)
                    if not image_bytes: raise ValueError("Failed to download image.")

                    # 3.5 Fetch Entities
                    doc_metadata = parse_doc_title(current_doc_title) # Use helper
                    current_target_image_number = doc_metadata.get('image_num', -1) if doc_metadata else -1
                    current_target_side = doc_metadata.get('side', 'unknown') if doc_metadata else 'unknown'
                    aalt_url = self.config.get("AALT_INDEX_URL")
                    if aalt_url and current_target_image_number != -1 and current_target_side != 'unknown':
                        # Assuming fetch_and_parse_waalt_wiki_index exists and is imported
                        named_entity_list = fetch_and_parse_waalt_wiki_index(aalt_url, current_target_image_number, current_target_side)
                        if named_entity_list is None: logger.warning("Failed to fetch/parse WAALT entities."); named_entity_list = "Entity fetch failed."
                        elif not named_entity_list: logger.info("No WAALT entities found."); named_entity_list = "No specific entities found."
                        else: logger.info("Successfully retrieved WAALT entities.")
                    else: logger.warning("Skipping entity fetch (missing URL, or failed title parse)."); named_entity_list = "Entity fetch skipped."

                    # --- Gemini Processing ---
                    self.logger.info("--- Starting Gemini Processing ---")
                    rotated_image_bytes_gemini, transformed_lines_gemini, new_dims_gemini = self.image_handler.rotate_image_and_coords(
                        image_bytes, original_lines_data, xml_data['image_width'], xml_data['image_height']
                    )
                    if not rotated_image_bytes_gemini or not transformed_lines_gemini: raise ValueError("Gemini image rotation/coord transformation failed.")

                    numbered_rotated_img_path_actual = self.image_handler.draw_baselines_numbers(
                        rotated_image_bytes_gemini, transformed_lines_gemini, str(numbered_rotated_image_path_gemini)
                    )
                    if not numbered_rotated_img_path_actual: raise ValueError("Failed to draw lines on rotated Gemini image.")

                    gemini_responses = []
                    for attempt in [1, 2]:
                        resp = gemini_client.generate_correction(
                            numbered_rotated_img_path_actual, htr_lines_from_xml, named_entity_list,
                            GEMINI_PROMPT, # Assuming GEMINI_PROMPT is defined globally or in config
                            run_identifier=f"G{attempt}",
                            max_output_tokens=self.config.get("GEMINI_MAX_TOKENS", 8192) # Example kwarg
                        )
                        if resp: gemini_responses.append(resp)
                        else: raise ValueError(f"Gemini call attempt {attempt} failed.")
                        time.sleep(1.5)
                    if len(gemini_responses) != 2: raise ValueError("Did not get 2 successful Gemini responses.")

                    lines_Gemini_A = [gemini_responses[0]['latin_transcription']['abbreviated_latin_draft'][str(k)] for k in range(1, num_original_lines + 1)]
                    lines_Gemini_B = [gemini_responses[1]['latin_transcription']['abbreviated_latin_draft'][str(k)] for k in range(1, num_original_lines + 1)]
                    if len(lines_Gemini_A) != num_original_lines or len(lines_Gemini_B) != num_original_lines: raise ValueError("Gemini line count mismatch.")
                    self.logger.info("--- Finished Gemini Processing ---")

                    # --- Claude Processing ---
                    self.logger.info("--- Starting Claude Processing ---")
                    # Draw lines on original image for Claude reference/segmentation input
                    numbered_original_img_path_actual = self.image_handler.draw_baselines_numbers(
                        image_bytes, original_lines_data, str(numbered_original_image_path_claude)
                    )
                    if not numbered_original_img_path_actual: raise ValueError("Failed to draw lines on original image for Claude.")

                    # Call Claude Batch (expects 2 results)
                    lines_Claude_A, lines_Claude_B = anthropic_client.get_claude_correction_batch(
                        image_bytes, # Pass original bytes
                        htr_lines_from_xml,
                        original_lines_data,
                        named_entity_list,
                        anthropic_client.client, # Pass the underlying client object if needed by the method
                        current_doc_title,
                        current_doc_id,
                        temp_chunk_image_paths_for_doc, # List to collect temp files
                        self.batch_cache,
                        self.batch_cache_file_path
                    )
                    if lines_Claude_A is None or lines_Claude_B is None: raise ValueError("Claude batch processing failed.")
                    if len(lines_Claude_A) != num_original_lines or len(lines_Claude_B) != num_original_lines: raise ValueError("Claude line count mismatch.")
                    self.logger.info("--- Finished Claude Processing ---")

                    # --- 4-Way CER Comparison ---
                    self.logger.info("Performing 4-way CER comparison...")
                    max_overall_pairwise_cer = 0.0
                    for line_idx in range(num_original_lines):
                        cers = [
                            calculate_cer(lines_Gemini_A[line_idx], lines_Gemini_B[line_idx]),
                            calculate_cer(lines_Gemini_A[line_idx], lines_Claude_A[line_idx]),
                            calculate_cer(lines_Gemini_A[line_idx], lines_Claude_B[line_idx]),
                            calculate_cer(lines_Gemini_B[line_idx], lines_Claude_A[line_idx]),
                            calculate_cer(lines_Gemini_B[line_idx], lines_Claude_B[line_idx]),
                            calculate_cer(lines_Claude_A[line_idx], lines_Claude_B[line_idx])
                        ]
                        max_cer_for_line = max(cers) if cers else 0.0
                        max_overall_pairwise_cer = max(max_overall_pairwise_cer, max_cer_for_line)
                    self.logger.info(f"4-Way CER comparison complete. Max Overall Pairwise CER = {max_overall_pairwise_cer:.4f}")

                    # --- Decision and Upload ---
                    cer_thresh = self.config.get('CER_THRESHOLD', 0.10)
                    if max_overall_pairwise_cer < cer_thresh:
                        self.logger.info(f"Max CER {max_overall_pairwise_cer:.4f} < {cer_thresh}. Proceeding with update.")
                        # Modify XML using Gemini A and add combined metadata tag
                        modified_xml_bytes = page_handler.modify_xml_tree_combined(
                            lines_Gemini_A, # Chosen transcription
                            self.config.get('METADATA_TAG_VALUE', 'CombinedCorrected_vDefault')
                        )
                        if not modified_xml_bytes: raise ValueError("Failed to modify XML tree.")

                        # Upload
                        if not self.dry_run:
                            update_success = self.transkribus_api.upload_page_xml(
                                collection_id, current_doc_id, current_page_nr, modified_xml_bytes,
                                {'status': 'DONE', 'note': f'Combined Correction v{SCRIPT_VERSION}', 'toolName': f'CombinedScript_v{SCRIPT_VERSION}'}
                            )
                            if update_success:
                                iteration_success = True
                                self.results["successful_updates"] += 1
                                remove_from_batch_cache(self.batch_cache_file_path, self.batch_cache, current_doc_id) # Remove from cache on success
                            else:
                                raise ValueError("Transkribus upload failed.")
                        else:
                            self.logger.info("[DRY RUN] Skipping Transkribus upload.")
                            iteration_success = True # Count as success in dry run if it got this far
                            self.results["successful_updates"] += 1

                    else:
                        self.logger.warning(f"Max CER {max_overall_pairwise_cer:.4f} >= {cer_thresh}. SKIPPING update for Doc {current_doc_id}.")
                        self.results["skipped_cer_threshold"] += 1
                        iteration_status = "skipped_cer" # Mark specific skip reason

                except Exception as e:
                    self.logger.error(f"Error processing Doc {current_doc_id}: {e}", exc_info=True)
                    iteration_status = "fail_processing"
                    self._add_error(f"Processing error for Doc {current_doc_id}", {"doc_id": current_doc_id, "title": current_doc_title, "exception": str(e)})
                    # No iteration_success = True
                finally:
                    # --- Cleanup for this iteration ---
                    self._cleanup_iteration_files(files_to_remove_finally, dirs_to_remove_finally)
                    if iteration_success:
                        log_entry['status'] = "success"
                    elif iteration_status != "pending": # If already marked as skipped_cer or failed
                         log_entry['status'] = iteration_status
                    else: # Default to fail if not success and not skipped
                         log_entry['status'] = "fail"
                         self.results["other_failures"] += 1 # Count generic failures

                    log_entry['timestamp_end'] = datetime.datetime.now().isoformat()
                    self.logger.info(f"======= FINISHED Doc {idx + 1}/{total_to_process} ({current_doc_title}) - Status: {log_entry['status'].upper()} =======")
                    # Update overall counters based on final status
                    if log_entry['status'] == 'success': self.results['successful_updates'] += 1 # Redundant if already counted? Check logic.
                    # Failures are counted within the except block or implicitly if iteration_success is False

            # --- End Document Loop ---

        except Exception as outer_e:
            self.logger.critical(f"Critical error occurred outside main loop: {outer_e}", exc_info=True)
            self._add_error(f"Critical workflow error: {outer_e}", {"exception": str(outer_e)})
        finally:
            # --- Final Save Cache ---
            save_llm_cache(self.llm_cache)
            save_batch_cache(self.batch_cache_file_path, self.batch_cache)
            self._log_end() # Log end time and summary

        return self.get_summary() # Return final summary

    def get_summary(self) -> Dict[str, Any]:
        """Returns the summary dictionary for the correction workflow."""
        duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
        # Consolidate results
        self.results["total_errors"] = len(self.errors) + self.results.get("llm_failures", 0) + self.results.get("other_failures", 0)

        summary = {
            "workflow_name": self.__class__.__name__,
            "start_time_iso": datetime.datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            "end_time_iso": datetime.datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration_seconds": round(duration, 2),
            "dry_run": self.dry_run,
            "collection_id": self.results.get("collection_id", "N/A"),
            "total_documents_in_collection": self.results.get("total_documents_in_collection", 0),
            "documents_checked_filter": self.results.get("documents_checked_filter", 0),
            "documents_skipped_filter": self.results.get("documents_skipped_filter", 0),
            "documents_processed_loop": self.results.get("documents_processed_loop", 0),
            "pages_processed": self.results.get("pages_processed", 0), # Might be same as docs processed if 1 page/doc
            "successful_updates": self.results.get("successful_updates", 0),
            "skipped_cer_threshold": self.results.get("skipped_cer_threshold", 0),
            "total_failures": self.results.get("total_errors", 0),
            "llm_failures": self.results.get("llm_failures", 0), # Specific LLM failures if tracked
            "other_failures": self.results.get("other_failures", 0), # Other processing failures
            "errors_list": self.errors # Include detailed errors
        }
        return summary

# --- Example Instantiation and Run (if needed for direct execution) ---
if __name__ == "__main__":
    # This block allows running the workflow directly.
    # It requires setting up config and dependencies manually here,
    # which is less ideal than using a dedicated run script.

    # Basic logging setup if running directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    logger.info("--- Running Correction Workflow Directly ---")

    # --- Load Configuration ---
    # Assuming config.py exists and loads .env
    from paleo_workflow_kit import config

    # --- Setup Dependencies ---
    try:
        session = get_transkribus_session() # Uses config implicitly
        transkribus_api = TranskribusAPI(session, config.TRANSKRIBUS_BASE_URL, config.API_DELAY)
        image_handler = ImageHandler() # Uses default fonts from config implicitly if set up that way
        # Initialize LLM Clients
        llm_clients = {}
        if config.GEMINI_API_KEY:
            try:
                from paleo_workflow_kit.llm_clients.gemini_client import GeminiClient
                llm_clients["gemini"] = GeminiClient(config.GEMINI_API_KEY, config.GEMINI_MODEL_NAME, config.LLM_TEMPERATURE_GEMINI)
            except ImportError: logger.error("GeminiClient could not be imported.")
            except Exception as e: logger.error(f"Failed to initialize GeminiClient: {e}")

        if config.ANTHROPIC_API_KEY:
            try:
                from paleo_workflow_kit.llm_clients.anthropic_client import AnthropicClient
                llm_clients["anthropic"] = AnthropicClient(config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL_NAME, config.LLM_TEMPERATURE_ANTHROPIC)
            except ImportError: logger.error("AnthropicClient could not be imported.")
            except Exception as e: logger.error(f"Failed to initialize AnthropicClient: {e}")

        if "gemini" not in llm_clients or "anthropic" not in llm_clients:
             raise ValueError("Required LLM clients ('gemini', 'anthropic') could not be initialized.")

        # --- Instantiate and Run Workflow ---
        workflow = CorrectionWorkflow(
            config=vars(config), # Pass config as dict
            transkribus_api=transkribus_api,
            page_xml_handler_cls=PageXMLHandler,
            image_handler=image_handler,
            llm_clients=llm_clients
        )

        target_collection = config.COLL_ID_PRIMARY # Use the primary collection ID from config
        if not target_collection:
            raise ValueError("Target Collection ID (COLL_ID_PRIMARY) not set in configuration.")

        # Run the workflow
        final_summary = workflow.run(collection_id=target_collection, process_all=False) # Example: don't process already corrected

        logger.info("--- Direct Workflow Execution Finished ---")
        # logger.info(f"Final Summary:\n{json.dumps(final_summary, indent=2)}")

    except Exception as e:
        logger.critical(f"Error during direct execution setup or run: {e}", exc_info=True)