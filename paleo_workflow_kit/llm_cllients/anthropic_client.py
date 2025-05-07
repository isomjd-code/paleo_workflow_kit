# paleo_workflow_kit/llm_clients/anthropic_client.py

import anthropic
import base64
import io
import json
import logging
import mimetypes
import os
import re
import time
from types import SimpleNamespace
from typing import List, Dict, Any, Optional, Union, Iterable

from PIL import Image

# Assuming BaseLLMClient is in the same directory level or accessible via package structure
from .base_llm import BaseLLMClient
# Import specific exceptions if defined, otherwise use standard ones
# from ..exceptions import LLMConfigurationError, LLMGenerationError

# Get logger for this module
logger = logging.getLogger(__name__)

# --- Default Values ---
DEFAULT_ANTHROPIC_MODEL = "claude-3-haiku-20240307"
DEFAULT_ANTHROPIC_TEMP = 0.1
DEFAULT_MAX_TOKENS = 4096 # Default for standard messages
DEFAULT_MAX_TOKENS_BATCH = 4096 # Default for batch messages
DEFAULT_BATCH_POLLING_INTERVAL = 15
DEFAULT_BATCH_TIMEOUT = 1800 # 30 minutes

class AnthropicClient(BaseLLMClient):
    """
    Client for interacting with the Anthropic Claude API.

    Handles standard message creation and batch processing for tasks like
    transcription correction, indexing, and auditing.
    """

    def __init__(self, api_key: str, model_name: str = DEFAULT_ANTHROPIC_MODEL, temperature: float = DEFAULT_ANTHROPIC_TEMP, **kwargs):
        """
        Initializes the Anthropic client.

        Args:
            api_key: Your Anthropic API key.
            model_name: The specific Claude model to use (e.g., "claude-3-opus-20240229").
            temperature: Default sampling temperature for generation.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(model_name, temperature, **kwargs)
        if not api_key:
            raise ValueError("Anthropic API key is required.")
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"Anthropic client initialized successfully for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            # raise LLMConfigurationError(f"Failed to initialize Anthropic client: {e}") from e # Use custom exception if defined
            raise RuntimeError(f"Failed to initialize Anthropic client: {e}") from e

    def _prepare_image_input(self, image_input: Union[str, bytes, Image.Image]) -> Optional[Dict[str, Any]]:
        """Encodes image data (path, bytes, or PIL Image) to base64 for Anthropic API."""
        try:
            image_data = None
            media_type = None

            if isinstance(image_input, str): # Path
                if not os.path.exists(image_input):
                    logger.error(f"Image file not found at path: {image_input}")
                    return None
                media_type, _ = mimetypes.guess_type(image_input)
                with open(image_input, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")
            elif isinstance(image_input, bytes):
                # Try to infer media type from bytes if possible (less reliable)
                # For now, assume JPEG or PNG are most likely
                try:
                    img_obj = Image.open(io.BytesIO(image_input))
                    fmt = img_obj.format
                    if fmt == "JPEG": media_type = "image/jpeg"
                    elif fmt == "PNG": media_type = "image/png"
                    elif fmt == "GIF": media_type = "image/gif"
                    elif fmt == "WEBP": media_type = "image/webp"
                    else: logger.warning("Could not reliably determine media type from bytes, defaulting to image/jpeg"); media_type = "image/jpeg"
                except Exception:
                     logger.warning("Could not open image bytes to determine type, defaulting to image/jpeg")
                     media_type = "image/jpeg"
                image_data = base64.b64encode(image_input).decode("utf-8")
            elif isinstance(image_input, Image.Image):
                buffered = io.BytesIO()
                save_format = "JPEG" # Default to JPEG for Claude
                img_to_save = image_input
                if img_to_save.mode == 'RGBA': # Convert RGBA for JPEG
                    bg = Image.new("RGB", img_to_save.size, (255, 255, 255))
                    try: bg.paste(img_to_save, mask=img_to_save.split()[3])
                    except IndexError: bg.paste(img_to_save)
                    img_to_save = bg
                elif img_to_save.mode != 'RGB': img_to_save = img_to_save.convert('RGB')

                img_to_save.save(buffered, format=save_format, quality=95)
                image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
                media_type = f"image/{save_format.lower()}"
            else:
                logger.error(f"Unsupported image_input type: {type(image_input)}")
                return None

            if not media_type:
                logger.warning(f"Could not determine media type for input, defaulting to image/jpeg.")
                media_type = "image/jpeg" # Default if guess failed

            if image_data:
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    }
                }
            else:
                return None
        except Exception as e:
            logger.error(f"Error preparing image input for Anthropic: {e}", exc_info=True)
            return None

    def _robust_json_parse(self, raw_text: str, context: str = "Anthropic response") -> Optional[Dict[str, Any]]:
        """Attempts to parse JSON, trying fixes for common issues."""
        if not raw_text or not raw_text.strip():
            logger.error(f"Cannot parse JSON: Raw text from {context} is empty.")
            return None

        logger.debug(f"Attempting to parse JSON from {context}. Raw text (first 500 chars):\n{raw_text[:500]}...")
        original_json_string = ""

        try:
            # 1. Try finding JSON within markdown code fences
            json_match_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL | re.IGNORECASE)
            if json_match_fence:
                json_string = json_match_fence.group(1)
                logger.debug(f"Extracted JSON block using markdown fence regex from {context}.")
            else:
                # 2. Fallback: Find the outermost curly braces
                json_start_index = raw_text.find('{')
                json_end_index = raw_text.rfind('}')
                if json_start_index != -1 and json_end_index != -1 and json_end_index >= json_start_index:
                    json_string = raw_text[json_start_index : json_end_index + 1]
                    logger.debug(f"Extracted JSON block using simple brace finding from {context}.")
                else:
                    raise ValueError("Could not find valid JSON object delimiters {...} in response.")

            original_json_string = json_string # Store for potential fixing

            # 3. Attempt 1: Try parsing directly
            parsed_json = json.loads(original_json_string)
            logger.debug(f"Successfully parsed original JSON from {context}.")

        except json.JSONDecodeError as e1:
            logger.warning(f"Failed to parse original JSON from {context} ({e1}). Attempting fixes.")
            fixed_json_string = original_json_string

            # Apply fixes (e.g., remove trailing commas)
            fixed_json_string = re.sub(r",\s*([}\]])", r"\1", fixed_json_string)
            # Add more regex fixes here if other common issues are identified

            if fixed_json_string != original_json_string:
                logger.debug(f"Attempting parse after applying fixes for {context}.")
                try:
                    parsed_json = json.loads(fixed_json_string)
                    logger.info(f"Successfully parsed JSON after fixes for {context}.")
                except json.JSONDecodeError as e2:
                    logger.error(f"Failed to parse JSON even after fixes for {context} ({e2}).")
                    logger.debug(f"JSON string after attempting fixes:\n{fixed_json_string}")
                    return None # Failed even after fixes
            else:
                logger.error(f"No fixes applied, original parse failed for {context}.")
                return None # Original parse failed, no fixes applied
        except ValueError as ve: # Catch delimiter finding error
             logger.error(f"Error finding JSON delimiters in {context}: {ve}")
             return None
        except Exception as e:
             logger.error(f"Unexpected error during JSON extraction/parsing for {context}: {e}", exc_info=True)
             return None

        # Final validation: Ensure it's a dictionary
        if not isinstance(parsed_json, dict):
            logger.error(f"Parsed JSON from {context} is not a dictionary (type: {type(parsed_json)}).")
            return None

        return parsed_json

    # --- Abstract Method Implementations ---

    def generate_correction(self, image_input: Union[str, bytes, List[str], List[bytes]], htr_lines: List[str], entities: str, prompt_template: str, run_identifier: str = "A", **kwargs) -> Optional[Dict[str, Any]]:
        """
        Generates correction using a standard (non-batch) Anthropic call.
        NOTE: For multi-run correction workflows, using the dedicated batch methods
              (create_batch, poll_batch_job, get_batch_results) is recommended for efficiency.
              This method is provided for potential single-shot correction tasks or compatibility.
        """
        logger.info(f"Anthropic: Generating single correction (Run {run_identifier})...")

        # Prepare image input (assuming single image for standard call)
        if isinstance(image_input, list):
            if len(image_input) == 1:
                image_input = image_input[0] # Use the first element if list provided
            else:
                logger.error("Anthropic standard generate_correction expects a single image input, not a list.")
                return None
        image_block = self._prepare_image_input(image_input)
        if not image_block:
            logger.error("Failed to prepare image input for Anthropic.")
            return None

        # Format prompt
        htr_str = "\n".join([f'  "{i+1}": "{line}"' for i, line in enumerate(htr_lines)])
        # Use the specific placeholders expected by the correction prompt
        full_prompt = prompt_template.replace("{named_entity_list}", entities).replace("{htr_xml_prompt_str}", htr_str) # Adjust placeholders if needed

        # Prepare message structure
        user_messages_content = [image_block, {"type": "text", "text": "Please transcribe the document based on the image and provided HTR/entities, following the detailed system prompt rules."}] # Simple user message
        system_prompt = kwargs.get("system_prompt", None) # Allow overriding system prompt via kwargs

        max_tokens = kwargs.get('max_tokens', DEFAULT_MAX_TOKENS)

        try:
            message_response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=self.temperature,
                system=system_prompt, # Pass system prompt if provided
                messages=[{"role": "user", "content": user_messages_content}]
            )

            # Process response
            if not message_response.content or not isinstance(message_response.content, list) or len(message_response.content) == 0:
                 raise ValueError("Claude response content is empty or invalid.")
            response_text = message_response.content[0].text

            # Parse JSON
            parsed_json = self._robust_json_parse(response_text, context=f"Anthropic correction (Run {run_identifier})")
            # Optional: Add validation specific to the correction output structure here
            return parsed_json

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error during correction: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic correction: {e}", exc_info=True)
            return None

    def generate_index(self, text_lines: List[str], doc_metadata: Dict[str, Any], prompt_template: str, **kwargs) -> Optional[Dict[str, Any]]:
        logger.info("Anthropic: Generating index...")
        # Format prompt
        page_text_input = "\n".join(f"{i+1}: {line}" for i, line in enumerate(text_lines))
        metadata_str = ", ".join(f"{k}: {v}" for k, v in doc_metadata.items())
        # Ensure placeholders match the indexing prompt template
        full_prompt = prompt_template.replace("{metadata_str_placeholder}", metadata_str).replace("{page_text_input_placeholder}", page_text_input)

        max_tokens = kwargs.get('max_tokens', DEFAULT_MAX_TOKENS)
        system_prompt = kwargs.get("system_prompt", None) # Allow passing system prompt if needed

        try:
            message_response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": full_prompt}] # No image needed
            )

            # Process response
            if not message_response.content or not isinstance(message_response.content, list) or len(message_response.content) == 0:
                 raise ValueError("Claude response content is empty or invalid.")
            response_text = message_response.content[0].text

            # Parse JSON
            parsed_json = self._robust_json_parse(response_text, context="Anthropic indexing")
            # Validate structure
            if parsed_json and isinstance(parsed_json.get("index_entries"), list):
                return parsed_json
            else:
                logger.error("Anthropic indexing response failed structure validation (missing 'index_entries' list).")
                logger.debug(f"Parsed JSON: {parsed_json}")
                return None

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error during indexing: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic indexing: {e}", exc_info=True)
            return None

    def generate_audit(self, image_input: Union[str, bytes], lines_data: List[Dict[str, Any]], prompt_template: str, **kwargs) -> Optional[Dict[str, Any]]:
        logger.info("Anthropic: Generating audit...")
        image_block = self._prepare_image_input(image_input)
        if not image_block:
            logger.error("Failed to prepare image input for Anthropic audit.")
            return None

        # Format prompt (needs placeholders for lines_data)
        # Example placeholder: {lines_data_placeholder}
        lines_data_str = json.dumps(lines_data, indent=2) # Simple JSON representation
        full_prompt = prompt_template.replace("{lines_data_placeholder}", lines_data_str)

        user_messages_content = [image_block, {"type": "text", "text": full_prompt}]
        system_prompt = kwargs.get("system_prompt", None)
        max_tokens = kwargs.get('max_tokens', DEFAULT_MAX_TOKENS)

        try:
            message_response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_messages_content}]
            )

            # Process response
            if not message_response.content or not isinstance(message_response.content, list) or len(message_response.content) == 0:
                 raise ValueError("Claude response content is empty or invalid.")
            response_text = message_response.content[0].text

            # Parse JSON
            parsed_json = self._robust_json_parse(response_text, context="Anthropic audit")
            # Optional: Validate audit structure (e.g., presence of "certified_lines")
            if parsed_json and isinstance(parsed_json.get("certified_lines"), dict):
                 return parsed_json
            else:
                 logger.error("Anthropic audit response failed structure validation (missing 'certified_lines' dict).")
                 logger.debug(f"Parsed JSON: {parsed_json}")
                 return None

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error during audit: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic audit: {e}", exc_info=True)
            return None

    # --- Batch API Methods ---

    def create_batch(self, requests: List[Request]) -> Optional[anthropic.types.Batch]:
        """
        Submits a batch job to the Anthropic API.

        Args:
            requests: A list of prepared batch request objects
                      (anthropic.types.messages.batch_create_params.Request).

        Returns:
            The batch job object if submission was successful, otherwise None.
        """
        if not requests:
            logger.error("Cannot create batch: No requests provided.")
            return None
        logger.info(f"Submitting batch job with {len(requests)} requests...")
        try:
            batch_job = self.client.messages.batches.create(requests=requests)
            logger.info(f"Batch submitted successfully. Batch ID: {batch_job.id}")
            return batch_job
        except anthropic.APIError as e:
            logger.error(f"Failed to submit batch job: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error submitting batch job: {e}", exc_info=True)
            return None

    def retrieve_batch(self, batch_id: str) -> Optional[anthropic.types.Batch]:
        """Retrieves the status and metadata of a specific batch job."""
        logger.debug(f"Retrieving status for batch ID: {batch_id}")
        try:
            batch_status = self.client.messages.batches.retrieve(batch_id)
            return batch_status
        except anthropic.NotFounderror:
             logger.error(f"Batch ID '{batch_id}' not found.")
             return None
        except anthropic.APIError as e:
            logger.error(f"API error retrieving batch '{batch_id}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving batch '{batch_id}': {e}", exc_info=True)
            return None

    def poll_batch_job(
        self,
        batch_id: str,
        timeout_seconds: int = DEFAULT_BATCH_TIMEOUT,
        interval_seconds: int = DEFAULT_BATCH_POLLING_INTERVAL
    ) -> Optional[anthropic.types.Batch]:
        """
        Polls the status of a batch job until it completes, fails, or times out.

        Args:
            batch_id: The ID of the batch job to poll.
            timeout_seconds: Maximum time to wait for completion.
            interval_seconds: How often to check the status.

        Returns:
            The final batch job object if it reached a terminal state within the timeout,
            otherwise None (if timed out or encountered critical errors).
        """
        logger.info(f"Polling batch {batch_id} (Timeout: {timeout_seconds}s, Interval: {interval_seconds}s)...")
        start_time = time.time()
        final_batch_object = None

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                logger.error(f"Batch {batch_id} timed out after {elapsed_time:.0f} seconds.")
                # Optionally try to cancel
                try: self.client.messages.batches.cancel(batch_id); logger.info(f"Attempted cancel on timeout for batch {batch_id}")
                except Exception as cancel_err: logger.warning(f"Could not cancel timed-out batch {batch_id}: {cancel_err}")
                return None # Timeout failure

            retrieved_batch = self.retrieve_batch(batch_id)

            if retrieved_batch is None:
                # Error occurred during retrieval (already logged by retrieve_batch)
                # Decide if this should be fatal or retryable. Let's assume fatal for now.
                logger.error(f"Failed to retrieve status for batch {batch_id}. Stopping polling.")
                return None

            current_status = retrieved_batch.processing_status
            logger.info(f"Batch {batch_id} status: {current_status} (Elapsed: {elapsed_time:.0f}s)")

            if current_status in ["completed", "ended", "failed", "cancelled", "expired"]:
                logger.info(f"Batch {batch_id} reached terminal status: {current_status}.")
                final_batch_object = retrieved_batch
                break

            time.sleep(interval_seconds)

        return final_batch_object # Return the final batch object (could be failed, completed, etc.)

    def get_batch_results(self, batch_id: str) -> Optional[Iterable[anthropic.types.BatchResult]]:
        """
        Retrieves the results for a completed batch job.

        Args:
            batch_id: The ID of the completed batch job.

        Returns:
            An iterable of BatchResult objects, or None if results cannot be fetched.
        """
        logger.info(f"Fetching results for completed batch {batch_id}...")
        try:
            # This returns a generator/iterable
            results_stream = self.client.messages.batches.results(batch_id)
            return results_stream
        except anthropic.APIError as e:
            logger.error(f"API error fetching results for batch {batch_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching results for batch {batch_id}: {e}", exc_info=True)
            return None