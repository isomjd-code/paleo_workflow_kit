# scripts/run_correction.py

"""
Command-line script to execute the Combined Correction Workflow.

This script initializes the necessary components (API clients, handlers)
based on configuration settings and runs the correction workflow on a
specified Transkribus collection.

Usage:
    python scripts/run_correction.py [-h] [--coll-id COLLECTION_ID] [--process-all] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--log-file LOG_FILE] [--dry-run]

Arguments:
    --coll-id COLLECTION_ID   (Optional) Transkribus Collection ID to process.
                               Overrides COLL_ID_PRIMARY in .env or config.
    --process-all             (Optional) Flag to process all documents, ignoring
                               the 'already corrected' metadata check.
    --log-level LEVEL         (Optional) Set console logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                               Default: INFO.
    --log-file FILE           (Optional) Path to a file for detailed logging (DEBUG level).
    --dry-run                 (Optional) Perform all steps except uploading changes to Transkribus.
"""

import logging
import argparse
import sys
import os
import json
from pathlib import Path

# --- Add package root to Python path ---
# This allows running the script from the project root directory (e.g., python scripts/run_correction.py)
# and ensures relative imports within the package work correctly.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))
# --- End Path Addition ---

# Import necessary components from the package
try:
    from paleo_workflow_kit import config # Load configuration first
    from paleo_workflow_kit.utils.logging_setup import setup_logging
    from paleo_workflow_kit.utils.auth import get_transkribus_session # Assuming auth moved here
    from paleo_workflow_kit.transkribus_api import TranskribusAPI
    from paleo_workflow_kit.page_xml_handler import PageXMLHandler
    from paleo_workflow_kit.image_handler import ImageHandler
    from paleo_workflow_kit.llm_clients import GeminiClient, AnthropicClient # Import specific clients
    from paleo_workflow_kit.workflows.correction import CorrectionWorkflow
except ImportError as e:
    print(f"Error importing package components: {e}", file=sys.stderr)
    print("Please ensure the package is installed correctly or the PYTHONPATH is set.", file=sys.stderr)
    sys.exit(1)

# Get a logger for this script
logger = logging.getLogger(__name__) # Use __name__ for the script's logger

def main():
    """Main function to parse arguments, set up, and run the workflow."""

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the Combined Correction Workflow for Transkribus documents.")
    parser.add_argument(
        "--coll-id",
        type=int,
        default=config.COLL_ID_PRIMARY, # Use default from config
        help=f"Transkribus Collection ID to process (default: {config.COLL_ID_PRIMARY} from config)."
    )
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Process all documents, ignoring the 'already corrected' metadata check."
    )
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Set the console logging level (default: INFO)."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None, # Default to no file logging unless specified
        help="Path to a file for detailed DEBUG logging."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform all steps except uploading changes to Transkribus."
    )
    args = parser.parse_args()

    # --- Determine effective dry_run setting ---
    # Command line flag overrides config/env var
    is_dry_run = args.dry_run or config.DRY_RUN

    # --- Logging Setup ---
    console_log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_file_path = args.log_file or config.TEMP_DIR / f"correction_workflow_{args.coll_id}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(
        console_level=console_log_level,
        file_level=logging.DEBUG, # Always log DEBUG to file if specified
        log_file=str(log_file_path) # Convert Path to string
    )

    logger.info("Starting Correction Workflow Script...")
    logger.info(f"Processing Collection ID: {args.coll_id}")
    logger.info(f"Process All Documents: {args.process_all}")
    logger.info(f"Dry Run Mode: {is_dry_run}")

    # --- Dependency Initialization ---
    try:
        logger.info("Initializing dependencies...")
        # 1. Transkribus Session
        # get_transkribus_session uses config implicitly
        session = get_transkribus_session()

        # 2. Transkribus API Client
        transkribus_api = TranskribusAPI(
            session=session,
            base_url=config.TRANSKRIBUS_BASE_URL,
            request_delay=config.API_DELAY
        )

        # 3. Image Handler
        image_handler = ImageHandler(
            default_font_path=config.DEFAULT_FONT_PATH, # Assuming these are in config
            fallback_font_path=config.FALLBACK_FONT_PATH,
            default_font_size=config.DEFAULT_FONT_SIZE
        )

        # 4. LLM Clients
        llm_clients = {}
        if config.GEMINI_API_KEY and GeminiClient:
            llm_clients["gemini"] = GeminiClient(
                api_key=config.GEMINI_API_KEY,
                model_name=config.GEMINI_MODEL_NAME,
                temperature=config.LLM_TEMPERATURE_GEMINI
            )
            logger.info("Gemini client initialized.")
        else:
            logger.warning("Gemini client not initialized (API key missing or import failed).")

        if config.ANTHROPIC_API_KEY and AnthropicClient:
            llm_clients["anthropic"] = AnthropicClient(
                api_key=config.ANTHROPIC_API_KEY,
                model_name=config.ANTHROPIC_MODEL_NAME,
                temperature=config.LLM_TEMPERATURE_ANTHROPIC
            )
            logger.info("Anthropic client initialized.")
        else:
            logger.warning("Anthropic client not initialized (API key missing or import failed).")

        if "gemini" not in llm_clients or "anthropic" not in llm_clients:
             logger.error("Both Gemini and Anthropic clients are required for the combined workflow but could not be initialized.")
             sys.exit(1)

        # 5. Workflow Configuration Dictionary
        # Pass relevant config values needed by the workflow
        workflow_config = {
            "DRY_RUN": is_dry_run, # Use the effective dry_run setting
            "CER_THRESHOLD": config.CER_THRESHOLD,
            "METADATA_TAG_VALUE": config.METADATA_TAG_VALUE,
            "SCRIPT_VERSION": config.SCRIPT_VERSION,
            "LLM_CACHE_FILE": str(config.LLM_CACHE_FILE), # Pass path as string
            "BATCH_CACHE_FILE": str(config.BATCH_CACHE_FILE),
            "TEMP_DIR": config.TEMP_DIR, # Pass Path object
            "TEMP_XML_FILENAME_BASE": config.TEMP_XML_FILENAME_BASE,
            "OUTPUT_IMAGE_GEMINI_BASE": config.OUTPUT_IMAGE_GEMINI_BASE,
            "OUTPUT_IMAGE_CLAUDE_BASE": config.OUTPUT_IMAGE_CLAUDE_BASE,
            "TEMP_CHUNK_IMAGE_DIR": config.TEMP_CHUNK_IMAGE_DIR,
            "CLEANUP_TEMP_FILES": config.CLEANUP_TEMP_FILES,
            "CLEANUP_SEGMENTED_IMAGES": config.CLEANUP_SEGMENTED_IMAGES,
            "AALT_INDEX_URL": config.AALT_INDEX_URL,
            # Add any other config values the workflow needs
        }

        # 6. Instantiate Workflow
        correction_workflow = CorrectionWorkflow(
            config=workflow_config,
            transkribus_api=transkribus_api,
            page_xml_handler_cls=PageXMLHandler, # Pass the class
            image_handler=image_handler,
            llm_clients=llm_clients
        )

        # --- Run Workflow ---
        logger.info("Executing Correction Workflow...")
        results = correction_workflow.run(
            collection_id=args.coll_id,
            process_all=args.process_all
        )

        logger.info("Workflow execution finished.")
        # Summary is logged by the workflow's _log_end method

    except (ValueError, ConnectionRefusedError, requests.exceptions.RequestException) as setup_err:
        logger.critical(f"Setup or Authentication Error: {setup_err}", exc_info=True)
        sys.exit(1)
    except ImportError as imp_err:
         logger.critical(f"Import Error: {imp_err}. Ensure package is installed correctly.", exc_info=True)
         sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()