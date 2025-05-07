# paleo_workflow_kit/utils/logging_setup.py

import logging
import sys
import os
from typing import Optional

# --- Default Configuration ---
DEFAULT_CONSOLE_LEVEL = logging.INFO
DEFAULT_FILE_LEVEL = logging.DEBUG
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)-8s - %(name)-25s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Libraries known to be potentially verbose
DEFAULT_LIBRARIES_TO_QUIET = {
    "requests": logging.WARNING,
    "urllib3": logging.WARNING,
    "PIL": logging.INFO, # Pillow can be verbose at DEBUG
    "google.generativeai": logging.INFO, # Gemini client
    "anthropic": logging.INFO, # Anthropic client
    "matplotlib": logging.WARNING, # If used for plotting later
    "numexpr": logging.WARNING, # Often used by pandas/numpy
}

def setup_logging(
    console_level: int = DEFAULT_CONSOLE_LEVEL,
    file_level: int = DEFAULT_FILE_LEVEL,
    log_file: Optional[str] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    quiet_libraries: Optional[Dict[str, int]] = None
) -> None:
    """
    Configures the root logger for the application.

    Sets up handlers for console (stdout) and optionally a file.
    Clears existing handlers on the root logger to prevent duplication.
    Allows setting different levels for console and file output.
    Quiets overly verbose third-party libraries.

    Args:
        console_level: Logging level for console output (e.g., logging.INFO, logging.DEBUG).
        file_level: Logging level for file output (e.g., logging.DEBUG). Only used if log_file is provided.
        log_file: Optional path to a log file. If None, file logging is disabled.
        log_format: Format string for log messages.
        date_format: Format string for the date/time part of log messages.
        quiet_libraries: A dictionary mapping library names (str) to their desired logging level (int).
                         Defaults to DEFAULT_LIBRARIES_TO_QUIET. Pass an empty dict {} to disable quieting.
    """
    root_logger = logging.getLogger()

    # Determine the lowest level needed for the root logger
    # It needs to be at least as low as the lowest level of its handlers
    overall_level = console_level
    if log_file:
        overall_level = min(console_level, file_level)
    root_logger.setLevel(overall_level)

    # Remove existing handlers to prevent duplicate messages
    # This is important if this function is called multiple times or if basicConfig was called before.
    if root_logger.hasHandlers():
        # Use slicing `[:]` to iterate over a copy, allowing safe removal
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # --- File Handler (Optional) ---
    if log_file:
        try:
            # Ensure the directory for the log file exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                root_logger.info(f"Created log directory: {log_dir}")

            # Create file handler
            # Use 'a' mode to append to existing log files between runs if desired,
            # or 'w' to overwrite each time. 'a' is often safer for long processes.
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            # Log confirmation *after* adding the handler so it goes to the file too
            root_logger.info(f"Logging to file: {log_file} (Level: {logging.getLevelName(file_level)})")
        except (OSError, IOError) as e:
            # Log error to console if file handler setup fails
            root_logger.error(f"Failed to create or open log file '{log_file}': {e}. File logging disabled.")
            log_file = None # Ensure we know file logging isn't active

    # --- Quiet Noisy Libraries ---
    libraries_config = quiet_libraries if quiet_libraries is not None else DEFAULT_LIBRARIES_TO_QUIET
    if libraries_config:
        root_logger.debug("Quieting specified libraries...")
        for lib_name, level in libraries_config.items():
            try:
                logging.getLogger(lib_name).setLevel(level)
                root_logger.debug(f"  Set '{lib_name}' logger level to {logging.getLevelName(level)}")
            except Exception as e:
                root_logger.warning(f"  Could not set level for library '{lib_name}': {e}")

    # --- Final Confirmation Log ---
    file_log_status = f"and file '{log_file}' (Level: {logging.getLevelName(file_level)})" if log_file else "(File logging disabled)"
    root_logger.info(f"Logging configured. Root level: {logging.getLevelName(overall_level)}, Console level: {logging.getLevelName(console_level)} {file_log_status}")

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    print("--- Testing Logging Setup ---")

    # Example 1: Default setup (INFO to console, no file)
    print("\n--- Test 1: Default Setup ---")
    setup_logging()
    logging.debug("This DEBUG message should NOT appear on console.")
    logging.info("This INFO message should appear on console.")
    logging.warning("This WARNING message should appear on console.")

    # Example 2: DEBUG to console, DEBUG to file
    print("\n--- Test 2: DEBUG Console, DEBUG File ---")
    test_log_file = "test_logging_debug.log"
    if os.path.exists(test_log_file): os.remove(test_log_file) # Clean up previous test
    setup_logging(console_level=logging.DEBUG, file_level=logging.DEBUG, log_file=test_log_file)
    logging.debug("This DEBUG message should appear on console AND in file.")
    logging.info("This INFO message should appear on console AND in file.")
    logging.warning("This WARNING message should appear on console AND in file.")
    print(f"Check '{test_log_file}' for DEBUG level output.")

    # Example 3: INFO to console, WARNING to file
    print("\n--- Test 3: INFO Console, WARNING File ---")
    test_log_file_warn = "test_logging_warn.log"
    if os.path.exists(test_log_file_warn): os.remove(test_log_file_warn)
    setup_logging(console_level=logging.INFO, file_level=logging.WARNING, log_file=test_log_file_warn)
    logging.debug("This DEBUG message should NOT appear anywhere.")
    logging.info("This INFO message should appear ONLY on console.")
    logging.warning("This WARNING message should appear on console AND in file.")
    print(f"Check '{test_log_file_warn}' for WARNING level output.")

    # Example 4: Disable library quieting
    print("\n--- Test 4: Disable Library Quieting ---")
    setup_logging(console_level=logging.DEBUG, quiet_libraries={}) # Pass empty dict
    logging.getLogger("requests").debug("This requests DEBUG message might appear now.")
    logging.info("This INFO message should appear.")

    print("\n--- Logging Setup Test Complete ---")