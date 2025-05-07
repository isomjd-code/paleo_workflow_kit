# paleo_workflow_kit/paleo_workflow_kit/__init__.py

"""
Paleo Workflow Kit: A Python package for interacting with Transkribus
and leveraging LLMs for paleographic document processing workflows.

This package provides tools for:
- Authenticating and interacting with the Transkribus REST API.
- Parsing, manipulating, and analyzing PAGE XML documents.
- Processing document images (downloading, drawing, rotating, segmenting).
- Interacting with Large Language Models (LLMs) like Google Gemini and Anthropic Claude.
- Implementing common paleographic workflows (correction, indexing, cleanup, etc.).
"""

import logging

# --- Package Version ---
# Option 1: Read from config (if config is simple and safe to import early)
try:
    from .config import SCRIPT_VERSION as __version__
except ImportError:
    # Option 2: Hardcode as fallback or primary method
    __version__ = "0.1.0" # Or your desired starting version

# --- Expose Core Classes ---
from .transkribus_api import TranskribusAPI
from .page_xml_handler import PageXMLHandler
from .image_handler import ImageHandler

# --- Expose LLM Clients ---
# Assuming these files exist and define the respective classes
try:
    from .llm_clients.anthropic_client import AnthropicClient
except ImportError:
    AnthropicClient = None # Indicate it's not available if import fails
    logging.getLogger(__name__).warning("AnthropicClient not available. Install required dependencies or check module.")
try:
    from .llm_clients.gemini_client import GeminiClient
except ImportError:
    GeminiClient = None # Indicate it's not available if import fails
    logging.getLogger(__name__).warning("GeminiClient not available. Install required dependencies or check module.")

# --- Expose Workflow Classes (Add as they are implemented) ---
# Example:
# from .workflows.correction import CorrectionWorkflow
# from .workflows.indexing import IndexingWorkflow
# from .workflows.cleanup import CleanupWorkflow
# from .workflows.find_replace import FindReplaceWorkflow
# from .workflows.comparison import ComparisonWorkflow
# from .workflows.copy_page import CopyPageWorkflow
# from .workflows.check_unclear import CheckUnclearWorkflow
# from .workflows.auditor import AuditorWorkflow

# --- Expose Utilities ---
from .utils.logging_setup import setup_logging
from .utils.text_utils import calculate_cer # Expose if generally useful
# Expose cache functions if users might need direct access
from .utils.cache_manager import load_llm_cache, save_llm_cache, load_batch_cache, save_batch_cache, remove_from_batch_cache

# --- Expose Custom Exceptions ---
# Create a base exception in exceptions.py
# from .exceptions import PaleoWorkflowError
# Expose specific exceptions if needed by users
from .page_xml_handler import PageXMLParseError, LineNotFoundError
from .image_handler import ImageProcessingError

# --- Define __all__ for explicit public interface ---
# List all the names you want to be imported when a user does `from paleo_workflow_kit import *`
# It's good practice even if `import *` is discouraged.
__all__ = [
    # Core Classes
    "TranskribusAPI",
    "PageXMLHandler",
    "ImageHandler",
    # LLM Clients (check for None before adding)
    "AnthropicClient" if AnthropicClient else None,
    "GeminiClient" if GeminiClient else None,
    # Workflow Classes (add names here as implemented)
    # "CorrectionWorkflow",
    # "IndexingWorkflow",
    # "CleanupWorkflow",
    # "FindReplaceWorkflow",
    # "ComparisonWorkflow",
    # "CopyPageWorkflow",
    # "CheckUnclearWorkflow",
    # "AuditorWorkflow",
    # Utilities
    "setup_logging",
    "calculate_cer",
    "load_llm_cache",
    "save_llm_cache",
    "load_batch_cache",
    "save_batch_cache",
    "remove_from_batch_cache",
    # Exceptions
    # "PaleoWorkflowError", # Base exception
    "PageXMLParseError",
    "LineNotFoundError",
    "ImageProcessingError",
    # Metadata
    "__version__",
]
# Filter out None values that might occur if optional dependencies aren't installed
__all__ = [name for name in __all__ if name is not None]


# --- Optional: Initial Log Message ---
# Use sparingly, as it runs every time the package is imported.
_init_logger = logging.getLogger(__name__)
# Check if logging is already configured (e.g., by the user) to avoid basicConfig clash
if not logging.getLogger().hasHandlers():
     logging.basicConfig(level=logging.WARNING) # Basic config if none exists yet
_init_logger.info(f"Paleo Workflow Kit package ({__name__}) initialized (Version: {__version__})")