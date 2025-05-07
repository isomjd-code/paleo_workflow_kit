# paleo_workflow_kit/llm_clients/__init__.py

"""
Subpackage for interacting with various Large Language Model (LLM) APIs.

Provides specific client implementations built upon a common base interface.
"""

import logging

# Get logger for this subpackage initialization
_init_logger = logging.getLogger(__name__)

# --- Expose the Base Class ---
# Makes it available for type hinting or direct use if needed
try:
    from .base_llm import BaseLLMClient
except ImportError as e:
    _init_logger.error(f"Could not import BaseLLMClient: {e}. Base functionality might be affected.")
    BaseLLMClient = None # Indicate unavailability

# --- Expose Concrete Client Implementations ---
# Use try-except blocks to handle optional dependencies gracefully

# Anthropic Claude Client
try:
    from .anthropic_client import AnthropicClient
except ImportError as e:
    # Log a warning if the specific client cannot be imported
    # This often happens if the required library (e.g., 'anthropic') is not installed
    _init_logger.warning(f"AnthropicClient not available. Install 'anthropic' package if needed. Error: {e}")
    AnthropicClient = None # Set to None so checks elsewhere work

# Google Gemini Client
try:
    from .gemini_client import GeminiClient
except ImportError as e:
    _init_logger.warning(f"GeminiClient not available. Install 'google-generativeai' package if needed. Error: {e}")
    GeminiClient = None # Set to None

# --- Add imports for other LLM clients here as they are created ---
# Example:
# try:
#     from .openai_client import OpenAIClient
# except ImportError:
#     OpenAIClient = None

# --- Define __all__ for the subpackage ---
# List only the names intended to be imported when using `from paleo_workflow_kit.llm_clients import *`
# Include the base class and successfully imported concrete clients.
__all__ = [
    "BaseLLMClient" if BaseLLMClient else None,
    "AnthropicClient" if AnthropicClient else None,
    "GeminiClient" if GeminiClient else None,
    # Add other client names here
    # "OpenAIClient" if OpenAIClient else None,
]
# Filter out None values from __all__
__all__ = [name for name in __all__ if name is not None]

_init_logger.debug(f"LLM Clients subpackage ({__name__}) initialized. Available clients: {', '.join(__all__)}")