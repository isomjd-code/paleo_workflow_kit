# paleo_workflow_kit/exceptions.py

"""
Custom exception classes for the Paleo Workflow Kit package.

Using custom exceptions allows for more specific error handling and
distinguishes errors originating from this package from built-in Python errors
or errors from third-party libraries.
"""

class PaleoWorkflowError(Exception):
    """Base exception class for all errors raised by the paleo_workflow_kit package."""
    def __init__(self, message="An error occurred in the Paleo Workflow Kit."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message

# --- Configuration Errors ---

class ConfigurationError(PaleoWorkflowError):
    """Base class for configuration-related errors."""
    pass

class MissingCredentialsError(ConfigurationError):
    """Raised when required API credentials (keys, user/pass, cookie) are missing."""
    def __init__(self, service_name: str):
        super().__init__(f"Missing credentials for {service_name}. Check .env file or environment variables.")
        self.service_name = service_name

class InvalidConfigurationError(ConfigurationError):
    """Raised when a configuration value is invalid or missing."""
    def __init__(self, setting_name: str, message: str = "Invalid or missing configuration setting."):
        super().__init__(f"{message} Setting: '{setting_name}'")
        self.setting_name = setting_name

# --- Transkribus API Errors ---

class TranskribusAPIError(PaleoWorkflowError):
    """Base class for errors related to Transkribus API interactions."""
    def __init__(self, message="An error occurred while interacting with the Transkribus API.", status_code=None, response_text=None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text

    def __str__(self):
        msg = self.message
        if self.status_code:
            msg += f" (Status Code: {self.status_code})"
        if self.response_text:
            msg += f" Response Snippet: {self.response_text[:200]}..." # Show snippet
        return msg

class TranskribusAuthenticationError(TranskribusAPIError):
    """Raised when Transkribus authentication fails."""
    def __init__(self, message="Transkribus authentication failed. Check credentials or session cookie."):
        super().__init__(message, status_code=401) # Typically 401

class TranskribusResourceNotFoundError(TranskribusAPIError):
    """Raised when a requested Transkribus resource (collection, doc, page) is not found."""
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(f"Transkribus resource not found: {resource_type} with ID '{resource_id}'.", status_code=404)
        self.resource_type = resource_type
        self.resource_id = resource_id

class TranskribusPermissionError(TranskribusAPIError):
    """Raised when the user lacks permissions for a Transkribus operation."""
    def __init__(self, message="Permission denied for Transkribus operation."):
        super().__init__(message, status_code=403) # Typically 403

# --- PAGE XML Errors ---

class PageXMLProcessingError(PaleoWorkflowError):
    """Base class for errors during PAGE XML processing."""
    pass

class PageXMLParseError(PageXMLProcessingError):
    """Raised when PAGE XML content cannot be parsed."""
    def __init__(self, message="Failed to parse PAGE XML content.", original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception

class LineNotFoundError(PageXMLProcessingError):
    """Raised when a specific TextLine cannot be found by ID or index."""
    def __init__(self, line_identifier):
        super().__init__(f"Could not find TextLine with identifier: {line_identifier}")
        self.line_identifier = line_identifier

# --- Image Handling Errors ---

class ImageProcessingError(PaleoWorkflowError):
    """Raised for errors during image download, manipulation, or encoding."""
    pass

# --- LLM Client Errors ---

class LLMError(PaleoWorkflowError):
    """Base class for errors related to LLM interactions."""
    def __init__(self, message="An error occurred during LLM interaction.", llm_provider=None, original_exception=None):
        super().__init__(message)
        self.llm_provider = llm_provider
        self.original_exception = original_exception

    def __str__(self):
        msg = self.message
        if self.llm_provider:
            msg += f" (Provider: {self.llm_provider})"
        if self.original_exception:
             msg += f" Original Exception: {type(self.original_exception).__name__}"
        return msg

class LLMConfigurationError(LLMError):
    """Raised for configuration issues with an LLM client (e.g., missing API key)."""
    pass

class LLMGenerationError(LLMError):
    """Raised when the LLM fails to generate content or returns an error."""
    pass

class LLMResponseParseError(LLMError):
    """Raised when the LLM response cannot be parsed (e.g., invalid JSON)."""
    pass

# --- Workflow Errors ---

class WorkflowError(PaleoWorkflowError):
    """Base class for errors occurring during workflow execution."""
    pass

class WorkflowConfigurationError(WorkflowError):
    """Raised when a workflow receives invalid or insufficient configuration."""
    pass

class WorkflowExecutionError(WorkflowError):
    """Raised for general errors during the execution of workflow steps."""
    pass

# --- Cache Errors ---

class CacheError(PaleoWorkflowError):
    """Raised for errors related to loading or saving cache files."""
    pass

# --- Utility Errors ---
class TextProcessingError(PaleoWorkflowError):
    """Raised for errors during text utility functions (e.g., CER calculation)."""
    pass


# Example of how to raise a custom exception:
# if not api_key:
#     raise MissingCredentialsError("Anthropic")
#
# try:
#     # some API call
# except requests.exceptions.HTTPError as e:
#     if e.response.status_code == 404:
#         raise TranskribusResourceNotFoundError("Document", doc_id) from e
#     else:
#         raise TranskribusAPIError(f"API Error: {e}", status_code=e.response.status_code, response_text=e.response.text) from e