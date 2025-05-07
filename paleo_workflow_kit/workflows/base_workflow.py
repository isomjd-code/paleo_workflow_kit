# paleo_workflow_kit/workflows/base_workflow.py

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type

# Import necessary components from the package
# Use absolute imports assuming standard package structure
from paleo_workflow_kit.config import DRY_RUN # Example: Accessing a specific config value
from paleo_workflow_kit.transkribus_api import TranskribusAPI
from paleo_workflow_kit.page_xml_handler import PageXMLHandler
from paleo_workflow_kit.image_handler import ImageHandler
from paleo_workflow_kit.llm_clients.base_llm import BaseLLMClient
# Import custom exceptions if defined
# from paleo_workflow_kit.exceptions import WorkflowConfigurationError

# Get logger for this module
logger = logging.getLogger(__name__)

class BaseWorkflow(ABC):
    """
    Abstract Base Class for defining high-level processing workflows.

    Provides common initialization for dependencies (API clients, handlers),
    logging setup, basic state tracking (results, errors), and enforces
    the implementation of a `run` method and a `get_summary` method
    in concrete workflow subclasses.
    """

    def __init__(
        self,
        config: Dict[str, Any], # Pass the loaded config dictionary or object
        transkribus_api: TranskribusAPI,
        page_xml_handler_cls: Type[PageXMLHandler], # Pass the class itself
        image_handler: Optional[ImageHandler] = None,
        llm_clients: Optional[Dict[str, BaseLLMClient]] = None,
    ):
        """
        Initializes the base workflow.

        Args:
            config: A dictionary or object containing configuration settings.
            transkribus_api: An initialized TranskribusAPI client instance.
            page_xml_handler_cls: The PageXMLHandler class (not an instance).
            image_handler: An optional initialized ImageHandler instance.
            llm_clients: An optional dictionary mapping client names (e.g., "gemini", "anthropic")
                         to initialized LLM client instances inheriting from BaseLLMClient.
        """
        self.config = config
        self.transkribus_api = transkribus_api
        self.page_xml_handler_cls = page_xml_handler_cls
        self.image_handler = image_handler
        self.llm_clients = llm_clients if llm_clients else {} # Ensure it's a dict

        # Get logger specific to the subclass name
        self.logger = logging.getLogger(self.__class__.__name__)

        # Workflow state
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.errors: List[Dict[str, Any]] = [] # List to store error details
        self.results: Dict[str, Any] = {} # Dictionary to store workflow-specific results

        # Common configuration access
        self.dry_run = self.config.get('DRY_RUN', True) # Default to True if not in config
        if self.dry_run:
            self.logger.warning("--- WORKFLOW INITIALIZED IN DRY RUN MODE ---")

        self.logger.info(f"{self.__class__.__name__} initialized.")

    def _log_start(self):
        """Logs the start of the workflow execution."""
        self.start_time = time.time()
        self.logger.info(f"--- Starting Workflow: {self.__class__.__name__} ---")
        if self.dry_run:
            self.logger.warning("--- DRY RUN MODE ACTIVE ---")

    def _log_end(self):
        """Logs the end of the workflow execution and duration."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time if self.start_time else 0
        self.logger.info(f"--- Finished Workflow: {self.__class__.__name__} ---")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        summary = self.get_summary()
        self.logger.info("--- Workflow Summary ---")
        for key, value in summary.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("------------------------")


    def _add_error(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Adds an error record to the workflow's error list."""
        error_record = {"message": message, "details": details or {}}
        self.errors.append(error_record)
        self.logger.error(f"Workflow Error: {message} {f'(Details: {details})' if details else ''}")

    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        The main execution method for the workflow.

        Subclasses MUST implement this method to define the specific steps
        of the workflow. It should orchestrate calls to the API, handlers,
        and LLM clients.

        Args:
            *args: Workflow-specific positional arguments.
            **kwargs: Workflow-specific keyword arguments.

        Returns:
            A dictionary summarizing the results of the workflow execution.
            This dictionary should align with what get_summary() returns.
        """
        self._log_start()
        # --- Subclass implementation goes here ---
        # Example:
        # try:
        #     # ... workflow steps ...
        #     self.results['items_processed'] = ...
        # except Exception as e:
        #     self._add_error(f"Critical failure during run: {e}", {"exception": str(e)})
        # finally:
        #     self._log_end()
        #     return self.get_summary()
        pass

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """
        Returns a dictionary summarizing the results and status of the workflow.

        Subclasses MUST implement this method to provide a structured summary
        of the workflow's execution (e.g., items processed, successes, failures,
        errors encountered, duration).

        Returns:
            A dictionary containing summary statistics and information.
        """
        # --- Subclass implementation provides specific summary keys ---
        # Example base summary:
        # duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
        # base_summary = {
        #     "workflow_name": self.__class__.__name__,
        #     "start_time": self.start_time,
        #     "end_time": self.end_time,
        #     "duration_seconds": round(duration, 2),
        #     "error_count": len(self.errors),
        #     "dry_run": self.dry_run,
        #     # Add workflow-specific results from self.results
        #     **self.results
        # }
        # return base_summary
        pass

    # --- Optional: Convenience methods ---
    def get_llm_client(self, client_name: str) -> Optional[BaseLLMClient]:
        """Safely retrieves a configured LLM client by name."""
        client = self.llm_clients.get(client_name.lower())
        if client is None:
            self.logger.warning(f"LLM Client '{client_name}' not found or not configured.")
        return client

    def create_page_xml_handler(self, xml_bytes: bytes) -> Optional[PageXMLHandler]:
        """Creates a PageXMLHandler instance from bytes using the configured class."""
        try:
            return self.page_xml_handler_cls(xml_bytes)
        except Exception as e: # Catch PageXMLParseError or others
            self.logger.error(f"Failed to create PageXMLHandler instance: {e}")
            self._add_error("Failed to parse PAGE XML", {"exception": str(e)})
            return None