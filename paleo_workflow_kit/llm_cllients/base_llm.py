# paleo_workflow_kit/llm_clients/base_llm.py

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Set

# Get logger for this module
logger = logging.getLogger(__name__)

class BaseLLMClient(ABC):
    """
    Abstract Base Class for Large Language Model clients.

    Defines a common interface for interacting with different LLM services
    for tasks relevant to paleographic workflows, such as transcription
    correction, indexing, and auditing.

    Concrete implementations (e.g., GeminiClient, AnthropicClient) should
    inherit from this class and implement the abstract methods.
    """

    def __init__(self, model_name: str, temperature: float = 0.1, **kwargs):
        """
        Initializes the base LLM client.

        Args:
            model_name (str): The specific model identifier for the LLM service.
            temperature (float): The sampling temperature for generation (controls randomness).
            **kwargs: Additional keyword arguments specific to the LLM service setup.
        """
        self.model_name = model_name
        self.temperature = temperature
        self._additional_config = kwargs # Store other config if needed
        logger.info(f"Initializing {self.__class__.__name__} for model: {self.model_name} with temp: {self.temperature}")

    @abstractmethod
    def generate_correction(
        self,
        image_input: Union[str, bytes, List[str], List[bytes]], # Path(s), bytes, or list of paths/bytes
        htr_lines: List[str],
        entities: str,
        prompt_template: str, # Allow passing the specific prompt
        run_identifier: str = "A", # To differentiate multiple runs if needed by implementation
        **kwargs
        ) -> Optional[Dict[str, Any]]:
        """
        Generates a corrected transcription based on image(s) and HTR text.

        Args:
            image_input: Input image(s). Can be a single path (str), single image bytes,
                         a list of paths (str), or a list of image bytes, depending on
                         what the specific LLM client implementation expects (e.g., full
                         image path for Gemini, list of segment paths/bytes for Claude).
            htr_lines: List of original HTR transcription lines.
            entities: String containing named entities for context.
            prompt_template: The base prompt string with placeholders (e.g.,
                             for htr_lines, entities). The implementation will fill these.
            run_identifier: A string (e.g., "A", "B", "C") to potentially differentiate
                            multiple calls for the same input, if the implementation needs it.
            **kwargs: Additional model-specific generation parameters (e.g., max_tokens).

        Returns:
            A dictionary representing the parsed JSON response from the LLM,
            containing the corrected transcription and potentially other analysis,
            or None if the generation or parsing failed.
            The exact structure depends on the prompt used.
        """
        pass

    @abstractmethod
    def generate_index(
        self,
        text_lines: List[str],
        doc_metadata: Dict[str, Any],
        prompt_template: str, # Allow passing the specific prompt
        **kwargs
        ) -> Optional[Dict[str, Any]]:
        """
        Generates index entries based on transcribed text lines and metadata.

        Args:
            text_lines: List of transcribed text lines from the page.
            doc_metadata: Dictionary containing metadata about the document/page
                          (e.g., roll, membrane, side, image_num).
            prompt_template: The base prompt string with placeholders.
            **kwargs: Additional model-specific generation parameters.

        Returns:
            A dictionary representing the parsed JSON response from the LLM,
            containing the list of index entries, or None on failure.
            Expected structure: {"index_entries": [ {...}, {...} ]}
        """
        pass

    @abstractmethod
    def generate_audit(
        self,
        image_input: Union[str, bytes], # Typically a single numbered image path/bytes
        lines_data: List[Dict[str, Any]], # List of dicts with 'htr_text', 'seq_index' etc.
        prompt_template: str, # Allow passing the specific prompt
        **kwargs
        ) -> Optional[Dict[str, Any]]:
        """
        Performs an audit based on an image and line data.

        Args:
            image_input: Path (str) or bytes of the image to be audited (usually with numbers/lines drawn).
            lines_data: List of dictionaries, each containing data about a line
                        (e.g., 'htr_text', 'seq_index').
            prompt_template: The base prompt string with placeholders.
            **kwargs: Additional model-specific generation parameters.

        Returns:
            A dictionary representing the parsed JSON response from the LLM,
            containing the audit results (e.g., certified lines), or None on failure.
            The exact structure depends on the prompt used (e.g., {"certified_lines": {"1": "...", "5": "..."}}).
        """
        pass

    # --- Optional Helper Methods (Can be implemented here or in subclasses) ---

    def _parse_json_response(self, raw_text: str, context: str = "LLM response") -> Optional[Dict[str, Any]]:
        """
        Attempts to parse a JSON object from the LLM's raw text response.
        Handles common issues like leading/trailing text or markdown fences.

        Args:
            raw_text: The raw text output from the LLM.
            context: A string describing the context (e.g., "correction response") for logging.

        Returns:
            The parsed dictionary, or None if parsing fails.
        """
        if not raw_text or not raw_text.strip():
            logger.error(f"Cannot parse JSON: Raw text from {context} is empty.")
            return None

        logger.debug(f"Attempting to parse JSON from {context}. Raw text (first 500 chars):\n{raw_text[:500]}...")

        # 1. Try finding JSON within markdown code fences (```json ... ```)
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
                logger.error(f"Could not find valid JSON object delimiters {{...}} in {context}.")
                logger.debug(f"Full raw text from {context}:\n{raw_text}")
                return None

        # 3. Attempt to parse the extracted string
        try:
            parsed_json = json.loads(json_string)
            if not isinstance(parsed_json, dict):
                 logger.error(f"Parsed JSON from {context} is not a dictionary (type: {type(parsed_json)}).")
                 return None
            logger.debug(f"Successfully parsed JSON from {context}.")
            return parsed_json
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse extracted JSON string from {context}: {json_err}")
            logger.error(f"Invalid JSON string was:\n{json_string}")
            return None
        except Exception as e:
             logger.error(f"Unexpected error during JSON parsing for {context}: {e}", exc_info=True)
             return None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(model='{self.model_name}', temp={self.temperature})>"