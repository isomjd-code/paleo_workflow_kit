# paleo_workflow_kit/config.py

import os
import logging
import re
from dotenv import load_dotenv
from typing import Optional, List, Tuple

# --- Logger Setup ---
# Use a specific logger for configuration loading issues
config_logger = logging.getLogger(__name__)
# Basic configuration if no handlers are set (e.g., when run directly)
if not config_logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# --- Load Environment Variables ---
# Searches for a .env file in the current directory or parent directories
dotenv_path = load_dotenv()
if dotenv_path:
    config_logger.info(f"Loaded environment variables from: {dotenv_path}")
else:
    config_logger.warning("No .env file found. Relying on environment variables or defaults.")

# --- Helper Function for Boolean Environment Variables ---
def get_bool_env(var_name: str, default: bool = False) -> bool:
    """Gets a boolean value from environment variables."""
    value = os.getenv(var_name)
    if value is None:
        return default
    return value.lower() in ('true', '1', 't', 'yes', 'y')

# --- API Keys & Credentials ---
# Secrets - should primarily be loaded from .env
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
TRANKRIBUS_USERNAME: Optional[str] = os.getenv("TRANKRIBUS_USERNAME")
TRANKRIBUS_PASSWORD: Optional[str] = os.getenv("TRANKRIBUS_PASSWORD")
TRANKRIBUS_SESSION_COOKIE: Optional[str] = os.getenv("TRANKRIBUS_SESSION_COOKIE")

# --- Transkribus Configuration ---
TRANSKRIBUS_BASE_URL: str = os.getenv("TRANSKRIBUS_BASE_URL", "https://transkribus.eu/TrpServer/rest")
# Define potential collection IDs - specific workflows will choose which one to use
# Provide default placeholders or None if they MUST be set in .env
COLL_ID_PRIMARY: Optional[int] = int(os.getenv("COLL_ID_PRIMARY", 0)) or None # Example: Main processing collection
COLL_ID_SOURCE: Optional[int] = int(os.getenv("COLL_ID_SOURCE", 0)) or None   # Example: Source for copying/comparison
COLL_ID_TARGET: Optional[int] = int(os.getenv("COLL_ID_TARGET", 0)) or None   # Example: Target for copying/comparison

# --- LLM Configuration ---
# Model Names (provide sensible defaults)
GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest") # Updated default
ANTHROPIC_MODEL_NAME: str = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-haiku-20240307") # Default to Haiku (cheaper)

# LLM Parameters (provide defaults)
LLM_TEMPERATURE_GEMINI: float = float(os.getenv("LLM_TEMPERATURE_GEMINI", 0.05))
LLM_TEMPERATURE_ANTHROPIC: float = float(os.getenv("LLM_TEMPERATURE_ANTHROPIC", 0.1))
ANTHROPIC_MAX_TOKENS: int = int(os.getenv("ANTHROPIC_MAX_TOKENS", 4096)) # Max tokens for standard Claude response
ANTHROPIC_MAX_TOKENS_BATCH: int = int(os.getenv("ANTHROPIC_MAX_TOKENS_BATCH", 4096)) # Max tokens for Claude batch response

# --- Workflow Parameters ---
# Thresholds
CER_THRESHOLD: float = float(os.getenv("CER_THRESHOLD", 0.10))
LLM_HTR_CER_THRESHOLD: float = float(os.getenv("LLM_HTR_CER_THRESHOLD", 0.08)) # For correction workflow
IMAGE_NUMBER_THRESHOLD: int = int(os.getenv("IMAGE_NUMBER_THRESHOLD", 540)) # For duplicate cleanup

# Text Processing
MIN_LINE_LENGTH: int = int(os.getenv("MIN_LINE_LENGTH", 10)) # For short line deletion
MIN_WORD_LENGTH: int = int(os.getenv("MIN_WORD_LENGTH", 1)) # For confusion matrix
IGNORE_CASE: bool = get_bool_env("IGNORE_CASE", False) # For confusion matrix
REMOVE_PUNCTUATION: bool = get_bool_env("REMOVE_PUNCTUATION", False) # For confusion matrix

# Operational Controls
DRY_RUN: bool = get_bool_env("DRY_RUN", True) # Default to DRY RUN for safety
API_DELAY: float = float(os.getenv("API_DELAY", 0.5)) # Delay between API calls
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 10)) # For Claude chunking
BBOX_PADDING: int = int(os.getenv("BBOX_PADDING", 25)) # Default padding around text for cropping
CLEANUP_TEMP_FILES: bool = get_bool_env("CLEANUP_TEMP_FILES", True) # General flag for temp files
CLEANUP_SEGMENTED_IMAGES: bool = get_bool_env("CLEANUP_SEGMENTED_IMAGES", True) # Specific flag

# Batch Processing
BATCH_POLLING_INTERVAL_SECONDS: int = int(os.getenv("BATCH_POLLING_INTERVAL_SECONDS", 15))
BATCH_TIMEOUT_SECONDS: int = int(os.getenv("BATCH_TIMEOUT_SECONDS", 1800)) # 30 minutes

# Metadata/Versioning
SCRIPT_VERSION: str = os.getenv("SCRIPT_VERSION", "2.1") # Example version
METADATA_TAG_VALUE: str = os.getenv("METADATA_TAG_VALUE", f"CombinedCorrected_v{SCRIPT_VERSION}")

# --- File/Directory Paths ---
# Use relative paths from project root or absolute paths
# Consider using pathlib for better path handling
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent # Assumes config.py is in paleo_workflow_kit/
OUTPUT_DIR_BASE: Path = Path(os.getenv("OUTPUT_DIR_BASE", PROJECT_ROOT / "output"))
INDEX_OUTPUT_DIR: Path = Path(os.getenv("INDEX_OUTPUT_DIR", OUTPUT_DIR_BASE / "kb_index_json_output"))
CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", PROJECT_ROOT / ".cache"))
TEMP_DIR: Path = Path(os.getenv("TEMP_DIR", PROJECT_ROOT / "temp"))

# Specific file names/bases
LLM_CACHE_FILE: Path = CACHE_DIR / os.getenv("LLM_CACHE_FILE", "llm_cache.json")
BATCH_CACHE_FILE: Path = CACHE_DIR / os.getenv("BATCH_CACHE_FILE", "anthropic_batch_cache.json")
TEMP_XML_FILENAME_BASE: str = os.getenv("TEMP_XML_FILENAME_BASE", "temp_page")
OUTPUT_IMAGE_GEMINI_BASE: str = os.getenv("OUTPUT_IMAGE_GEMINI_BASE", "numbered_rotated_page_gemini")
OUTPUT_IMAGE_CLAUDE_BASE: str = os.getenv("OUTPUT_IMAGE_CLAUDE_BASE", "numbered_original_page_claude")
TEMP_CHUNK_IMAGE_DIR: Path = TEMP_DIR / os.getenv("TEMP_CHUNK_IMAGE_DIR_NAME", "temp_chunk_images")
OUTPUT_CSV_FILE: Path = OUTPUT_DIR_BASE / os.getenv("OUTPUT_CSV_FILENAME", "word_confusion_matrix.csv")
OUTPUT_EXCEL_FILE: Path = OUTPUT_DIR_BASE / os.getenv("OUTPUT_EXCEL_FILENAME", "report.xlsx")
COLLATED_TEXT_OUTPUT: Path = OUTPUT_DIR_BASE / os.getenv("COLLATED_TEXT_OUTPUT", "collated_output.txt")

# --- Patterns ---
# Regex pattern for parsing document titles (adjust as needed per collection)
# Example: "CP40_951_f_0425.JPG"
DOC_TITLE_REGEX_PATTERN: str = os.getenv("DOC_TITLE_REGEX_PATTERN", r"^(?P<roll>CP\d+)_(?P<membrane>\d+)_(?P<side>[fd])_(?P<image_num>\d+)\.JPG$")
DOC_TITLE_REGEX: re.Pattern = re.compile(DOC_TITLE_REGEX_PATTERN, re.IGNORECASE)

# --- External URLs ---
AALT_INDEX_URL: Optional[str] = os.getenv("AALT_INDEX_URL") # Allow it to be optional

# --- Tool Names for Comparison/Filtering ---
CLAUDE_TOOL_NAME_TO_FIND: str = os.getenv("CLAUDE_TOOL_NAME_TO_FIND", "Claude Correction Script v1.0")
PYLAIA_TOOL_PREFIX: str = os.getenv("PYLAIA_TOOL_PREFIX", "PyLaia")
VALIDATION_TOOL_PREFIX: str = os.getenv("VALIDATION_TOOL_PREFIX", "ValidationPageMove")

# --- Find/Replace Rules ---
# Load from a separate JSON file or define here? Defining here for simplicity now.
# Format: List[Tuple[str, str, bool]] -> (find, replace, case_sensitive)
# Consider moving to a separate file (e.g., find_replace_rules.json) if it gets large.
REPLACEMENT_RULES: List[Tuple[str, str, bool]] = [
    ("j", "i", True),
    ("v", "u", False),
    (" Et ", " & ", True),
    (" et ", " & ", True),
    ("&c'", "&c", True),
    (" quod ", " q'd ", True),
    (" Com'", "com'", True),
    (" domini ", " d'ni ", True),
    (" domino ", " d'no ", True),
    (" Iohannem ", " Ioh'em ", True),
    (" Iohannes ", " Ioh'es ", True),
    (" Ric'um ", " Ricardum ", True),
    (" Ric'us ", " Ricardus ", True),
    (" Will'm ", " Willelmum ", True),
    (" Willmum ", " Willelmum ", True), # Variant
    (" Will'mus ", " Willelmus ", True),
    (" Willus ", " Willelmus ", True), # Variant
    (" Thoma' ", " Thomam ", True),
    (" Cur' ", " Curia ", True),
    ("yoman'", "yoman", True),
    (" Husbandmannu'", " Husbandmannum ", True),
]

# --- Validation Checks ---
# Check for essential API keys
if not GOOGLE_API_KEY and not ANTHROPIC_API_KEY:
    config_logger.warning("Neither GOOGLE_API_KEY nor ANTHROPIC_API_KEY found in environment. LLM features will fail.")
elif not GOOGLE_API_KEY:
    config_logger.warning("GOOGLE_API_KEY not found in environment. Gemini features will fail.")
elif not ANTHROPIC_API_KEY:
    config_logger.warning("ANTHROPIC_API_KEY not found in environment. Anthropic features will fail.")

# Check for Transkribus credentials if cookie isn't set
if not TRANKRIBUS_SESSION_COOKIE and (not TRANKRIBUS_USERNAME or not TRANKRIBUS_PASSWORD):
    config_logger.error("Transkribus credentials (TRANKRIBUS_USERNAME/PASSWORD) or TRANKRIBUS_SESSION_COOKIE are required.")
    # Depending on usage, you might want to raise an error here
    # raise ValueError("Missing Transkribus credentials or session cookie.")

# Check if primary collection ID is set (often essential)
if not COLL_ID_PRIMARY:
    config_logger.warning("COLL_ID_PRIMARY is not set. Many workflows may require this.")

# --- Create Directories ---
# Ensure essential directories exist at import time
try:
    OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_CHUNK_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    config_logger.info("Ensured base output, cache, temp, and index directories exist.")
except OSError as e:
    config_logger.error(f"Error creating necessary directories: {e}")

config_logger.info("Configuration loaded.")