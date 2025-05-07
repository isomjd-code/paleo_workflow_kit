# Paleo Workflow Kit

**Version:** 0.1.0 (Replace with actual version from `__init__.py`)

## Overview

Paleo Workflow Kit is a Python package designed to streamline paleographic document processing workflows, particularly those involving the [Transkribus](https://transkribus.eu/) platform and Large Language Models (LLMs) like Google Gemini and Anthropic Claude.

It provides a structured, object-oriented approach to common tasks such as:

*   Interacting with the Transkribus REST API (authentication, listing documents/pages, downloading/uploading PAGE XML).
*   Parsing and manipulating PAGE XML files.
*   Processing document images (downloading, drawing annotations, rotating, segmenting).
*   Leveraging LLMs for tasks like transcription correction, document indexing, and auditing.
*   Implementing reusable workflows for common paleographic processing pipelines.
*   Performing cleanup tasks (e.g., deleting duplicates, empty regions, short lines).
*   Executing find-and-replace operations across collections.

## Features

*   **Modular Design:** Separates concerns into distinct classes for API interaction, XML handling, image processing, and LLM clients.
*   **Configuration Driven:** Easily configure API keys, collection IDs, model names, thresholds, and paths using a `.env` file.
*   **Transkribus API Client:** Robust interaction with the Transkribus REST API, including pagination handling and error management.
*   **PAGE XML Handling:** Parse, query, and modify PAGE XML structures using `xml.etree.ElementTree`.
*   **Image Processing:** Download, draw baselines/numbers, rotate images and coordinates, segment images for LLM input using Pillow.
*   **LLM Integration:** Includes client implementations for:
    *   Google Gemini (via `google-generativeai`)
    *   Anthropic Claude (via `anthropic`, including Batch API support)
    *   Abstract base class (`BaseLLMClient`) for potential future extensions.
*   **Workflow Orchestration:** Provides a `BaseWorkflow` class and examples for building complex, multi-step processing pipelines (e.g., combined LLM correction, indexing).
*   **Utility Functions:** Includes helpers for logging setup, caching, CER calculation, etc.
*   **Error Handling:** Incorporates error handling and logging throughout the components.
*   **Command-Line Scripts:** Example scripts in the `scripts/` directory demonstrate how to run common workflows.

## Installation

1.  **Prerequisites:**
    *   Python 3.8+ recommended.
    *   `pip` (Python package installer).
    *   **Build Tools (Potentially Required):** The `python-Levenshtein` library requires C compiler tools.
        *   **Linux (Debian/Ubuntu):** `sudo apt-get update && sudo apt-get install build-essential python3-dev`
        *   **macOS:** Install Xcode Command Line Tools: `xcode-select --install`
        *   **Windows:** Install "Microsoft C++ Build Tools" from the Visual Studio Installer (select "Desktop development with C++").

2.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd paleo-workflow-kit # Or your project directory name
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate it:
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note:* This installs all dependencies, including optional ones like `pandas` and `beautifulsoup4`. If you only need specific LLMs or features, you can install packages individually (see `requirements.txt` for details). If `python-Levenshtein` fails, try installing build tools or consider alternatives mentioned in `requirements.txt`.

## Configuration

1.  **Copy the Example Environment File:**
    ```bash
    cp .env.example .env
    ```

2.  **Edit `.env`:** Open the `.env` file in a text editor and fill in your actual credentials and desired settings:
    *   **API Keys:** `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`.
    *   **Transkribus Credentials:** *Either* `TRANKRIBUS_USERNAME` and `TRANKRIBUS_PASSWORD` *or* `TRANKRIBUS_SESSION_COOKIE`. Do not provide both password and cookie.
    *   **Collection IDs:** Set `COLL_ID_PRIMARY` and others (`COLL_ID_SOURCE`, `COLL_ID_TARGET`) as needed for your workflows.
    *   **Other Settings (Optional):** Review and override defaults for model names, thresholds, paths, `DRY_RUN` mode, etc., if necessary.

    **IMPORTANT:** Add the `.env` file to your `.gitignore` to prevent committing secrets.