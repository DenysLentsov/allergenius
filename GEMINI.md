# Allergenius Project Context

## Project Overview
Allergenius is a Python-based CLI tool designed to detect food allergens in ingredient lists. It utilizes local LLMs via Ollama (`embeddinggemma` for vector embeddings and `gemma3` for text processing) to analyze text and highlight potential allergens.

## Architecture & Key Components
*   **`main.py`**: The core application logic.
    *   Parses `data.json` for ingredient data (specifically "QUID" fields).
    *   Generates and caches embeddings using `embeddinggemma`.
    *   Performs cosine similarity search to find relevant context.
    *   Queries `gemma3` with a system prompt to identify and highlight allergens.
*   **`data.json`**: (Ignored) The expected input data source containing a list of objects with "QUID" fields representing ingredient text.
*   **`embeddings/`**: (Ignored) Directory where generated embeddings are cached to speed up subsequent runs.

## Prerequisites & Dependencies
*   **Python 3.11**
*   **Ollama**: Must be running locally.
    *   Required models: `embeddinggemma`, `gemma3`.
*   **Dependencies**: Managed via `requirements.txt`. Key libraries include:
    *   `ollama`: For interacting with local LLMs.
    *   `numpy`: For vector calculations (cosine similarity).
    *   `tqdm`, `alive-progress`: For progress indicators.

## Setup & Usage
This project uses `uv` for modern Python package management.

### Initial Setup
1.  **Environment**: `uv venv --python 3.11`
2.  **Install Dependencies**: `uv pip install -r requirements.txt`
3.  **Pull Models**:
    ```bash
    ollama pull embeddinggemma
    ollama pull gemma3
    ```

### Running the Application
1.  Ensure `data.json` is present in the root directory.
2.  Run with:
    ```bash
    uv run main.py
    ```
3.  Enter an ingredient list when prompted.

## Development Conventions
*   **Style**: Follows standard Python (PEP 8) conventions.
*   **Embeddings**: Embeddings are cached in `embeddings.json` inside the `embeddings/` folder. Delete this folder to force regeneration.
*   **Input Data**: The parser specifically looks for a `data` list containing objects with a `QUID` key in `data.json`.
