# Food Allergen Detection Assistant

An AI-powered tool that analyzes food ingredient lists and identifies potential allergens using embeddings similarity search and LLM processing.

## Models Used

- **nomic-embed-text**: For generating embeddings from ingredient lists
- **Llama 3.2**: For allergen detection and response generation

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed locally
- At least 8GB RAM recommended
- Windows/Linux/macOS supported

## Setup

1. Clone the repository:
```bash
git clone https://github.com/nutritics/allergens.git
cd allergens
```

2. Create and activate virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/macOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Pull required models using Ollama:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Usage
1. Prepare your ingredient data in JSON format (add data.json in root directory)
2. Run the main script:
```bash
python main.py
```
3. Enter ingredient list when prompted

## Features
- Generates and caches embeddings for efficient similarity search
- Uses semantic similarity to find relevant ingredient patterns
- Identifies common food allergens with LLM processing
- Highlights allergens in the text output

## Requirements
See requirements.txt for full dependency list:
- tqdm>=4.65.0
- numpy>=1.24.0
- ollama>=0.1.6
- alive-progress>=3.1.5

## Development
Embeddings are cached in embeddings directory for faster subsequent runs. Delete this directory to regenerate embeddings