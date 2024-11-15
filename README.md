# gptme-rag

RAG (Retrieval-Augmented Generation) implementation for gptme context management.

## Features

- Document indexing with ChromaDB
- Semantic search with embeddings
- Context assembly with token management
- CLI interface for testing and development

## Installation

```bash
# Clone the repository
git clone https://github.com/ErikBjare/gptme-rag.git
cd gptme-rag

# Install with poetry
poetry install
```

## Usage

### Indexing Documents

```bash
poetry run python -m gptme_rag index /path/to/documents --pattern "**/*.md"
```

### Searching

```bash
poetry run python -m gptme_rag search "your query here" --n-results 5
```

## Development

### Running Tests

```bash
poetry run pytest
```

### Project Structure
