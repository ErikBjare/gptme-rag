# gptme-rag

RAG (Retrieval-Augmented Generation) implementation for gptme context management.

<p align="center">
  <a href="https://github.com/ErikBjare/gptme-rag/actions/workflows/test.yml">
    <img src="https://github.com/ErikBjare/gptme-rag/actions/workflows/test.yml/badge.svg" alt="Tests" />
  </a>
  <a href="https://pypi.org/project/gptme-rag/">
    <img src="https://img.shields.io/pypi/v/gptme-rag" alt="PyPI version" />
  </a>
  <a href="https://github.com/ErikBjare/gptme-rag/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/ErikBjare/gptme-rag" alt="License" />
  </a>
</p>

## Features

- 📚 Document indexing with ChromaDB
  - Fast and efficient vector storage
  - Semantic search capabilities
  - Persistent storage
- 🔍 Semantic search with embeddings
  - Relevance scoring
  - Token-aware context assembly
  - Clean output formatting
- 🛠️ CLI interface for testing and development
  - Index management
  - Search functionality
  - Context assembly

## Installation

```bash
# From PyPI (coming soon)
pip install gptme-rag

# From source
git clone https://github.com/ErikBjare/gptme-rag.git
cd gptme-rag
poetry install
```

## Usage

### Indexing Documents

```bash
# Index markdown files in a directory
poetry run python -m gptme_rag index /path/to/documents --pattern "**/*.md"

# Index with custom persist directory
poetry run python -m gptme_rag index /path/to/documents --persist-dir ./index
```

### Searching

```bash
# Basic search
poetry run python -m gptme_rag search "your query here"

# Advanced search with options
poetry run python -m gptme_rag search "your query" \
  --n-results 5 \
  --persist-dir ./index \
  --max-tokens 4000 \
  --show-context
```

### Example Output
```plaintext
Most Relevant Documents:

1. ARCHITECTURE.md (relevance: 0.82)
  The task system is designed to help track and manage work effectively across sessions. Components include task registry, status tracking, and journal integration.

2. TASKS.md (relevance: 0.75)
  Active tasks and their current status. Includes task categories, status indicators, and progress tracking.

3. docs/workflow.md (relevance: 0.65)
  Documentation about workflow integration and best practices for task management.

Full Context:
Total tokens: 1250
Documents included: 3
Truncated: False
```

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=gptme_rag
```

### Project Structure

```plaintext
gptme_rag/
├── __init__.py
├── cli.py               # CLI interface
├── indexing/           # Document indexing
│   ├── document.py    # Document model
│   └── indexer.py     # ChromaDB integration
├── query/             # Search functionality
│   └── context_assembler.py  # Context assembly
└── utils/             # Utility functions
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run tests and linting
6. Submit a pull request

## Integration with gptme

This package is designed to integrate with [gptme](https://github.com/ErikBjare/gptme) as a plugin, providing:

- Automatic context enhancement
- Semantic search across project files
- Knowledge base integration
- Smart context assembly

## License

MIT License. See [LICENSE](LICENSE) for details.
