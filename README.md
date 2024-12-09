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
  <br>
  <a href="https://github.com/ErikBjare/gptme">
    <img src="https://img.shields.io/badge/built%20using-gptme%20%F0%9F%A4%96-5151f5?style=flat" alt="Built using gptme" />
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
- 📄 Smart document processing
  - Streaming large file handling
  - Automatic document chunking
  - Configurable chunk size/overlap
  - Document reconstruction
- 👀 File watching and auto-indexing
  - Real-time index updates
  - Pattern-based file filtering
  - Efficient batch processing
  - Automatic persistence
- 🛠️ CLI interface for testing and development
  - Index management
  - Search functionality
  - Context assembly
  - File watching

## Installation

```bash
# Using pip
pip install gptme-rag

# Using pipx (recommended for CLI tools)
pipx install gptme-rag

# From source (for development)
git clone https://github.com/ErikBjare/gptme-rag.git
cd gptme-rag
poetry install
```

After installation, the `gptme-rag` command will be available in your terminal.

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

### File Watching

The watch command monitors directories for changes and automatically updates the index:

```bash
# Watch a directory with default settings
poetry run python -m gptme_rag watch /path/to/documents

# Watch with custom pattern and ignore rules
poetry run python -m gptme_rag watch /path/to/documents \
  --pattern "**/*.{md,py}" \
  --ignore-patterns "*.tmp" "*.log" \
  --persist-dir ./index
```

Features:
- 🔄 Real-time index updates
- 🎯 Pattern matching for file types
- 🚫 Configurable ignore patterns
- 🔋 Efficient batch processing
- 💾 Automatic persistence

The watcher will:
- Perform initial indexing of existing files
- Monitor for file changes (create/modify/delete/move)
- Update the index automatically
- Handle rapid changes efficiently with debouncing
- Continue running until interrupted (Ctrl+C)

### Performance Benchmarking

The benchmark commands help measure and optimize performance:

```bash
# Benchmark document indexing
poetry run python -m gptme_rag benchmark indexing /path/to/documents \
  --pattern "**/*.md" \
  --persist-dir ./benchmark_index

# Benchmark search performance
poetry run python -m gptme_rag benchmark search /path/to/documents \
  --queries "python" "documentation" "example" \
  --n-results 10

# Benchmark file watching
poetry run python -m gptme_rag benchmark watch-perf /path/to/documents \
  --duration 10 \
  --updates-per-second 5
```

Features:
- 📊 Comprehensive metrics
  - Operation duration
  - Memory usage
  - Throughput
  - Custom metrics per operation
- 🔬 Multiple benchmark types
  - Document indexing
  - Search operations
  - File watching
- 📈 Performance tracking
  - Memory efficiency
  - Processing speed
  - System resource usage

Example benchmark output:
```plaintext
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Operation      ┃ Duration(s) ┃ Memory(MB) ┃ Throughput ┃ Additional Metrics ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ indexing       │      0.523 │     15.42 │   19.12/s │ files: 10         │
│ search         │      0.128 │      5.67 │   23.44/s │ queries: 3        │
│ file_watching  │      5.012 │      8.91 │    4.99/s │ updates: 25       │
└────────────────┴────────────┴───────────┴───────────┴──────────────────┘
```

### Document Chunking

The indexer supports automatic document chunking for efficient processing of large files:

```bash
# Index with custom chunk settings
poetry run python -m gptme_rag index /path/to/documents \
  --chunk-size 1000 \
  --chunk-overlap 200

# Search with chunk grouping
poetry run python -m gptme_rag search "your query" \
  --group-chunks \
  --n-results 5
```

Features:
- 🔄 Streaming processing
  - Handles large files efficiently
  - Minimal memory usage
  - Progress reporting
- 📑 Smart chunking
  - Configurable chunk size
  - Overlapping chunks for context
  - Token-aware splitting
- 🔍 Enhanced search
  - Chunk-aware relevance
  - Result grouping by document
  - Full document reconstruction

Example Output:
```plaintext
Most Relevant Documents:

1. documentation.md#chunk2 (relevance: 0.85)
  Detailed section about configuration options, including chunk size and overlap settings.
  [Part of: documentation.md]

2. guide.md#chunk5 (relevance: 0.78)
  Example usage showing how to process large documents efficiently.
  [Part of: guide.md]

3. README.md#chunk1 (relevance: 0.72)
  Overview of the chunking system and its benefits for large document processing.
  [Part of: README.md]

Full Context:
Total tokens: 850
Documents included: 3 (from 3 source documents)
Truncated: False
```

The chunking system automatically:
- Splits large documents into manageable pieces
- Maintains context across chunk boundaries
- Groups related chunks in search results
- Provides document reconstruction when needed

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

### Releases

Releases are automated through GitHub Actions. The process is:
1. Update version in pyproject.toml
2. Commit the change: `git commit -am "chore: bump version to x.y.z"`
3. Create and push a tag: `git tag vx.y.z && git push origin master vx.y.z`
4. Create a GitHub release (can be done with `gh release create vx.y.z`)
5. The publish workflow will automatically:
   - Run tests
   - Build the package
   - Publish to PyPI

## Integration with gptme

This package is designed to integrate with [gptme](https://github.com/ErikBjare/gptme) as a plugin, providing:

- Automatic context enhancement
- Semantic search across project files
- Knowledge base integration
- Smart context assembly

## License

MIT License. See [LICENSE](LICENSE) for details.
