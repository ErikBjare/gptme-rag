# Code Documentation Search Example

This example demonstrates how to use gptme-rag to create a searchable index of your project's documentation and code.

## Project Structure

```plaintext
calculator/
├── calculator.py     # Example library with docstrings
└── docs/
    ├── usage.md     # Usage guide and examples
    └── api.md       # API reference documentation
```

## Features Demonstrated

- Indexing both Python source code and Markdown documentation
- Handling different types of documentation:
  - Docstrings from source code
  - Usage guides and examples
  - API reference documentation
- Smart chunking for better search results
- Result grouping by document source

## Usage

1. Install gptme-rag:
```bash
pip install gptme-rag
```

2. Run the search script:
```bash
python search_docs.py
```

3. Try example queries:
- "How do I use the memory functions?"
- "What happens if I divide by zero?"
- "Show me multiplication examples"

## Key Concepts

1. **Mixed Source Indexing**
   - Indexes both `.py` and `.md` files
   - Preserves code context from docstrings
   - Maintains markdown formatting

2. **Smart Chunking**
   - Splits large documents into manageable pieces
   - Maintains context across chunks
   - Configurable chunk size and overlap

3. **Result Grouping**
   - Groups chunks from the same source
   - Shows most relevant section
   - Preserves document context

## Customization

The example can be customized by modifying `search_docs.py`:

```python
# Adjust chunking parameters
indexer = Indexer(
    persist_directory=index_dir,
    chunk_size=500,    # Change chunk size
    chunk_overlap=50,  # Change overlap
)

# Change file patterns
indexer.index_directory(docs_path, glob_pattern="**/*.md")  # Documentation
indexer.index_directory(docs_path, glob_pattern="**/*.py")  # Source code
```

## Extending

This example can be extended to:
1. Index additional file types
2. Customize result formatting
3. Add filtering by document type
4. Implement advanced search features

## Learn More

- See the [gptme-rag documentation](https://github.com/ErikBjare/gptme-rag) for more details
- Explore other examples in the `examples/` directory
- Check out the [API reference](https://gptme.org/docs/) for advanced usage
