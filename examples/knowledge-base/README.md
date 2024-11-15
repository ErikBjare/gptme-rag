# Knowledge Base Search Example

This example demonstrates how to use gptme-rag to create a searchable knowledge base with live updates.

## Project Structure

```plaintext
knowledge-base/
├── content/
│   ├── development/        # Development guides
│   ├── troubleshooting/    # Problem-solving guides
│   └── best-practices/     # Best practices docs
├── search_kb.py           # Search script
└── watch_kb.py           # File watcher script
```

## Features Demonstrated

- Organizing structured documentation
- Live indexing with file watching
- Semantic search across categories
- Automatic reindexing on changes
- Cross-referenced documentation

## Example Documentation

This example includes a complete set of development documentation:

### Development Guides
- **setup.md**: Development environment setup
  - Prerequisites
  - Installation steps
  - Editor configuration
  - Verification steps

- **workflow.md**: Development workflow
  - Git workflow
  - Code review process
  - Testing requirements
  - Deployment process

### Best Practices
- **testing.md**: Testing standards
  - Test organization
  - Test types and patterns
  - Running tests
  - CI integration

- **coding-style.md**: Code standards
  - Python style guide
  - Git conventions
  - Project structure
  - Code quality tools

### Troubleshooting
- **common-issues.md**: Problem solving
  - Environment issues
  - Git problems
  - Testing issues
  - Deployment problems

## Usage

1. Install gptme-rag:
```bash
pip install gptme-rag
```

2. Start the file watcher:
```bash
python watch_kb.py
```

3. In another terminal, search the knowledge base:
```bash
# Basic search
python search_kb.py "your search query"

# Interactive mode
python search_kb.py --interactive

# Show full content
python search_kb.py --show-content "testing best practices"
```

## Example Queries

Try these queries to explore the knowledge base:

### Development Setup
- "How do I set up my development environment?"
- "What editor settings should I use?"
- "How to verify my setup is correct?"

### Workflow Questions
- "What's the process for creating a pull request?"
- "How should I name my branches?"
- "What are the commit message conventions?"

### Best Practices
- "Show me the testing requirements"
- "What are our Python coding standards?"
- "How should I organize my tests?"

### Troubleshooting
- "How to fix virtual environment issues?"
- "What to do when tests fail unexpectedly?"
- "How to handle deployment problems?"

## Extending the Knowledge Base

### Adding New Categories

1. Create a new directory in `content/`:
```bash
mkdir content/new-category
```

2. Add markdown files with consistent structure:
- Clear headings
- Code examples
- Cross-references
- Practical examples

### Document Organization

- Use clear, hierarchical headings
- Include practical examples
- Add cross-references between documents
- Maintain consistent formatting
- Update related documents

### Search Optimization

- Use descriptive headings
- Include common problem statements
- Add code examples
- Cross-reference related topics
- Use consistent terminology

## Customization

### Search Parameters

Modify `search_kb.py` to adjust:
```python
indexer = Indexer(
    chunk_size=500,     # Adjust chunk size
    chunk_overlap=50,   # Adjust overlap
)

documents, distances = indexer.search(
    query,
    n_results=5,        # Adjust number of results
    group_chunks=True,  # Toggle chunk grouping
)
```

### Watch Settings

Modify `watch_kb.py` to customize:
```python
FileWatcher(
    paths=[str(kb_dir)],
    pattern="**/*.md",              # Change file patterns
    ignore_patterns=[".git"],       # Update ignore patterns
)
```

## Learn More

- See the [gptme-rag documentation](https://github.com/ErikBjare/gptme-rag)
- Explore other examples in the `examples/` directory
- Check out the [file watching guide](https://gptme.org/docs/file-watching.html)
