#!/usr/bin/env python3
"""Example script demonstrating documentation search with gptme-rag."""

import sys
from pathlib import Path

from gptme_rag.indexing.indexer import Indexer
from rich.console import Console
from rich.table import Table

console = Console()


def setup_index(docs_path: Path) -> Indexer:
    """Set up the search index for documentation.

    Args:
        docs_path: Path to the documentation directory

    Returns:
        Configured indexer
    """
    # Create indexer with chunking enabled for large files
    index_dir = Path("calculator_index")
    indexer = Indexer(
        persist_directory=index_dir,
        chunk_size=500,  # Smaller chunks for documentation
        chunk_overlap=50,  # Some overlap to maintain context
    )

    # Index both Python files and documentation
    console.print(f"Indexing documentation in {docs_path}")
    indexer.index_directory(docs_path, glob_pattern="**/*.md")
    indexer.index_directory(docs_path, glob_pattern="**/*.py")

    return indexer


def display_results(documents, distances):
    """Display search results in a formatted table."""
    table = Table(title="Search Results")
    table.add_column("Score", justify="right", style="cyan")
    table.add_column("Source", style="green")
    table.add_column("Content")

    for doc, score in zip(documents, distances):
        # Get source file and type
        source = doc.metadata.get("filename", "unknown")
        content = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content

        table.add_row(
            f"{1 - score:.2f}",  # Convert distance to similarity score
            source,
            content,
        )

    console.print(table)


def main():
    """Run the documentation search example."""
    # Get the calculator project root
    project_root = Path(__file__).parent / "calculator"

    try:
        # Set up the search index
        indexer = setup_index(project_root)

        # Interactive search loop
        console.print("\nSearch the calculator documentation (Ctrl+C to exit)")
        console.print("Example queries:")
        console.print("  - How do I use the memory functions?")
        console.print("  - What happens if I divide by zero?")
        console.print("  - Show me multiplication examples")

        while True:
            try:
                query = input("\nEnter search query: ").strip()
                if not query:
                    continue

                # Search with chunk grouping
                documents, distances = indexer.search(
                    query,
                    n_results=5,
                    group_chunks=True,
                )

                # Display results
                display_results(documents, distances)

            except KeyboardInterrupt:
                console.print("\nExiting...")
                break

    except Exception as e:
        console.print(f"[red]Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
