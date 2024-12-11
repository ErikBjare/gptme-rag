#!/usr/bin/env python3
"""Simple example of using gptme-rag for document search."""

import sys
from pathlib import Path

from gptme_rag.indexing.indexer import Indexer
from rich.console import Console

console = Console()


def main():
    """Run document search."""
    if len(sys.argv) < 2:
        console.print("Usage: python search.py 'your search query'")
        return 1

    try:
        # Get docs directory and query
        docs_dir = Path(__file__).parent / "docs"
        query = sys.argv[1]

        # Create indexer and index docs
        indexer = Indexer(persist_directory=Path("basic_index"))
        indexer.index_directory(docs_dir, glob_pattern="**/*.md")

        # Search
        documents, distances, _ = indexer.search(query, n_results=3)

        # Display results
        console.print(f"\nResults for: [cyan]{query}[/cyan]\n")
        for doc, score in zip(documents, distances):
            console.print(f"[green]{doc.metadata['filename']}[/green]")
            console.print(f"Score: {1 - score:.2f}")
            console.print(
                doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            )
            console.print()

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
