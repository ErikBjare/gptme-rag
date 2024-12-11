#!/usr/bin/env python3
"""Knowledge base search script."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from gptme_rag.indexing.indexer import Indexer

console = Console()


def format_result(doc, score: float, show_content: bool = True) -> Panel:
    """Format a search result as a rich Panel.

    Args:
        doc: Document result
        score: Relevance score
        show_content: Whether to show full content

    Returns:
        Formatted panel
    """
    # Get source information
    source = doc.metadata.get("filename", "unknown")
    category = Path(source).parent.name

    # Create title with score and source
    title = f"[cyan]{source}[/cyan] [yellow](relevance: {score:.2f})[/yellow]"
    if category:
        title = f"[green]{category}[/green] > {title}"

    # Format content
    if show_content:
        content = doc.content
        # If it's markdown, render it
        if source.endswith(".md"):
            content = Markdown(content)
    else:
        # Show just a preview
        content = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content

    return Panel(content, title=title, border_style="blue")


@click.command()
@click.argument("query", required=False)
@click.option(
    "--index-dir",
    type=click.Path(path_type=Path),
    default="kb_index",
    help="Directory containing the search index",
)
@click.option(
    "--interactive",
    is_flag=True,
    help="Run in interactive mode",
)
@click.option(
    "--show-content",
    is_flag=True,
    help="Show full content of results",
)
def main(query: str | None, index_dir: Path, interactive: bool, show_content: bool):
    """Search the knowledge base.

    If no query is provided or --interactive is set, runs in interactive mode.
    """
    try:
        # Check if index exists
        if not index_dir.exists():
            console.print(
                "[red]Error: Index not found. Run watch_kb.py first to create the index."
            )
            return 1

        # Create indexer
        indexer = Indexer(persist_directory=index_dir)

        def do_search(search_query: str):
            """Perform search and display results."""
            # Search with chunk grouping
            documents, distances, _ = indexer.search(
                search_query,
                n_results=5,
                group_chunks=True,
            )

            if not documents:
                console.print("No results found.")
                return

            # Display results
            console.print(f"\nResults for: [cyan]{search_query}[/cyan]\n")
            for doc, distance in zip(documents, distances):
                # Convert distance to similarity score (0-1)
                score = 1 - distance
                panel = format_result(doc, score, show_content)
                console.print(panel)
                console.print()  # Add spacing between results

        if interactive or not query:
            # Interactive mode
            console.print("Search the knowledge base (Ctrl+C to exit)")
            console.print("Example queries:")
            console.print("  - How do I set up my development environment?")
            console.print("  - What's the workflow for creating a pull request?")
            console.print("  - Show me the testing requirements")

            while True:
                try:
                    query = input("\nEnter search query: ").strip()
                    if not query:
                        continue
                    do_search(query)
                except KeyboardInterrupt:
                    console.print("\nExiting...")
                    break
        else:
            # Single query mode
            do_search(query)

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
