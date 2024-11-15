import os
import sys
from pathlib import Path

import click
from rich.console import Console

from .indexing.indexer import Indexer
from .query.context_assembler import ContextAssembler

console = Console()

# TODO: change this to a more appropriate location
default_persist_dir = Path(__file__).parent / "data"


@click.group()
def cli():
    """RAG implementation for gptme context management."""
    pass


@cli.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--pattern", "-p", default="**/*.*", help="Glob pattern for files to index"
)
@click.option(
    "--persist-dir",
    type=click.Path(path_type=Path),
    default=default_persist_dir,
    help="Directory to persist the index",
)
def index(directory: Path, pattern: str, persist_dir: Path):
    """Index documents in a directory."""
    try:
        indexer = Indexer(persist_directory=persist_dir)
        console.print(f"Indexing files in {directory} with pattern {pattern}")

        # List files that will be indexed
        files = list(directory.glob(pattern))
        console.print(f"Found {len(files)} files:")
        for file in files:
            console.print(f"  - {file}")

        # Index the files
        with console.status(f"Indexing {len(files)} files..."):
            indexer.index_directory(directory, pattern)

        console.print(f"✅ Successfully indexed {len(files)} files", style="green")
    except Exception as e:
        console.print(f"❌ Error indexing directory: {e}", style="red")


@cli.command()
@click.argument("query")
@click.option("--n-results", "-n", default=5, help="Number of results to return")
@click.option(
    "--persist-dir",
    type=click.Path(path_type=Path),
    default=default_persist_dir,
    help="Directory to persist the index",
)
@click.option("--max-tokens", default=4000, help="Maximum tokens in context window")
@click.option("--show-context", is_flag=True, help="Show the full context content")
def search(
    query: str,
    n_results: int,
    persist_dir: Path,
    max_tokens: int,
    show_context: bool,
):
    """Search the index and assemble context."""
    try:
        # Hide ChromaDB output during initialization and search
        with console.status("Initializing..."):
            # Temporarily redirect stdout to suppress ChromaDB output
            stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            try:
                indexer = Indexer(persist_directory=persist_dir)
                assembler = ContextAssembler(max_tokens=max_tokens)
                documents, distances = indexer.search(query, n_results=n_results)
            finally:
                sys.stdout.close()
                sys.stdout = stdout

        # Show a summary of the most relevant documents
        console.print("\n[bold]Most Relevant Documents:[/bold]")
        for i, doc in enumerate(documents):
            filename = doc.metadata.get("filename", "unknown")
            distance = distances[i]
            relevance = 1 - distance  # Convert distance to similarity score

            # Show document header with relevance score
            console.print(
                f"\n[cyan]{i+1}. {filename}[/cyan] [yellow](relevance: {relevance:.2f})[/yellow]"
            )

            # Extract first meaningful section (after headers)
            lines = doc.content.split("\n")
            content = []
            for line in lines:
                if line.strip() and not line.startswith("#"):
                    content.append(line.strip())
                    if len(" ".join(content)) > 200:
                        break

            # Show the first paragraph or meaningful content
            content_preview = (
                " ".join(content)[:200] + "..."
                if len(" ".join(content)) > 200
                else " ".join(content)
            )
            console.print(f"  {content_preview}")

        # Assemble context window
        context = assembler.assemble_context(documents, user_query=query)

        # Show assembled context
        console.print("\n[bold]Full Context:[/bold]")
        console.print(f"Total tokens: {context.total_tokens}")
        console.print(f"Documents included: {len(context.documents)}")
        console.print(f"Truncated: {context.truncated}")

        if click.get_current_context().params.get("show_context", False):
            console.print("\n[bold]Context Content:[/bold]")
            console.print(context.content)

    except Exception:
        console.print("❌ Error searching index:", style="red")
        console.print_exception()


def main(args=None):
    """Entry point for the CLI."""
    return cli(args=args)


if __name__ == "__main__":
    main()
