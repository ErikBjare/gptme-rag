import logging
import os
import signal
import sys
import time
from pathlib import Path

import click
from rich.console import Console

from .benchmark import RagBenchmark
from .indexing.indexer import Indexer
from .indexing.watcher import FileWatcher
from .query.context_assembler import ContextAssembler

console = Console()

# TODO: change this to a more appropriate location
default_persist_dir = Path.home() / ".cache" / "gptme" / "rag"


@click.group()
@click.option("--verbose/-v", is_flag=True, help="Enable verbose output")
def cli(verbose: bool):
    """RAG implementation for gptme context management."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s - %(name)s - %(message)s",
    )


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
        indexer = Indexer(persist_directory=persist_dir, enable_persist=True)
        console.print(f"Indexing files in {directory} with pattern {pattern}")

        # Index the files
        n_indexed = indexer.index_directory(directory, pattern)

        console.print(f"✅ Successfully indexed {n_indexed} files", style="green")
    except Exception as e:
        console.print(f"❌ Error indexing directory: {e}", style="red")


@cli.command()
@click.argument("query")
@click.argument("paths", nargs=-1, type=click.Path(path_type=Path))
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
    paths: list[Path],
    n_results: int,
    persist_dir: Path,
    max_tokens: int,
    show_context: bool,
):
    """Search the index and assemble context."""
    paths = [path.resolve() for path in paths]
    try:
        # Hide ChromaDB output during initialization and search
        with console.status("Initializing..."):
            # Temporarily redirect stdout to suppress ChromaDB output
            stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            try:
                indexer = Indexer(persist_directory=persist_dir, enable_persist=True)
                assembler = ContextAssembler(max_tokens=max_tokens)
                documents, distances = indexer.search(
                    query, n_results=n_results, paths=paths
                )
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
@click.option(
    "--ignore-patterns",
    "-i",
    multiple=True,
    default=[".git", "__pycache__", "*.pyc"],
    help="Glob patterns to ignore",
)
def watch(directory: Path, pattern: str, persist_dir: Path, ignore_patterns: list[str]):
    """Watch directory for changes and update index automatically."""
    try:
        indexer = Indexer(persist_directory=persist_dir)

        # Initial indexing
        console.print(f"Performing initial indexing of {directory}")
        with console.status("Indexing..."):
            indexer.index_directory(directory, pattern)

        console.print("Starting file watcher...")

        try:
            file_watcher = FileWatcher(
                indexer, [str(directory)], pattern, ignore_patterns
            )
            with file_watcher:
                console.print("Watching for changes. Press Ctrl+C to stop.")
                # Keep the main thread alive

                try:
                    signal.pause()
                except AttributeError:  # Windows doesn't have signal.pause
                    while True:
                        time.sleep(1)
        except KeyboardInterrupt:
            console.print("\nStopping file watcher...")

    except Exception as e:
        console.print(f"❌ Error watching directory: {e}", style="red")
        console.print_exception()


@cli.group()
def benchmark():
    """Run performance benchmarks."""
    pass


@benchmark.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--pattern", "-p", default="**/*.*", help="Glob pattern for files to benchmark"
)
@click.option(
    "--persist-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to persist the index",
)
def indexing(directory: Path, pattern: str, persist_dir: Path | None):
    """Benchmark document indexing performance."""

    benchmark = RagBenchmark(index_dir=persist_dir)

    with console.status("Running indexing benchmark..."):
        benchmark.run_indexing_benchmark(directory, pattern)

    benchmark.print_results()


@benchmark.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--queries",
    "-q",
    multiple=True,
    default=["test", "document", "example"],
    help="Queries to benchmark",
)
@click.option(
    "--n-results",
    "-n",
    default=5,
    help="Number of results per query",
)
@click.option(
    "--persist-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to persist the index",
)
def search_benchmark(
    directory: Path,
    queries: list[str],
    n_results: int,
    persist_dir: Path | None,
):
    """Benchmark search performance."""

    benchmark = RagBenchmark(index_dir=persist_dir)

    # First index the directory
    with console.status("Indexing documents..."):
        benchmark.run_indexing_benchmark(directory)

    # Then run search benchmark
    with console.status("Running search benchmark..."):
        benchmark.run_search_benchmark(list(queries), n_results)

    benchmark.print_results()


@benchmark.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--duration",
    "-d",
    default=5.0,
    help="Duration of the benchmark in seconds",
)
@click.option(
    "--updates-per-second",
    "-u",
    default=2.0,
    help="Number of updates per second",
)
@click.option(
    "--persist-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to persist the index",
)
def watch_perf(
    directory: Path,
    duration: float,
    updates_per_second: float,
    persist_dir: Path | None,
):
    """Benchmark file watching performance."""

    benchmark = RagBenchmark(index_dir=persist_dir)

    with console.status("Running file watching benchmark..."):
        benchmark.run_watch_benchmark(
            directory,
            duration=duration,
            updates_per_second=updates_per_second,
        )

    benchmark.print_results()


def main(args=None):
    """Entry point for the CLI."""
    return cli(args=args)


if __name__ == "__main__":
    main()
