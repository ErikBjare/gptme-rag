import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.syntax import Syntax
from tqdm import tqdm

from .benchmark import RagBenchmark
from .indexing.indexer import Indexer
from .indexing.watcher import FileWatcher
from .query.context_assembler import ContextAssembler

console = Console()
logger = logging.getLogger(__name__)

# TODO: change this to a more appropriate location
default_persist_dir = Path.home() / ".cache" / "gptme" / "rag"


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def cli(verbose: bool):
    """RAG implementation for gptme context management."""
    handler = RichHandler()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
    )


@cli.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--pattern", "-p", default="**/*.*", help="Glob pattern for files to index"
)
@click.option(
    "--persist-dir",
    type=click.Path(path_type=Path),
    default=default_persist_dir,
    help="Directory to persist the index",
)
def index(paths: list[Path], pattern: str, persist_dir: Path):
    """Index documents in one or more directories."""
    if not paths:
        console.print("❌ No paths provided", style="red")
        return

    try:
        indexer = Indexer(persist_directory=persist_dir, enable_persist=True)

        # Get existing files and their metadata from the index, using absolute paths
        existing_docs = indexer.get_all_documents()
        logger.debug("Found %d existing documents in index", len(existing_docs))

        existing_files = {}
        for doc in existing_docs:
            if "source" in doc.metadata:
                abs_path = os.path.abspath(doc.metadata["source"])
                last_modified = doc.metadata.get("last_modified")
                if last_modified:
                    try:
                        # Parse ISO format timestamp to float
                        existing_files[abs_path] = datetime.fromisoformat(
                            last_modified
                        ).timestamp()
                    except ValueError:
                        logger.warning(
                            "Invalid last_modified format: %s", last_modified
                        )
                        existing_files[abs_path] = 0
                else:
                    existing_files[abs_path] = 0
                # logger.debug("Existing file: %s", abs_path)  # Too spammy

        logger.debug("Loaded %d existing files from index", len(existing_files))

        # First, collect all documents and filter for new/modified
        all_documents = []
        with console.status("Collecting documents...") as status:
            for path in paths:
                if path.is_file():
                    status.update(f"Processing file: {path}")
                else:
                    status.update(f"Processing directory: {path}")

                documents = indexer.collect_documents(path)

                # Filter for new or modified documents
                filtered_documents = []
                for doc in documents:
                    source = doc.metadata.get("source")
                    if source:
                        # Resolve to absolute path for consistent comparison
                        abs_source = os.path.abspath(source)
                        doc.metadata["source"] = abs_source
                        current_mtime = os.path.getmtime(abs_source)

                        # Include if file is new or modified
                        if abs_source not in existing_files:
                            logger.debug("New file: %s", abs_source)
                            filtered_documents.append(doc)
                        # Round to microseconds (6 decimal places) for comparison
                        elif round(current_mtime, 6) > round(
                            existing_files[abs_source], 6
                        ):
                            logger.debug(
                                "Modified file: %s (current: %s, stored: %s)",
                                abs_source,
                                current_mtime,
                                existing_files[abs_source],
                            )
                            filtered_documents.append(doc)
                        else:
                            logger.debug("Unchanged file: %s", abs_source)

                all_documents.extend(filtered_documents)

        if not all_documents:
            console.print("No new or modified documents to index", style="yellow")
            return

        # Then process them with a progress bar
        n_files = len(set(doc.metadata.get("source", "") for doc in all_documents))
        n_chunks = len(all_documents)

        logger.info(f"Found {n_files} new/modified files to index ({n_chunks} chunks)")

        with tqdm(
            total=n_chunks,
            desc="Indexing documents",
            unit="chunk",
            disable=not sys.stdout.isatty(),
        ) as pbar:
            for progress in indexer.add_documents_progress(all_documents):
                pbar.update(progress)

        console.print(
            f"✅ Successfully indexed {n_files} files ({n_chunks} chunks)",
            style="green",
        )
    except Exception as e:
        console.print(f"❌ Error indexing directory: {e}", style="red")
        if logger.isEnabledFor(logging.DEBUG):
            console.print_exception()


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
@click.option("--raw", is_flag=True, help="Skip syntax highlighting")
@click.option("--explain", is_flag=True, help="Show scoring explanations")
@click.option(
    "--weights",
    type=click.STRING,
    help="Custom scoring weights as JSON string, e.g. '{\"recency_boost\": 0.3}'",
)
def search(
    query: str,
    paths: list[Path],
    n_results: int,
    persist_dir: Path,
    max_tokens: int,
    show_context: bool,
    raw: bool,
    explain: bool,
    weights: str | None,
):
    """Search the index and assemble context."""
    paths = [path.resolve() for path in paths]

    # Hide ChromaDB output during initialization and search
    with console.status("Initializing..."):
        # Parse custom weights if provided
        scoring_weights = None
        if weights:
            try:
                scoring_weights = json.loads(weights)
            except json.JSONDecodeError as e:
                console.print(f"❌ Invalid weights JSON: {e}", style="red")
                return
            except Exception as e:
                console.print(f"❌ Error parsing weights: {e}", style="red")
                return

        # Temporarily redirect stdout to suppress ChromaDB output
        stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            indexer = Indexer(
                persist_directory=persist_dir,
                enable_persist=True,
                scoring_weights=scoring_weights,
            )
            assembler = ContextAssembler(max_tokens=max_tokens)
            if explain:
                documents, distances, explanations = indexer.search(
                    query, n_results=n_results, paths=paths, explain=True
                )
            else:
                documents, distances, _ = indexer.search(
                    query, n_results=n_results, paths=paths
                )
        finally:
            sys.stdout.close()
            sys.stdout = stdout

    # Assemble context window
    context = assembler.assemble_context(documents, user_query=query)

    # if show_context, just print file contents of all matches
    if show_context:
        for doc in context.documents:
            # Display file with syntax highlighting
            lexer = doc.metadata.get("extension", "").lstrip(".") or "text"
            output = doc.format_xml()
            console.print(
                output
                if raw
                else Syntax(
                    output,
                    lexer,
                    theme="monokai",
                    word_wrap=True,
                )
            )
            console.print()
        return

    # Show a summary of the most relevant documents
    console.print("\n[bold]Most Relevant Documents:[/bold]")
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "unknown")
        distance = distances[i]

        # Show document header
        console.print(f"\n[cyan]{i+1}. {source}[/cyan]")

        # Show scoring explanation if requested
        if explain and explanations:  # Make sure explanations is not None
            explanation = explanations[i]
            console.print("\n[bold]Scoring Breakdown:[/bold]")

            # Show individual score components
            scores = explanation.get("scores", {})
            for factor, score in scores.items():
                # Color code the scores
                if score > 0:
                    score_color = "green"
                    sign = "+"
                elif score < 0:
                    score_color = "red"
                    sign = ""
                else:
                    score_color = "yellow"
                    sign = " "

                # Print score and explanation
                console.print(
                    f"  {factor:15} [{score_color}]{sign}{score:>6.3f}[/{score_color}] | {explanation['explanations'][factor]}"
                )

            # Show total score
            total = explanation["total_score"]
            console.print(f"\n  {'Total':15} [bold blue]{total:>7.3f}[/bold blue]")
        else:
            # Just show the base relevance score
            relevance = 1 - distance
            console.print(f"[yellow](relevance: {relevance:.2f})[/yellow]")

        # Use file extension as lexer (strip the dot)
        lexer = doc.metadata.get("extension", "").lstrip(".") or "text"

        # Extract preview content (first ~200 chars)
        preview = doc.content[:200] + ("..." if len(doc.content) > 200 else "")

        # Display preview with syntax highlighting
        console.print("\n[bold]Preview:[/bold]")
        syntax = Syntax(
            preview,
            lexer,
            theme="monokai",
            word_wrap=True,
        )
        console.print(syntax)

    # Show assembled context
    console.print("\n[bold]Full Context:[/bold]")
    console.print(f"Total tokens: {context.total_tokens}")
    console.print(f"Documents included: {len(context.documents)}")
    console.print(f"Truncated: {context.truncated}")


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
    default=[],
    help="Glob patterns to ignore",
)
def watch(directory: Path, pattern: str, persist_dir: Path, ignore_patterns: list[str]):
    """Watch directory for changes and update index automatically."""
    try:
        indexer = Indexer(persist_directory=persist_dir, enable_persist=True)

        # Initial indexing
        console.print(f"Performing initial indexing of {directory}")
        with console.status("Indexing..."):
            indexer.index_directory(directory, pattern)

        console.print("Starting file watcher...")

        try:
            # TODO: FileWatcher should use same gitignore patterns as indexer
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


@cli.command()
def status():
    """Show the status of the index."""
    try:
        with console.status("Getting index status..."):
            indexer = Indexer(
                persist_directory=default_persist_dir, enable_persist=True
            )
            status = indexer.get_status()

        # Print basic information
        console.print("\n[bold]Index Status[/bold]")
        console.print(f"Collection: [cyan]{status['collection_name']}[/cyan]")
        console.print(f"Storage Type: [cyan]{status['storage_type']}[/cyan]")
        if "persist_directory" in status:
            console.print(
                f"Persist Directory: [cyan]{status['persist_directory']}[/cyan]"
            )

        # Print document statistics
        console.print("\n[bold]Document Statistics[/bold]")
        console.print(f"Total Documents: [green]{status['document_count']:,}[/green]")
        console.print(f"Total Chunks: [green]{status['chunk_count']:,}[/green]")

        # Print source statistics
        if status["source_stats"]:
            console.print("\n[bold]Source Statistics[/bold]")
            for ext, count in sorted(
                status["source_stats"].items(), key=lambda x: x[1], reverse=True
            ):
                ext_display = ext if ext else "no extension"
                percentage = (
                    (count / status["chunk_count"]) * 100
                    if status["chunk_count"]
                    else 0
                )
                console.print(
                    f"  {ext_display:12} [yellow]{count:4}[/yellow] chunks ([yellow]{percentage:4.1f}%[/yellow])"
                )

        # Print configuration
        console.print("\n[bold]Configuration[/bold]")
        console.print(
            f"Chunk Size: [blue]{status['config']['chunk_size']:,}[/blue] tokens"
        )
        console.print(
            f"Chunk Overlap: [blue]{status['config']['chunk_overlap']:,}[/blue] tokens"
        )

    except Exception as e:
        console.print(f"❌ Error getting index status: {e}", style="red")
        if logging.getLogger().level <= logging.DEBUG:
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
