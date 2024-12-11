"""Benchmarking tools for gptme-rag."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
from rich.console import Console
from rich.table import Table

from .indexing.indexer import Indexer
from .indexing.watcher import FileWatcher

console = Console()


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    operation: str
    duration: float
    memory_usage: float
    throughput: float
    additional_metrics: dict[str, float]

    def __str__(self) -> str:
        """Format the result as a string."""
        return (
            f"{self.operation}:\n"
            f"  Duration: {self.duration:.3f}s\n"
            f"  Memory Usage: {self.memory_usage / 1024 / 1024:.2f}MB\n"
            f"  Throughput: {self.throughput:.2f} items/s\n"
            + "".join(f"  {k}: {v}\n" for k, v in self.additional_metrics.items())
        )


class RagBenchmark:
    """Benchmark suite for gptme-rag operations."""

    def __init__(self, index_dir: Path | None = None):
        """Initialize the benchmark suite.

        Args:
            index_dir: Directory for the index. If None, uses an in-memory database.
        """
        self.index_dir = index_dir
        self.results: list[BenchmarkResult] = []

    def measure_operation(
        self,
        operation_fn: Callable[[], dict[str, Any]],
        name: str,
    ) -> BenchmarkResult:
        """Measure the performance of an operation.

        Args:
            operation_fn: Function to benchmark. Should return a dict with:
                - items_processed: number of items processed
                - metrics: dict of additional metrics
            name: Name of the operation

        Returns:
            BenchmarkResult with performance metrics
        """

        # Calculate metrics
        def get_process_memory():
            """Get total memory usage for the current process and its children."""
            process = psutil.Process()
            try:
                # Get memory info for main process
                mem_info = process.memory_full_info()
                mem = mem_info.uss  # Use USS (Unique Set Size) instead of RSS

                # Include memory of all children
                for child in process.children(recursive=True):
                    try:
                        child_mem = child.memory_full_info()
                        mem += child_mem.uss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                return mem
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return 0

        # Take multiple measurements before operation
        start_measurements = [get_process_memory() for _ in range(3)]
        start_mem = min(start_measurements)  # Use minimum to avoid GC spikes

        # Time the operation and get result
        start_time = time.time()
        result = operation_fn()
        duration = time.time() - start_time

        # Take multiple measurements after operation
        end_measurements = [get_process_memory() for _ in range(3)]
        end_mem = max(end_measurements)  # Use maximum to capture peak usage

        # Calculate metrics - ensure we have a positive value
        memory_used = max(end_mem - start_mem, 1024)  # Minimum 1KB to avoid zero

        items = result.get("items_processed", 1)
        throughput = items / duration if duration > 0 else 0

        # Create and store result
        benchmark_result = BenchmarkResult(
            operation=name,
            duration=duration,
            memory_usage=memory_used,
            throughput=throughput,
            additional_metrics=result.get("metrics", {}),
        )
        self.results.append(benchmark_result)
        return benchmark_result

    def run_indexing_benchmark(
        self,
        docs_path: Path,
        pattern: str = "**/*.*",
    ) -> BenchmarkResult:
        """Benchmark document indexing.

        Args:
            docs_path: Path to documents
            pattern: Glob pattern for files

        Returns:
            BenchmarkResult for the indexing operation
        """

        def index_operation():
            indexer = Indexer(
                persist_directory=self.index_dir,
            )
            files = list(docs_path.glob(pattern))
            indexer.index_directory(docs_path, pattern)
            return {
                "items_processed": len(files),
                "metrics": {
                    "files_processed": len(files),
                    "total_size_mb": sum(f.stat().st_size for f in files) / 1024 / 1024,
                },
            }

        return self.measure_operation(index_operation, "document_indexing")

    def run_search_benchmark(
        self,
        queries: list[str],
        n_results: int = 5,
    ) -> BenchmarkResult:
        """Benchmark search operations.

        Args:
            queries: List of queries to test
            n_results: Number of results per query

        Returns:
            BenchmarkResult for the search operations
        """

        def search_operation():
            indexer = Indexer(
                persist_directory=self.index_dir,
            )
            total_results = 0
            for query in queries:
                results, _, _ = indexer.search(query, n_results=n_results)
                total_results += len(results)
            return {
                "items_processed": len(queries),
                "metrics": {
                    "queries_processed": len(queries),
                    "total_results": total_results,
                    "avg_results_per_query": total_results / len(queries),
                },
            }

        return self.measure_operation(search_operation, "search_operations")

    def run_watch_benchmark(
        self,
        docs_path: Path,
        duration: float = 5.0,
        updates_per_second: float = 2.0,
    ) -> BenchmarkResult:
        """Benchmark file watching operations.

        Args:
            docs_path: Path to test directory
            duration: How long to run the benchmark (seconds)
            updates_per_second: How many updates to perform per second

        Returns:
            BenchmarkResult for the watching operations
        """

        def watch_operation():
            indexer = Indexer(
                persist_directory=self.index_dir,
            )
            test_file = docs_path / "benchmark_test.txt"
            updates = 0

            with FileWatcher(indexer, [str(docs_path)]) as _watcher:
                end_time = time.time() + duration
                while time.time() < end_time:
                    # Write update
                    test_file.write_text(f"Update {updates}")
                    updates += 1
                    # Sleep until next update
                    time.sleep(1 / updates_per_second)

            return {
                "items_processed": updates,
                "metrics": {
                    "total_updates": updates,
                    "updates_per_second": updates / duration,
                },
            }

        return self.measure_operation(watch_operation, "file_watching")

    def print_results(self):
        """Print benchmark results in a formatted table."""
        table = Table(title="Benchmark Results")
        table.add_column("Operation", style="cyan")
        table.add_column("Duration (s)", justify="right", style="green")
        table.add_column("Memory (MB)", justify="right", style="yellow")
        table.add_column("Throughput", justify="right", style="blue")
        table.add_column("Additional Metrics", style="magenta")

        for result in self.results:
            table.add_row(
                result.operation,
                f"{result.duration:.3f}",
                f"{result.memory_usage / 1024 / 1024:.2f}",
                f"{result.throughput:.2f}/s",
                "\n".join(f"{k}: {v}" for k, v in result.additional_metrics.items()),
            )

        console.print(table)
