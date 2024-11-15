"""Tests for the benchmarking functionality."""

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from gptme_rag.benchmark import RagBenchmark


@pytest.fixture
def temp_docs():
    """Create a temporary directory with test documents."""
    with TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir) / "docs"
        docs_dir.mkdir()
        
        # Create some test files
        for i in range(5):
            file = docs_dir / f"test{i}.txt"
            file.write_text(f"Test document {i} with some content for indexing.")
        
        yield docs_dir


def test_measure_operation():
    """Test basic operation measurement."""
    benchmark = RagBenchmark()
    
    def test_op():
        time.sleep(0.1)  # Simulate work
        return {"items_processed": 10, "metrics": {"test_metric": 42}}
    
    result = benchmark.measure_operation(test_op, "test_operation")
    
    assert result.operation == "test_operation"
    assert result.duration >= 0.1
    assert result.memory_usage >= 0
    assert result.throughput == 10 / result.duration
    assert result.additional_metrics["test_metric"] == 42


def test_indexing_benchmark(temp_docs):
    """Test document indexing benchmark."""
    benchmark = RagBenchmark()
    result = benchmark.run_indexing_benchmark(temp_docs)
    
    assert result.operation == "document_indexing"
    assert result.duration > 0
    assert result.memory_usage > 0
    assert result.throughput > 0
    assert result.additional_metrics["files_processed"] == 5


def test_search_benchmark(temp_docs):
    """Test search operations benchmark."""
    benchmark = RagBenchmark()
    # First index some documents
    benchmark.run_indexing_benchmark(temp_docs)
    
    # Then run search benchmark
    queries = ["test", "document", "content"]
    result = benchmark.run_search_benchmark(queries)
    
    assert result.operation == "search_operations"
    assert result.duration > 0
    assert result.memory_usage > 0
    assert result.throughput > 0
    assert result.additional_metrics["queries_processed"] == len(queries)


def test_watch_benchmark(temp_docs):
    """Test file watching benchmark."""
    benchmark = RagBenchmark()
    result = benchmark.run_watch_benchmark(temp_docs, duration=1.0, updates_per_second=5.0)
    
    assert result.operation == "file_watching"
    assert result.duration >= 1.0
    assert result.memory_usage > 0
    assert result.throughput > 0
    assert result.additional_metrics["updates_per_second"] >= 4.0  # Allow some timing variance


def test_print_results(capsys):
    """Test results printing."""
    benchmark = RagBenchmark()
    
    def test_op():
        return {"items_processed": 1, "metrics": {"test": 123}}
    
    benchmark.measure_operation(test_op, "test_op")
    benchmark.print_results()
    
    captured = capsys.readouterr()
    assert "Benchmark Results" in captured.out
    assert "test_op" in captured.out
