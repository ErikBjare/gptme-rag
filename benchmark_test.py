import logging
import shutil
import time
from pathlib import Path
from gptme_rag.benchmark import RagBenchmark

# Configure logging to show only ERROR level
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("gptme_rag").setLevel(logging.ERROR)

# Create and clean test directories
test_dir = Path("benchmark_data")
index_dir = Path("benchmark_index")

# Clean up any existing test data
if test_dir.exists():
    shutil.rmtree(test_dir)
if index_dir.exists():
    shutil.rmtree(index_dir)

test_dir.mkdir(exist_ok=True)

# Create test documents in a subdirectory
docs_dir = test_dir / "docs"
docs_dir.mkdir(exist_ok=True)

# Create some test documents
for i in range(10):
    content = f"""# Document {i}

This is test document {i} with some content about Python, documentation, and examples.

## Section 1

Detailed content about programming concepts.

## Section 2

More technical details and implementation examples.

## Section 3

Additional information and use cases.
"""
    (docs_dir / f"doc{i}.md").write_text(content)

# Create a separate file for watch testing
watch_test_file = docs_dir / "watch_test.md"
watch_test_file.write_text("Initial content for watch testing")

try:
    # Initialize benchmark suite
    benchmark = RagBenchmark(index_dir=index_dir)

    # Run indexing benchmark
    print("\nRunning indexing benchmark...")
    index_result = benchmark.run_indexing_benchmark(docs_dir, pattern="*.md")

    # Run search benchmark with various queries
    print("\nRunning search benchmark...")
    search_result = benchmark.run_search_benchmark(
        queries=[
            "python programming",
            "documentation examples",
            "technical details",
            "implementation guide",
        ],
        n_results=5,
    )

    # Custom file watching benchmark
    print("\nRunning file watch benchmark...")

    def watch_operation():
        updates = 0
        start_time = time.time()
        duration = 5.0
        update_interval = 1.0  # 1 second between updates

        while time.time() - start_time < duration:
            # Write meaningful content that changes each time
            content = f"""# Watch Test Update {updates}

This is update number {updates} for the watch test.
Time: {time.time()}

## Content
- Update sequence: {updates}
- Timestamp: {time.time()}
- Test data: Example content for indexing
"""
            watch_test_file.write_text(content)
            updates += 1
            time.sleep(update_interval)

        return {
            "items_processed": updates,
            "metrics": {
                "total_updates": updates,
                "updates_per_second": updates / duration,
            },
        }

    watch_result = benchmark.measure_operation(watch_operation, "file_watching")

    # Print results
    print("\nBenchmark Results:")
    benchmark.print_results()

finally:
    # Clean up test data
    if test_dir.exists():
        shutil.rmtree(test_dir)
    if index_dir.exists():
        shutil.rmtree(index_dir)
