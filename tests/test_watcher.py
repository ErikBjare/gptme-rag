"""Tests for the file watcher functionality."""

import logging
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from gptme_rag.indexing.indexer import Indexer
from gptme_rag.indexing.watcher import FileWatcher

logger = logging.getLogger(__name__)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def indexer(temp_workspace) -> Indexer:
    """Create an indexer for testing."""
    return Indexer(persist_directory=temp_workspace / "index")


def test_file_watcher_basic(temp_workspace, indexer: Indexer):
    """Test basic file watching functionality."""
    test_file = temp_workspace / "test.txt"

    with FileWatcher(indexer, [str(temp_workspace)], update_delay=0):
        # Create a new file
        test_file.write_text("Initial content")
        time.sleep(1)  # Wait for the watcher to process

        # Verify file was indexed
        results, _ = indexer.search("Initial content")
        assert len(results) == 1
        assert results[0].metadata["filename"] == test_file.name

        # Modify the file
        test_file.write_text("Updated content")
        time.sleep(1)  # Wait for the watcher to process

        # Verify update was indexed
        results, _ = indexer.search("Updated content")
        assert len(results) == 1
        assert results[0].metadata["filename"] == test_file.name


def test_file_watcher_pattern_matching(temp_workspace, indexer: Indexer):
    """Test that pattern matching works correctly."""
    with FileWatcher(indexer, [str(temp_workspace)], pattern="*.txt", update_delay=0):
        # Create files with different extensions
        txt_file = temp_workspace / "test.txt"
        py_file = temp_workspace / "test.py"

        txt_file.write_text("Text file content")
        py_file.write_text("Python file content")

        time.sleep(1)  # Wait for the watcher to process

        # Verify only txt file was indexed
        results, _ = indexer.search("file content")
        assert len(results) == 1
        assert results[0].metadata["filename"] == txt_file.name


def test_file_watcher_ignore_patterns(temp_workspace, indexer: Indexer):
    """Test that ignore patterns work correctly."""
    with FileWatcher(
        indexer, [str(temp_workspace)], ignore_patterns=["*.ignore"], update_delay=0
    ):
        # Create an ignored file and a normal file
        ignored_file = temp_workspace / "test.ignore"
        normal_file = temp_workspace / "test.txt"

        ignored_file.write_text("Should be ignored")
        normal_file.write_text("Should be indexed")

        time.sleep(1)  # Wait for the watcher to process

        # Verify only normal file was indexed
        results, _ = indexer.search("Should be")
        assert len(results) == 1
        assert results[0].metadata["filename"] == normal_file.name


def test_file_watcher_move(temp_workspace, indexer: Indexer):
    """Test handling of file moves."""
    src_file = temp_workspace / "source.txt"
    dst_file = temp_workspace / "destination.txt"

    def wait_for_index(
        content: str, filename: str | None = None, retries: int = 15, delay: float = 0.3
    ) -> bool:
        """Wait for content to appear in index with retries."""
        logger.info(f"Waiting for content to be indexed: {content}")
        for attempt in range(retries):
            results, _ = indexer.search(content)
            if len(results) == 1:
                if filename is None or results[0].metadata["filename"] == filename:
                    logger.info(f"Found content after {attempt + 1} attempts")
                    return True
            logger.debug(f"Attempt {attempt + 1}: content not found, waiting {delay}s")
            time.sleep(delay)
        logger.warning(f"Content not found after {retries} attempts: {content}")
        return False

    with FileWatcher(indexer, [str(temp_workspace)], update_delay=0):
        # Create source file and wait for it to be indexed
        src_file.write_text("Test content")
        assert wait_for_index("Test content", src_file.name), "Source file not indexed"

        # Move the file
        src_file.rename(dst_file)

        # Wait for the moved file to be indexed at new location
        assert wait_for_index("Test content", dst_file.name), "Moved file not indexed"

        # Final verification
        results, _ = indexer.search("Test content")
        assert len(results) == 1, "Expected exactly one result"
        assert (
            results[0].metadata["filename"] == dst_file.name
        ), "Wrong filename in metadata"


def test_file_watcher_batch_updates(temp_workspace, indexer: Indexer):
    """Test handling of multiple rapid updates."""
    test_file = temp_workspace / "test.txt"

    def wait_for_index(content: str, retries: int = 15, delay: float = 0.3) -> bool:
        """Wait for content to appear in index with retries."""
        logger.info(f"Waiting for content to be indexed: {content}")
        for attempt in range(retries):
            results, _ = indexer.search("Content version")
            if results and len(results) == 1 and content in (results[0].content or ""):
                logger.info(f"Found content after {attempt + 1} attempts")
                return True
            logger.debug(f"Attempt {attempt + 1}: content not found, waiting {delay}s")
            time.sleep(delay)
        logger.warning(f"Content not found after {retries} attempts: {content}")
        # Show current index state for debugging
        results, _ = indexer.search("Content version")
        if results:
            logger.warning(f"Current content in index: {[r.content for r in results]}")
        return False

    with FileWatcher(indexer, [str(temp_workspace)], update_delay=0.2):
        # Wait for watcher to initialize
        time.sleep(1.0)
        logger.info("Watcher initialized")

        # Make multiple updates
        for i in range(5):
            content = f"Content version {i}"
            logger.info(f"Writing content: {content}")
            test_file.write_text(content)
            time.sleep(0.3)  # Wait for file to be fully written

            # Wait for update to be indexed with retries
            assert wait_for_index(content), f"Update not indexed: '{content}'"

        # Verify final version is indexed
        results, _ = indexer.search("Content version")
        assert len(results) == 1, "Expected exactly one result"
        assert (
            "version 4" in results[0].content
        ), f"Expected version 4, got: {results[0].content}"
