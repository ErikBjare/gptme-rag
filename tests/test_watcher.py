"""Tests for the file watcher functionality."""

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from gptme_rag.indexing.indexer import Indexer
from gptme_rag.indexing.watcher import FileWatcher


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def indexer(temp_workspace):
    """Create an indexer for testing."""
    return Indexer(persist_directory=temp_workspace / "index")


def test_file_watcher_basic(temp_workspace, indexer):
    """Test basic file watching functionality."""
    test_file = temp_workspace / "test.txt"
    
    with FileWatcher(indexer, [str(temp_workspace)], update_delay=0) as watcher:
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


def test_file_watcher_pattern_matching(temp_workspace, indexer):
    """Test that pattern matching works correctly."""
    with FileWatcher(
        indexer,
        [str(temp_workspace)],
        pattern="*.txt",
        update_delay=0
    ) as watcher:
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


def test_file_watcher_ignore_patterns(temp_workspace, indexer):
    """Test that ignore patterns work correctly."""
    with FileWatcher(
        indexer,
        [str(temp_workspace)],
        ignore_patterns=["*.ignore"],
        update_delay=0
    ) as watcher:
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


def test_file_watcher_move(temp_workspace, indexer):
    """Test handling of file moves."""
    src_file = temp_workspace / "source.txt"
    dst_file = temp_workspace / "destination.txt"
    
    with FileWatcher(indexer, [str(temp_workspace)], update_delay=0) as watcher:
        # Create source file
        src_file.write_text("Test content")
        time.sleep(0.1)  # Small wait for file system events
        
        # Move the file
        src_file.rename(dst_file)
        time.sleep(0.1)  # Small wait for file system events
        
        # Verify file was reindexed at new location
        results, _ = indexer.search("Test content")
        assert len(results) == 1
        assert results[0].metadata["filename"] == dst_file.name


def test_file_watcher_batch_updates(temp_workspace, indexer):
    """Test handling of multiple rapid updates."""
    test_file = temp_workspace / "test.txt"
    
    with FileWatcher(indexer, [str(temp_workspace)], update_delay=0) as watcher:
        # Make multiple rapid updates
        for i in range(5):
            test_file.write_text(f"Content version {i}")
            time.sleep(0.05)  # Minimal delay between updates
        
        time.sleep(0.1)  # Small wait for final update
        
        # Verify only latest version is indexed
        results, _ = indexer.search("Content version")
        assert len(results) == 1
        assert "version 4" in results[0].content
