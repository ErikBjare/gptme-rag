"""Tests for the document processor."""

import tempfile
from pathlib import Path

import pytest

from gptme_rag.indexing.document_processor import DocumentProcessor


def test_process_text_basic():
    """Test basic text processing."""
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    text = "This is a test document. " * 50  # Create text long enough to split
    
    chunks = list(processor.process_text(text))
    
    assert len(chunks) > 1  # Should split into multiple chunks
    assert all(isinstance(c["text"], str) for c in chunks)
    assert all(isinstance(c["metadata"], dict) for c in chunks)
    assert all(c["metadata"]["chunk_index"] >= 0 for c in chunks)


def test_process_text_with_metadata():
    """Test text processing with metadata."""
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    text = "Test document content. " * 20
    metadata = {"source": "test", "category": "example"}
    
    chunks = list(processor.process_text(text, metadata))
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["metadata"]["source"] == "test"
        assert chunk["metadata"]["category"] == "example"
        assert "chunk_index" in chunk["metadata"]
        assert "token_count" in chunk["metadata"]


def test_chunk_overlap():
    """Test that chunks overlap correctly."""
    processor = DocumentProcessor(chunk_size=10, chunk_overlap=5)
    text = "This is a test of the chunking overlap system."
    
    chunks = list(processor.process_text(text))
    
    # Check that consecutive chunks have overlapping content
    for i in range(len(chunks) - 1):
        current_text = chunks[i]["text"]
        next_text = chunks[i + 1]["text"]
        
        # There should be some overlap between chunks
        assert any(word in next_text for word in current_text.split())


def test_process_file():
    """Test file processing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file
        file_path = Path(temp_dir) / "test.txt"
        content = "This is a test file.\n" * 100
        file_path.write_text(content)
        
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        chunks = list(processor.process_file(file_path))
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert "filename" in chunk["metadata"]
            assert chunk["metadata"]["filename"] == "test.txt"
            assert "file_path" in chunk["metadata"]
            assert "file_type" in chunk["metadata"]
            assert chunk["metadata"]["file_type"] == "txt"


def test_max_chunks():
    """Test max_chunks limitation."""
    processor = DocumentProcessor(chunk_size=10, chunk_overlap=2, max_chunks=3)
    text = "This is a test. " * 20
    
    chunks = list(processor.process_text(text))
    
    assert len(chunks) <= 3


def test_token_estimation():
    """Test token count estimation."""
    processor = DocumentProcessor()
    text = "This is a test document."
    
    token_count = processor.estimate_token_count(text)
    assert token_count > 0
    
    chunks = processor.estimate_chunks(token_count)
    assert chunks > 0


def test_optimal_chunk_size():
    """Test optimal chunk size calculation."""
    processor = DocumentProcessor(chunk_overlap=10)
    token_count = 1000
    target_chunks = 5
    
    chunk_size = processor.get_optimal_chunk_size(target_chunks, token_count)
    
    assert chunk_size >= processor.chunk_overlap
    assert chunk_size >= 100  # min_chunk_size


def test_invalid_chunk_settings():
    """Test handling of invalid chunk settings."""
    processor = DocumentProcessor(chunk_size=50, chunk_overlap=60)
    
    with pytest.raises(ValueError):
        processor.estimate_chunks(100)


def test_large_file_streaming():
    """Test processing of a large file in streaming fashion."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a large test file
        file_path = Path(temp_dir) / "large.txt"
        with open(file_path, "w") as f:
            for i in range(1000):
                f.write(f"This is line {i} of the test document.\n")
        
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        chunks = list(processor.process_file(file_path))
        
        assert len(chunks) > 0
        # Check that we're getting reasonable chunk sizes
        for chunk in chunks:
            assert len(chunk["text"]) > 0
            assert chunk["metadata"]["token_count"] <= processor.chunk_size


def test_empty_text():
    """Test processing of empty text."""
    processor = DocumentProcessor()
    chunks = list(processor.process_text(""))
    
    assert len(chunks) == 0


def test_small_text():
    """Test processing of text smaller than chunk size."""
    processor = DocumentProcessor(chunk_size=1000)
    text = "Small text."
    chunks = list(processor.process_text(text))
    
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
