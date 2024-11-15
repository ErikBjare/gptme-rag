"""Tests for document chunking functionality."""

import tempfile
from pathlib import Path

import pytest

from gptme_rag.indexing.document import Document
from gptme_rag.indexing.document_processor import DocumentProcessor
from gptme_rag.indexing.indexer import Indexer


@pytest.fixture
def test_file():
    """Create a test file with multiple paragraphs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test.txt"
        content = "\n\n".join([
            f"This is paragraph {i} with some content that should be indexed."
            for i in range(10)
        ])
        file_path.write_text(content)
        yield file_path


def test_document_chunking(test_file):
    """Test that documents are properly chunked."""
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    chunks = list(Document.from_file(test_file, processor=processor))
    
    assert len(chunks) > 1
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all(chunk.is_chunk for chunk in chunks)
    assert all(chunk.chunk_index is not None for chunk in chunks)
    
    # Check chunk IDs are unique and properly formatted
    chunk_ids = [chunk.doc_id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert all("#chunk" in id_ for id_ in chunk_ids)


def test_indexing_with_chunks(test_file):
    """Test indexing documents with chunking enabled."""
    with tempfile.TemporaryDirectory() as index_dir:
        indexer = Indexer(
            persist_directory=Path(index_dir),
            chunk_size=100,
            chunk_overlap=20,
        )
        
        # Index the test file
        indexer.index_directory(test_file.parent)
        
        # Search should return results
        docs, distances = indexer.search("paragraph", n_results=5)
        assert len(docs) > 0
        assert len(distances) == len(docs)
        assert all(doc.is_chunk for doc in docs)


def test_chunk_grouping(test_file):
    """Test that chunks are properly grouped in search results."""
    with tempfile.TemporaryDirectory() as index_dir:
        indexer = Indexer(
            persist_directory=Path(index_dir),
            chunk_size=100,
            chunk_overlap=20,
        )
        
        # Index the test file
        indexer.index_directory(test_file.parent)
        
        # Search with and without grouping
        grouped_docs, _ = indexer.search("paragraph", n_results=3, group_chunks=True)
        ungrouped_docs, _ = indexer.search("paragraph", n_results=3, group_chunks=False)
        
        # Grouped results should have unique source documents
        grouped_sources = set(doc.doc_id.split("#chunk")[0] for doc in grouped_docs)
        assert len(grouped_sources) == len(grouped_docs)
        
        # Ungrouped results might have multiple chunks from same document
        ungrouped_sources = set(doc.doc_id.split("#chunk")[0] for doc in ungrouped_docs)
        assert len(ungrouped_sources) <= len(ungrouped_docs)


def test_document_reconstruction(test_file):
    """Test reconstructing full documents from chunks."""
    with tempfile.TemporaryDirectory() as index_dir:
        indexer = Indexer(
            persist_directory=Path(index_dir),
            chunk_size=100,
            chunk_overlap=20,
        )
        
        # Index the test file
        indexer.index_directory(test_file.parent)
        
        # Get a document ID from search results
        docs, _ = indexer.search("paragraph")
        doc_id = docs[0].doc_id.split("#chunk")[0]
        
        # Reconstruct the document
        full_doc = indexer.reconstruct_document(doc_id)
        
        # Check the reconstructed document
        assert not full_doc.is_chunk
        assert full_doc.doc_id == doc_id
        assert "chunk_index" not in full_doc.metadata
        assert len(full_doc.content) > len(docs[0].content)


def test_chunk_retrieval(test_file):
    """Test retrieving all chunks for a document."""
    with tempfile.TemporaryDirectory() as index_dir:
        indexer = Indexer(
            persist_directory=Path(index_dir),
            chunk_size=100,
            chunk_overlap=20,
        )
        
        # Index the test file
        indexer.index_directory(test_file.parent)
        
        # Get a document ID from search results
        docs, _ = indexer.search("paragraph")
        doc_id = docs[0].doc_id.split("#chunk")[0]
        
        # Get all chunks
        chunks = indexer.get_document_chunks(doc_id)
        
        # Check chunks
        assert len(chunks) > 1
        assert all(chunk.is_chunk for chunk in chunks)
        assert all(chunk.doc_id.startswith(doc_id) for chunk in chunks)
        # Check chunks are in order
        chunk_indices = [chunk.chunk_index for chunk in chunks]
        assert chunk_indices == sorted(chunk_indices)
