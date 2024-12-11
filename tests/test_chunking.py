"""Tests for document chunking functionality."""

import tempfile
from pathlib import Path

import pytest

from gptme_rag.indexing.document import Document
from gptme_rag.indexing.document_processor import DocumentProcessor


@pytest.fixture
def test_file():
    """Create a test file with multiple paragraphs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test.txt"
        # Create content with multiple sections and longer paragraphs to ensure chunking
        paragraphs = []
        for i in range(5):  # Fewer sections but more content per section
            paragraphs.extend(
                [
                    f"# Section {i}",
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                    * 10,  # Much longer paragraphs
                    "Ut enim ad minim veniam, quis nostrud exercitation ullamco. " * 10,
                    "Duis aute irure dolor in reprehenderit in voluptate velit. " * 10,
                    "",  # Empty line between sections
                ]
            )
        content = "\n".join(paragraphs)
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
    assert all(id_ is not None and "#chunk" in id_ for id_ in chunk_ids)


def test_indexing_with_chunks(test_file, indexer):
    """Test indexing documents with chunking enabled."""
    # Index the test file
    indexer.index_directory(test_file.parent)

    # Search should return results
    docs, distances, _ = indexer.search("Lorem ipsum", n_results=5)

    assert len(docs) > 0, "No documents found in search results"
    assert len(distances) == len(docs), "Distances don't match documents"
    assert all(doc.is_chunk for doc in docs), "Not all results are chunks"


def test_chunk_grouping(test_file, indexer):
    """Test that chunks are properly grouped in search results."""
    # Index the test file
    indexer.index_directory(test_file.parent)

    # Search with and without grouping
    grouped_docs, _, _ = indexer.search("Lorem ipsum", n_results=3, group_chunks=True)
    ungrouped_docs, _, _ = indexer.search(
        "Lorem ipsum", n_results=3, group_chunks=False
    )

    # Grouped results should have unique source documents
    grouped_sources = set(
        doc.doc_id.split("#chunk")[0] if doc.doc_id else "" for doc in grouped_docs
    )
    assert len(grouped_sources) == len(grouped_docs)

    # Ungrouped results might have multiple chunks from same document
    ungrouped_sources = set(
        doc.doc_id.split("#chunk")[0] if doc.doc_id else "" for doc in ungrouped_docs
    )
    assert len(ungrouped_sources) <= len(ungrouped_docs)


def test_document_reconstruction(test_file, indexer):
    """Test reconstructing full documents from chunks."""
    # Index the test file
    indexer.index_directory(test_file.parent)

    # Get a document ID from search results
    docs, _, _ = indexer.search("Lorem ipsum")  # Search for text we know exists
    base_doc_id = docs[0].doc_id
    assert base_doc_id is not None
    doc_id = base_doc_id.split("#chunk")[0]

    # Reconstruct the document
    full_doc = indexer.reconstruct_document(doc_id)

    # Check the reconstructed document
    assert not full_doc.is_chunk
    assert full_doc.doc_id == doc_id
    assert "chunk_index" not in full_doc.metadata
    assert len(full_doc.content) > len(docs[0].content)


def test_chunk_retrieval(test_file, indexer):
    """Test retrieving all chunks for a document."""
    # Index the test file
    indexer.index_file(test_file)

    # Get a document ID from search results
    docs, _, _ = indexer.search("Lorem ipsum")
    base_doc_id = docs[0].doc_id
    assert base_doc_id is not None
    doc_id = base_doc_id.split("#chunk")[0]

    # Get all chunks
    chunks = indexer.get_document_chunks(doc_id)

    # Check chunks
    assert len(chunks) > 1, "Document should be split into multiple chunks"
    assert all(chunk.is_chunk for chunk in chunks), "All items should be chunks"
    assert all(
        chunk.doc_id is not None and chunk.doc_id.startswith(doc_id) for chunk in chunks
    ), "All chunks should belong to the same document"

    # Check chunks are in order
    chunk_indices = [chunk.chunk_index or 0 for chunk in chunks]
    assert chunk_indices == sorted(chunk_indices), "Chunks should be in order"
