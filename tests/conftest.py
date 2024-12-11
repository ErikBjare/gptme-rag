import pytest
import chromadb


@pytest.fixture(autouse=True)
def cleanup_chroma():
    """Clean up ChromaDB between tests."""
    yield
    # Reset the ChromaDB client system
    if hasattr(chromadb.api.client.SharedSystemClient, "_identifer_to_system"):
        chromadb.api.client.SharedSystemClient._identifer_to_system = {}


@pytest.fixture
def indexer(request, tmp_path):
    """Create an indexer with a unique collection name based on the test name."""
    from gptme_rag.indexing.indexer import Indexer
    import logging

    logger = logging.getLogger(__name__)

    collection_name = request.node.name.replace("[", "_").replace("]", "_")
    idx = Indexer(
        persist_directory=tmp_path / "index",
        chunk_size=50,  # Smaller chunk size to ensure multiple chunks
        chunk_overlap=10,
        enable_persist=True,  # Enable persistent storage
        collection_name=collection_name,  # Unique collection name per test
    )

    # Reset collection before test
    idx.reset_collection()
    logger.debug("Reset collection before test")

    yield idx

    # Cleanup after test
    idx.reset_collection()
    logger.debug("Reset collection after test")
