import pytest
from gptme_rag.indexing.document import Document


@pytest.fixture
def test_docs():
    return [
        Document(
            content="This is a test document about Python programming.",
            metadata={"source": "test1.txt", "category": "programming"},
            doc_id="1",
        ),
        Document(
            content="Another document discussing machine learning.",
            metadata={"source": "test2.txt", "category": "ml"},
            doc_id="2",
        ),
    ]


def test_document_from_file(tmp_path):
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_content = "Test content"
    test_file.write_text(test_content)

    # Create document from file
    docs = list(Document.from_file(test_file))
    assert len(docs) > 0
    doc = docs[0]  # Get the first document

    assert doc.content == test_content
    assert doc.source_path == test_file
    assert doc.metadata["filename"] == "test.txt"
    assert doc.metadata["extension"] == ".txt"


def test_indexer_add_document(indexer, test_docs):
    # Add single document
    indexer.add_document(test_docs[0])
    results, distances, _ = indexer.search("Python programming")

    assert len(results) > 0
    assert "Python programming" in results[0].content
    assert len(distances) > 0


def test_indexer_add_documents(indexer, test_docs):
    # Reset collection to ensure clean state
    indexer.reset_collection()

    # Add multiple documents
    indexer.add_documents(test_docs)

    # Verify documents were added
    results = indexer.collection.get()
    assert len(results["documents"]) == len(test_docs), "Not all documents were added"

    # Search for programming-related content
    prog_results, prog_distances, _ = indexer.search("programming")
    assert len(prog_results) > 0
    assert any("Python" in doc.content for doc in prog_results)
    assert len(prog_distances) > 0

    # Search for ML-related content
    ml_results, ml_distances, _ = indexer.search("machine learning")
    assert len(ml_results) > 0, "No results found for 'machine learning'"
    assert any(
        "machine learning" in doc.content.lower() for doc in ml_results
    ), f"Expected 'machine learning' in results: {[doc.content for doc in ml_results]}"
    assert len(ml_distances) > 0, "No distances returned"


def test_indexer_directory(indexer, tmp_path):
    # Create test files
    (tmp_path / "test1.txt").write_text("Content about Python")
    (tmp_path / "test2.txt").write_text("Content about JavaScript")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "test3.txt").write_text("Content about TypeScript")

    indexer.index_directory(tmp_path)

    # Search for programming languages
    python_results, python_distances, _ = indexer.search("Python")
    js_results, js_distances, _ = indexer.search("JavaScript")
    ts_results, ts_distances, _ = indexer.search("TypeScript")

    assert len(python_results) > 0
    assert len(js_results) > 0
    assert len(ts_results) > 0

    # Verify distances are returned
    assert len(python_distances) > 0
    assert len(js_distances) > 0
    assert len(ts_distances) > 0
