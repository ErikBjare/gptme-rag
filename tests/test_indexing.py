from pathlib import Path

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
    # Create test files in different directories with different extensions
    docs_dir = tmp_path / "docs"
    src_dir = tmp_path / "src"
    docs_dir.mkdir()
    src_dir.mkdir()

    # Create markdown files in docs
    (docs_dir / "guide.md").write_text("Python programming guide")
    (docs_dir / "tutorial.md").write_text("JavaScript tutorial")

    # Create Python files in src
    (src_dir / "main.py").write_text("def main(): print('Hello')")
    (src_dir / "utils.py").write_text("def util(): return True")

    # Create a text file in root
    (tmp_path / "notes.txt").write_text("Random notes")

    # Index everything
    indexer.index_directory(tmp_path)

    # Test extension filter (*.md)
    md_results, _, _ = indexer.search(
        "programming",
        path_filters=("*.md",),
    )
    assert len(md_results) > 0
    assert all(doc.metadata["source"].endswith(".md") for doc in md_results)

    # Test directory pattern (src/*.py)
    py_results, _, _ = indexer.search(
        "def",
        path_filters=(str(src_dir / "*.py"),),
    )
    assert len(py_results) > 0
    assert all(
        Path(doc.metadata["source"]).parent.name == "src"
        and doc.metadata["source"].endswith(".py")
        for doc in py_results
    )

    # Test multiple patterns
    multi_results, _, _ = indexer.search(
        "programming",
        path_filters=("*.md", "*.py"),
    )
    assert len(multi_results) > 0
    assert all(doc.metadata["source"].endswith((".md", ".py")) for doc in multi_results)

    # Test with path and filter combined
    docs_md_results, _, _ = indexer.search(
        "tutorial",
        paths=[docs_dir],
        path_filters=("*.md",),
    )
    assert len(docs_md_results) > 0
    assert all(
        Path(doc.metadata["source"]).parent.name == "docs"
        and doc.metadata["source"].endswith(".md")
        for doc in docs_md_results
    )


def test_path_matching(indexer):
    # Test the _matches_paths method directly
    doc = Document(
        content="test",
        metadata={"source": "/home/user/project/docs/guide.md"},
        doc_id="test",
    )

    # Test simple extension filter
    assert indexer._matches_paths(doc, path_filters=("*.md",))
    assert not indexer._matches_paths(doc, path_filters=("*.py",))

    # Test directory pattern
    assert indexer._matches_paths(doc, path_filters=("docs/*.md",))
    assert not indexer._matches_paths(doc, path_filters=("src/*.md",))

    # Test multiple patterns
    assert indexer._matches_paths(doc, path_filters=("*.py", "*.md"))
    assert indexer._matches_paths(doc, path_filters=("src/*.py", "docs/*.md"))

    # Test with exact paths
    assert indexer._matches_paths(doc, paths=[Path("/home/user/project/docs")])
    assert not indexer._matches_paths(doc, paths=[Path("/home/user/project/src")])

    # Test combining paths and filters
    assert indexer._matches_paths(
        doc,
        paths=[Path("/home/user/project/docs")],
        path_filters=("*.md",),
    )
    assert not indexer._matches_paths(
        doc,
        paths=[Path("/home/user/project/docs")],
        path_filters=("*.py",),
    )
