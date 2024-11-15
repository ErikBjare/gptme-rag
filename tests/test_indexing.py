from pathlib import Path
import pytest
import tempfile
from gptme_rag.indexing.document import Document
from gptme_rag.indexing.indexer import Indexer

@pytest.fixture
def test_docs():
    return [
        Document(
            content="This is a test document about Python programming.",
            metadata={"source": "test1.txt", "category": "programming"},
            doc_id="1"
        ),
        Document(
            content="Another document discussing machine learning.",
            metadata={"source": "test2.txt", "category": "ml"},
            doc_id="2"
        )
    ]

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

def test_document_from_file(temp_dir):
    # Create a test file
    test_file = temp_dir / "test.txt"
    test_content = "Test content"
    test_file.write_text(test_content)
    
    # Create document from file
    doc = Document.from_file(test_file)
    
    assert doc.content == test_content
    assert doc.source_path == test_file
    assert doc.metadata["filename"] == "test.txt"
    assert doc.metadata["extension"] == ".txt"

def test_indexer_add_document(temp_dir, test_docs):
    indexer = Indexer(persist_directory=temp_dir)
    
    # Add single document
    indexer.add_document(test_docs[0])
    results = indexer.search("Python programming")
    
    assert len(results) > 0
    assert "Python programming" in results[0].content

def test_indexer_add_documents(temp_dir, test_docs):
    indexer = Indexer(persist_directory=temp_dir)
    
    # Add multiple documents
    indexer.add_documents(test_docs)
    
    # Search for programming-related content
    prog_results = indexer.search("programming")
    assert len(prog_results) > 0
    assert any("Python" in doc.content for doc in prog_results)
    
    # Search for ML-related content
    ml_results = indexer.search("machine learning")
    assert len(ml_results) > 0
    assert any("machine learning" in doc.content.lower() for doc in ml_results)

def test_indexer_directory(temp_dir):
    # Create test files
    (temp_dir / "test1.txt").write_text("Content about Python")
    (temp_dir / "test2.txt").write_text("Content about JavaScript")
    (temp_dir / "subdir").mkdir()
    (temp_dir / "subdir" / "test3.txt").write_text("Content about TypeScript")
    
    indexer = Indexer(persist_directory=temp_dir / "index")
    indexer.index_directory(temp_dir)
    
    # Search for programming languages
    python_results = indexer.search("Python")
    js_results = indexer.search("JavaScript")
    ts_results = indexer.search("TypeScript")
    
    assert len(python_results) > 0
    assert len(js_results) > 0
    assert len(ts_results) > 0
