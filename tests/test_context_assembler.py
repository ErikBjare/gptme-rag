import pytest
from gptme_rag.query.context_assembler import ContextAssembler
from gptme_rag.indexing.document import Document


@pytest.fixture
def test_docs():
    return [
        Document(
            content="This is document 1 about Python programming.",
            metadata={"source": "doc1.txt"},
            doc_id="1",
        ),
        Document(
            content="This is document 2 about machine learning.",
            metadata={"source": "doc2.txt"},
            doc_id="2",
        ),
        Document(
            content="This is document 3 about data science.",
            metadata={"source": "doc3.txt"},
            doc_id="3",
        ),
    ]


@pytest.fixture
def assembler():
    return ContextAssembler(max_tokens=1000)


def test_context_assembly_basic(assembler, test_docs):
    context = assembler.assemble_context(test_docs)

    # Check that all documents are included
    assert len(context.documents) == len(test_docs)
    assert context.total_tokens > 0
    assert not context.truncated

    # Check content formatting
    for doc in test_docs:
        assert doc.content in context.content
        assert doc.metadata["source"] in context.content


def test_context_assembly_with_system_prompt(assembler, test_docs):
    system_prompt = "You are a helpful AI assistant."
    context = assembler.assemble_context(test_docs, system_prompt=system_prompt)

    # Check system prompt is included
    assert system_prompt in context.content
    assert context.total_tokens > 0


def test_context_assembly_with_query(assembler, test_docs):
    query = "Tell me about Python programming"
    context = assembler.assemble_context(test_docs, user_query=query)

    # Check query is included
    assert query in context.content
    assert context.total_tokens > 0


def test_context_assembly_token_limit():
    # Create assembler with very low token limit
    small_assembler = ContextAssembler(max_tokens=50)

    # Create documents with very long content
    long_docs = [
        Document(
            content="A"
            * 1000,  # Much longer content that will definitely exceed token limit
            metadata={"source": f"doc{i}.txt"},
            doc_id=str(i),
        )
        for i in range(3)
    ]

    context = small_assembler.assemble_context(long_docs)

    # Check that truncation occurred
    assert context.truncated
    assert len(context.documents) < len(long_docs)
    assert context.total_tokens <= small_assembler.max_tokens


def test_context_assembly_empty_docs(assembler):
    context = assembler.assemble_context([])

    assert context.content == ""
    assert len(context.documents) == 0
    assert context.total_tokens == 0
    assert not context.truncated


def test_context_assembly_all_components(assembler, test_docs):
    system_prompt = "You are a helpful AI assistant."
    query = "Tell me about programming"

    context = assembler.assemble_context(
        test_docs, system_prompt=system_prompt, user_query=query
    )

    # Check all components are present
    assert system_prompt in context.content
    assert query in context.content
    for doc in test_docs:
        assert doc.content in context.content
