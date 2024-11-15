import logging
from pathlib import Path

import chromadb
from chromadb.api import Collection
from chromadb.config import Settings

from .document import Document

logger = logging.getLogger(__name__)


class Indexer:
    """Handles document indexing and embedding storage."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "gptme_docs",
    ):
        if persist_directory:
            persist_directory = Path(persist_directory).expanduser().resolve()
            persist_directory.mkdir(parents=True, exist_ok=True)
            print(f"Using persist directory: {persist_directory}")

        settings = Settings()
        if persist_directory:
            settings.persist_directory = str(persist_directory)
            settings.allow_reset = True  # Allow resetting for testing
            settings.is_persistent = True

        logger.debug(f"Creating ChromaDB client with settings: {settings}")
        self.client = chromadb.PersistentClient(path=str(persist_directory))

        def create_collection():
            return self.client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )

        print(f"Getting or creating collection: {collection_name}")
        try:
            self.collection: Collection = create_collection()
            print(f"Collection created/retrieved. Count: {self.collection.count()}")
        except Exception as e:
            print(f"Error creating collection, resetting: {e}")
            self.client.reset()
            self.collection = create_collection()

    def __del__(self):
        """Cleanup when the indexer is destroyed."""
        try:
            self.client.reset()
        except Exception as e:
            if "Resetting is not allowed" not in e.args[0]:
                logger.exception("Error resetting ChromaDB client")

    def add_document(self, document: Document) -> None:
        """Add a single document to the index."""
        if not document.doc_id:
            document.doc_id = str(hash(document.content))

        self.collection.add(
            documents=[document.content],
            metadatas=[document.metadata],
            ids=[document.doc_id],
        )

    def add_documents(self, documents: list[Document]) -> None:
        """Add multiple documents to the index."""
        contents = []
        metadatas = []
        ids = []

        for doc in documents:
            if not doc.doc_id:
                doc.doc_id = str(hash(doc.content))
            contents.append(doc.content)
            metadatas.append(doc.metadata)
            ids.append(doc.doc_id)

        self.collection.add(documents=contents, metadatas=metadatas, ids=ids)

    def index_directory(self, directory: Path, glob_pattern: str = "**/*.*") -> None:
        """Index all files in a directory matching the glob pattern."""
        directory = directory.resolve()  # Convert to absolute path
        files = list(directory.glob(glob_pattern))

        # Filter out database files and get valid files
        valid_files = [
            f
            for f in files
            if f.is_file()
            and not f.name.endswith(".sqlite3")
            and not f.name.endswith(".db")
        ]

        print(f"Found {len(valid_files)} indexable files in {directory}:")
        for f in valid_files:
            print(f"  {f.relative_to(directory)}")

        documents = [Document.from_file(f) for f in valid_files]
        if not documents:
            raise ValueError(
                f"No valid documents found in {directory} with pattern {glob_pattern}"
            )
        self.add_documents(documents)

    def search(
        self, query: str, n_results: int = 5, where: dict | None = None
    ) -> tuple[list[Document], list[float]]:
        """Search for documents similar to the query.

        Returns:
            tuple: (list of Documents, list of distances)
        """
        results = self.collection.query(
            query_texts=[query], n_results=n_results, where=where
        )

        documents = []
        distances = results["distances"][0] if "distances" in results else []

        for i, doc_id in enumerate(results["ids"][0]):
            doc = Document(
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                doc_id=doc_id,
            )
            documents.append(doc)

        return documents, distances
