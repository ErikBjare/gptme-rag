import logging
from pathlib import Path

import chromadb
from chromadb.api import Collection
from chromadb.config import Settings

from .document import Document
from .document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class Indexer:
    """Handles document indexing and embedding storage."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "gptme_docs",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        if persist_directory:
            persist_directory = Path(persist_directory).expanduser().resolve()
            persist_directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Using persist directory: {persist_directory}")

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

        logger.debug(f"Getting or creating collection: {collection_name}")
        try:
            self.collection: Collection = create_collection()
            logger.debug(
                f"Collection created/retrieved. Count: {self.collection.count()}"
            )
        except Exception as e:
            logger.error(f"Error creating collection, resetting: {e}")
            self.client.reset()
            self.collection = create_collection()

        # Initialize document processor
        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

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

    def add_documents(self, documents: list[Document], batch_size: int = 100) -> None:
        """Add multiple documents to the index.

        Args:
            documents: List of documents to add
            batch_size: Number of documents to process in each batch
        """
        total_docs = len(documents)
        processed = 0

        while processed < total_docs:
            # Process a batch of documents
            batch = documents[processed : processed + batch_size]
            contents = []
            metadatas = []
            ids = []

            for doc in batch:
                # Generate consistent ID if not provided
                if not doc.doc_id:
                    base_id = str(
                        hash(
                            doc.source_path.absolute()
                            if doc.source_path
                            else doc.content
                        )
                    )
                    doc.doc_id = (
                        f"{base_id}#chunk{doc.chunk_index}" if doc.is_chunk else base_id
                    )

                contents.append(doc.content)
                metadatas.append(doc.metadata)
                ids.append(doc.doc_id)

            # Add batch to collection
            self.collection.add(documents=contents, metadatas=metadatas, ids=ids)
            processed += len(batch)

            # Report progress
            progress = (processed / total_docs) * 100
            logging.debug(
                f"Indexed {processed}/{total_docs} documents ({progress:.1f}%)"
            )

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

        logging.debug(f"Found {len(valid_files)} indexable files in {directory}:")
        for f in valid_files:
            logging.debug(f"  {f.relative_to(directory)}")

        if not valid_files:
            logger.debug(
                f"No valid documents found in {directory} with pattern {glob_pattern}"
            )
            return

        # Process files in batches to manage memory
        batch_size = 100
        current_batch = []

        for file_path in valid_files:
            # Process each file into chunks
            for doc in Document.from_file(file_path, processor=self.processor):
                current_batch.append(doc)
                if len(current_batch) >= batch_size:
                    self.add_documents(current_batch)
                    current_batch = []

        # Add any remaining documents
        if current_batch:
            self.add_documents(current_batch)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict | None = None,
        group_chunks: bool = True,
    ) -> tuple[list[Document], list[float]]:
        """Search for documents similar to the query.

        Args:
            query: Search query
            n_results: Number of results to return
            where: Optional filter conditions
            group_chunks: Whether to group chunks from the same document

        Returns:
            tuple: (list of Documents, list of distances)
        """
        # Get more results if grouping chunks to ensure we have enough unique documents
        query_n_results = n_results * 3 if group_chunks else n_results

        results = self.collection.query(
            query_texts=[query], n_results=query_n_results, where=where
        )

        documents = []
        distances = results["distances"][0] if "distances" in results else []

        # Group chunks by source document if requested
        if group_chunks:
            doc_groups: dict[str, list[tuple[Document, float]]] = {}

            for i, doc_id in enumerate(results["ids"][0]):
                doc = Document(
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    doc_id=doc_id,
                )

                # Get source document ID (remove chunk suffix if present)
                source_id = doc_id.split("#chunk")[0]

                if source_id not in doc_groups:
                    doc_groups[source_id] = []
                doc_groups[source_id].append((doc, distances[i]))

            # Take the best chunk from each document
            for source_docs in list(doc_groups.values())[:n_results]:
                best_doc, best_distance = min(source_docs, key=lambda x: x[1])
                documents.append(best_doc)
                distances[len(documents) - 1] = best_distance
        else:
            # Return individual chunks
            for i, doc_id in enumerate(results["ids"][0][:n_results]):
                doc = Document(
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    doc_id=doc_id,
                )
                documents.append(doc)

        return documents, distances[: len(documents)]

    def get_document_chunks(self, doc_id: str) -> list[Document]:
        """Get all chunks for a document.

        Args:
            doc_id: Base document ID (without chunk suffix)

        Returns:
            List of document chunks, ordered by chunk index
        """
        results = self.collection.get(where={"source": doc_id})

        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            chunk = Document(
                content=results["documents"][i],
                metadata=results["metadatas"][i],
                doc_id=chunk_id,
            )
            chunks.append(chunk)

        # Sort chunks by index
        chunks.sort(key=lambda x: x.chunk_index or 0)
        return chunks

    def reconstruct_document(self, doc_id: str) -> Document:
        """Reconstruct a full document from its chunks.

        Args:
            doc_id: Base document ID (without chunk suffix)

        Returns:
            Complete document
        """
        chunks = self.get_document_chunks(doc_id)
        if not chunks:
            raise ValueError(f"No chunks found for document {doc_id}")

        # Combine chunk contents
        content = "\n".join(chunk.content for chunk in chunks)

        # Use metadata from first chunk, removing chunk-specific fields
        metadata = chunks[0].metadata.copy()
        metadata.pop("chunk_index", None)
        metadata.pop("token_count", None)

        return Document(
            content=content,
            metadata=metadata,
            doc_id=doc_id,
            source_path=chunks[0].source_path,
            last_modified=chunks[0].last_modified,
        )

    def index_file(self, path: Path) -> None:
        """Index a single file.

        Args:
            path: Path to the file to index
        """
        documents = list(Document.from_file(path, processor=self.processor))
        if documents:
            self.add_documents(documents)
