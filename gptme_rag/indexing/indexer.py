import logging
import time
from fnmatch import fnmatch
from pathlib import Path

import chromadb
from chromadb import Collection
from chromadb.api import ClientAPI
from chromadb.config import Settings

from .document import Document
from .document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


def get_client(settings: Settings | None = None) -> ClientAPI:
    """Create a new ChromaDB client with the given settings."""
    if settings is None:
        settings = Settings(
            allow_reset=True,
            is_persistent=False,
            anonymized_telemetry=False,
        )
    return chromadb.Client(settings)


def get_collection(client: ClientAPI, name: str) -> Collection:
    """Get or create a collection with consistent ID."""
    try:
        # Try to get existing collection
        return client.get_collection(name=name)
    except ValueError:
        # Create if it doesn't exist
        return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})


class Indexer:
    """Handles document indexing and embedding storage."""

    processor: DocumentProcessor
    is_persistent: bool = False

    def __init__(
        self,
        persist_directory: Path | None = None,
        collection_name: str = "default",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        enable_persist: bool = False,  # Default to False due to multi-threading issues
    ):
        """Initialize the indexer."""
        self.collection_name = collection_name

        # Initialize settings
        settings = Settings(
            allow_reset=True,
            is_persistent=enable_persist,
            anonymized_telemetry=False,
        )

        if persist_directory and enable_persist:
            self.is_persistent = True
            persist_directory = Path(persist_directory).expanduser().resolve()
            persist_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using persist directory: {persist_directory}")
            settings.persist_directory = str(persist_directory)
            self.client = chromadb.PersistentClient(
                path=str(persist_directory), settings=settings
            )
        else:
            self.client = get_client(settings)

        # Initialize collection
        self.collection = get_collection(self.client, collection_name)

        # Initialize document processor
        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _generate_doc_id(self, document: Document) -> Document:
        if not document.doc_id:
            base = str(hash(document.content))
            ts = int(time.time() * 1000)
            document.doc_id = f"{base}-{ts}"
        return document

    def reset_collection(self) -> None:
        """Reset the collection to a clean state."""
        try:
            self.client.delete_collection(self.collection_name)
        except ValueError:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.debug(f"Reset collection: {self.collection_name}")

    def add_document(self, document: Document) -> None:
        """Add a single document to the index."""
        document = self._generate_doc_id(document)
        assert document.doc_id is not None

        try:
            self.collection.add(
                documents=[document.content],
                metadatas=[document.metadata],
                ids=[document.doc_id],
            )
            logger.debug(f"Added document with ID: {document.doc_id}")
        except Exception as e:
            logger.error(f"Error adding document: {e}", exc_info=True)
            # Reset collection and retry
            self.reset_collection()
            self.collection.add(
                documents=[document.content],
                metadatas=[document.metadata],
                ids=[document.doc_id],
            )

    def delete_documents(self, where: dict) -> None:
        """Delete documents matching the where clause."""
        try:
            self.collection.delete(where=where)
            logger.debug(f"Deleted documents matching: {where}")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}", exc_info=True)
            # Reset collection if needed
            self.reset_collection()

    def add_documents(self, documents: list[Document], batch_size: int = 100) -> None:
        """Add multiple documents to the index.

        Args:
            documents: List of documents to add
            batch_size: Number of documents to process in each batch
        """
        total_docs = len(documents)
        processed = 0

        while processed < total_docs:
            try:
                # Process a batch of documents
                batch = documents[processed : processed + batch_size]
                contents = []
                metadatas = []
                ids = []

                for doc in batch:
                    doc = self._generate_doc_id(doc)
                    assert doc.doc_id is not None

                    contents.append(doc.content)
                    metadatas.append(doc.metadata)
                    ids.append(doc.doc_id)

                # Add batch to collection
                self.collection.add(documents=contents, metadatas=metadatas, ids=ids)

                processed += len(batch)
            except Exception as e:
                logger.error(f"Failed to process batch: {e}")
                raise

            # Report progress
            progress = (processed / total_docs) * 100
            logging.debug(
                f"Indexed {processed}/{total_docs} documents ({progress:.1f}%)"
            )

    def _load_gitignore(self, directory: Path) -> list[str]:
        """Load gitignore patterns from all .gitignore files up to root."""
        # arguably only .git/** should be here, with the rest in system global gitignore (which should be respected)
        patterns: list[str] = [
            ".git/**",
            "*.sqlite3",
            "*.db",
            "*.pyc",
            "__pycache__",
            ".*cache/**",
            "*.lock",
            ".DS_Store",
        ]
        current_dir = directory.resolve()
        max_depth = 10  # Limit traversal to avoid infinite loops

        # Collect all .gitignore files up to root or max depth
        depth = 0
        while current_dir.parent != current_dir and depth < max_depth:
            gitignore_path = current_dir / ".gitignore"
            if gitignore_path.exists():
                try:
                    patterns.extend(
                        line.strip()
                        for line in gitignore_path.read_text().splitlines()
                        if line.strip() and not line.startswith("#")
                    )
                except Exception as e:
                    logger.warning(f"Error reading {gitignore_path}: {e}")
            current_dir = current_dir.parent
            depth += 1

        return patterns

    def _is_ignored(self, file_path: Path, gitignore_patterns: list[str]) -> bool:
        """Check if a file matches any gitignore pattern."""

        # Convert path to relative for pattern matching
        rel_path = str(file_path)

        for pattern in gitignore_patterns:
            if (
                fnmatch(rel_path, pattern)
                or fnmatch(rel_path, f"**/{pattern}")
                or fnmatch(rel_path, f"**/{pattern}/**")
            ):
                return True
        return False

    def index_directory(
        self, directory: Path, glob_pattern: str = "**/*.*", file_limit: int = 100
    ) -> int:
        """Index all files in a directory matching the glob pattern.

        Args:
            directory: Directory to index
            glob_pattern: Pattern to match files
            file_limit: Maximum number of files to index

        Returns:
            Number of files indexed
        """
        directory = directory.resolve()  # Convert to absolute path
        files = list(directory.glob(glob_pattern))

        # Load gitignore patterns
        gitignore_patterns = self._load_gitignore(directory)

        # Filter files
        valid_files = set()
        for f in files:
            if f.is_file() and not self._is_ignored(f, gitignore_patterns):
                valid_files.add(f)

            # Check file limit
            if len(valid_files) >= file_limit:
                logger.warning(
                    f"File limit ({file_limit}) reached. Consider adding patterns to .gitignore "
                    f"or using a more specific glob pattern than '{glob_pattern}' to exclude unwanted files."
                )
                break

        logging.debug(f"Found {len(valid_files)} indexable files in {directory}:")
        for f in valid_files:
            logging.debug(f"  {f.relative_to(directory)}")

        if not valid_files:
            logger.debug(
                f"No valid documents found in {directory} with pattern {glob_pattern}"
            )
            return 0

        # Process files in batches to manage memory
        batch_size = 100
        current_batch = []

        for file_path in valid_files:
            logger.debug(f"Processing file: {file_path}")
            # Process each file into chunks
            for doc in Document.from_file(file_path, processor=self.processor):
                logger.debug(f"Processing chunk: {doc.source_path} ({doc.chunk_index})")
                current_batch.append(doc)
                if len(current_batch) >= batch_size:
                    logger.info(f"Adding {len(current_batch)} documents")
                    self.add_documents(current_batch)
                    current_batch = []

        # Add any remaining documents
        if current_batch:
            self.add_documents(current_batch)
            logger.info(f"Adding {len(current_batch)} documents.")
            self.add_documents(current_batch)

        logger.info(f"Indexed {len(valid_files)} documents from {directory}")
        return len(valid_files)

    def search(
        self,
        query: str,
        paths: list[Path] | None = None,
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

        # Add batch to collection
        # TODO: can we do file filtering here to ensure we get exactly n_results?
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

                path = doc.metadata.get("source", "unknown")
                if paths:
                    matches_paths = [
                        filter_path in Path(path).parents for filter_path in paths
                    ]
                    if not any(matches_paths):
                        continue

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

    def list_documents(self, group_by_source: bool = True) -> list[Document]:
        """List all documents in the index.

        Args:
            group_by_source: Whether to group chunks from the same document

        Returns:
            List of documents
        """
        # Get all documents from collection
        results = self.collection.get()

        if not results["ids"]:
            return []

        if group_by_source:
            # Group chunks by source document
            doc_groups: dict[str, list[Document]] = {}

            for i, doc_id in enumerate(results["ids"]):
                doc = Document(
                    content=results["documents"][i],
                    metadata=results["metadatas"][i],
                    doc_id=doc_id,
                )

                # Get source document ID (remove chunk suffix if present)
                source_id = doc_id.split("#chunk")[0]

                if source_id not in doc_groups:
                    doc_groups[source_id] = []
                doc_groups[source_id].append(doc)

            # Return first chunk from each document group
            return [chunks[0] for chunks in doc_groups.values()]
        else:
            # Return all documents/chunks
            return [
                Document(
                    content=results["documents"][i],
                    metadata=results["metadatas"][i],
                    doc_id=doc_id,
                )
                for i, doc_id in enumerate(results["ids"])
            ]

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
        # Create clean metadata without chunk-specific fields
        metadata = chunks[0].metadata.copy()
        for key in [
            "chunk_index",
            "token_count",
            "is_chunk",
            "chunk_start",
            "chunk_end",
        ]:
            metadata.pop(key, None)

        return Document(
            content=content,
            metadata=metadata,
            doc_id=doc_id,
            source_path=chunks[0].source_path,
            last_modified=chunks[0].last_modified,
        )

    def verify_document(
        self,
        path: Path,
        content: str | None = None,
        retries: int = 3,
        delay: float = 0.2,
    ) -> bool:
        """Verify that a document is properly indexed.

        Args:
            path: Path to the document
            content: Optional content to verify (if different from file)
            retries: Number of verification attempts
            delay: Delay between retries

        Returns:
            bool: True if document is verified in index
        """
        search_content = content if content is not None else path.read_text()[:100]
        canonical_path = str(path.resolve())

        for attempt in range(retries):
            try:
                results, _ = self.search(
                    search_content, n_results=1, where={"source": canonical_path}
                )
                if results and search_content in results[0].content:
                    logger.debug(f"Document verified on attempt {attempt + 1}: {path}")
                    return True
                time.sleep(delay)
            except Exception as e:
                logger.warning(f"Verification attempt {attempt + 1} failed: {e}")
                time.sleep(delay)

        logger.warning(f"Failed to verify document after {retries} attempts: {path}")
        return False

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks from the index.

        Args:
            doc_id: Base document ID (without chunk suffix)

        Returns:
            bool: True if deletion was successful
        """
        try:
            # First try to delete by exact ID
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Deleted document: {doc_id}")

            # Then delete any related chunks
            try:
                self.collection.delete(where={"source": doc_id})
                logger.debug(f"Deleted related chunks for: {doc_id}")
            except Exception as chunk_e:
                logger.warning(f"Error deleting chunks for {doc_id}: {chunk_e}")

            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def index_file(self, path: Path) -> None:
        """Index a single file.

        Args:
            path: Path to the file to index
        """
        documents = list(Document.from_file(path, processor=self.processor))
        if documents:
            self.add_documents(documents)
