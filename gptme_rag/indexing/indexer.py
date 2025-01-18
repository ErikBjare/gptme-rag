import logging
import os
import subprocess
import time
from collections.abc import Generator
from datetime import datetime
from fnmatch import fnmatch as fnmatch_path
from logging import Filter
from pathlib import Path
from typing import Any

import chromadb
from chromadb import Collection
from chromadb.api import ClientAPI
from chromadb.config import Settings

from .document import Document
from .document_processor import DocumentProcessor


class ChromaDBFilter(Filter):
    """Filter out expected ChromaDB warnings about existing IDs."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno == logging.WARNING:
            # Filter out specific ChromaDB warnings about existing IDs
            if record.name.startswith("chromadb.segment.impl"):
                msg = record.getMessage()
                if "existing embedding ID" in msg:
                    return False
        return True


# Add filter to ChromaDB loggers
for logger_name in [
    "chromadb.segment.impl.metadata.sqlite",
    "chromadb.segment.impl.vector.local_persistent_hnsw",
]:
    logging.getLogger(logger_name).addFilter(ChromaDBFilter())


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
    persist_directory: Path | None

    def __init__(
        self,
        persist_directory: Path | None = None,
        collection_name: str = "default",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        enable_persist: bool = False,  # Default to False due to multi-threading issues
        scoring_weights: dict | None = None,
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
            self.persist_directory = Path(persist_directory).expanduser().resolve()
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using persist directory: {self.persist_directory}")

            settings.persist_directory = str(self.persist_directory)
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory), settings=settings
            )
        else:
            self.persist_directory = None
            self.client = get_client(settings)

        # Initialize collection
        self.collection = get_collection(self.client, collection_name)

        # Initialize document processor
        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Initialize scoring weights with defaults
        self.scoring_weights = {
            "term_overlap": 0.4,  # Term frequency scoring
            "depth_penalty": 0.1,  # Path depth penalty (max)
            "recency_boost": 0.1,  # Recent files (max)
        }
        if scoring_weights:
            self.scoring_weights.update(scoring_weights)

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

    def add_documents(self, documents: list[Document], batch_size: int = 10) -> None:
        """Add multiple documents to the index.

        Args:
            documents: List of documents to add
            batch_size: Number of documents to process in each batch
        """
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self._add_documents(batch)

            # Update stored timestamps after successful indexing
            for doc in batch:
                if "source" in doc.metadata:
                    abs_path = str(Path(doc.metadata["source"]).resolve())
                    current_mtime = int(os.path.getmtime(abs_path))
                    doc.metadata["last_modified"] = current_mtime
                    # Update the document in the collection
                    self.collection.update(
                        ids=[doc.doc_id],
                        metadatas=[doc.metadata],
                    )

    def add_documents_progress(
        self, documents: list[Document], batch_size: int = 10
    ) -> Generator[int, None, None]:
        n_files = len(set(doc.metadata["source"] for doc in documents))
        logger.debug(f"Adding {len(documents)} chunks from {n_files} files")

        processed = 0
        while processed < len(documents):
            batch = documents[processed : processed + batch_size]
            self._add_documents(batch)
            processed += len(batch)
            yield len(batch)

    def _add_documents(self, documents: list[Document]) -> None:
        try:
            contents = []
            metadatas = []
            ids = []

            for doc in documents:
                doc = self._generate_doc_id(doc)
                assert doc.doc_id is not None

                # Update timestamp in metadata to current time
                if "source" in doc.metadata:
                    abs_path = str(Path(doc.metadata["source"]).resolve())
                    current_mtime = os.path.getmtime(abs_path)
                    doc.metadata["last_modified"] = self._normalize_timestamp(
                        current_mtime
                    )

                contents.append(doc.content)
                metadatas.append(doc.metadata)
                ids.append(doc.doc_id)

            # Add batch to collection
            self.collection.add(documents=contents, metadatas=metadatas, ids=ids)
        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            raise

    def _load_gitignore(self, directory: Path) -> list[str]:
        """Load gitignore patterns from all .gitignore files up to root."""
        patterns: list[str] = []

        # Load global gitignore
        global_gitignore = Path.home() / ".config/git/ignore"
        if global_gitignore.exists():
            try:
                with open(global_gitignore) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            patterns.append(line)
            except Exception as e:
                logger.warning(f"Error reading global gitignore: {e}")

        # Essential patterns for non-git directories
        patterns.extend(
            [
                ".git",
                ".git/**",  # Ensure .git dirs are always ignored
                "*.sqlite3",
                "*.db",
            ]
        )
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
                fnmatch_path(rel_path, pattern)
                or fnmatch_path(rel_path, f"**/{pattern}")
                or fnmatch_path(rel_path, f"**/{pattern}/**")
            ):
                return True
        return False

    def index_directory(
        self,
        directory: Path,
        glob_pattern: str = "**/*.*",
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

        # Collect documents using the new method
        documents = self.collect_documents(directory, glob_pattern)

        if not documents:
            return 0

        # Get unique file count
        n_files = len(set(doc.metadata.get("source", "") for doc in documents))

        # Process the documents
        self.add_documents(documents)

        logger.info(f"Indexed {n_files} files from {directory}")
        return n_files

    def debug_collection(self):
        """Debug function to check collection state."""
        # Get all documents
        results = self.collection.get()

        # Print unique document IDs
        unique_ids = set(results["ids"])
        print(f"\nUnique document IDs: {len(unique_ids)} of {len(results['ids'])}")

        # Print a few example documents
        print("\nExample documents:")
        for i in range(min(3, len(results["ids"]))):
            print(f"\nDoc {i}:")
            print(f"ID: {results['ids'][i]}")
            print(f"Content (first 100 chars): {results['documents'][i][:100]}...")
            print(f"Metadata: {results['metadatas'][i]}")

        # Now do a test search
        print("\nTest search for 'Lorem ipsum':")
        search_results = self.collection.query(query_texts=["Lorem ipsum"], n_results=3)
        print("\nRaw search results:")
        print(f"IDs: {search_results['ids'][0]}")
        print(f"Distances: {search_results['distances'][0]}")

    def compute_relevance_score(
        self,
        doc: Document,
        distance: float,
        query: str,
        debug: bool = False,
    ) -> tuple[float, dict[str, float]]:
        """Compute a relevance score for a document based on multiple factors.

        Args:
            doc: The document to score
            distance: The embedding distance from the query
            query: The search query
            debug: Whether to log debug information

        Returns:
            tuple[float, dict[str, float]]: The total score and a dictionary of individual scores
        """
        scores = {}

        # Base similarity score (convert distance to similarity)
        scores["base"] = 1 - distance
        total_score = scores["base"]

        # Term matches (simple tf scoring)
        query_terms = set(query.lower().split())
        content_terms = set(doc.content.lower().split())
        term_overlap = len(query_terms & content_terms) / len(query_terms)
        scores["term_overlap"] = self.scoring_weights["term_overlap"] * term_overlap
        total_score += scores["term_overlap"]

        # Metadata boosts
        if doc.metadata:
            # Path depth penalty
            path_depth = len(Path(doc.metadata.get("source", "")).parts)
            max_depth = 10  # Normalize depth to max of 10 levels
            depth_factor = min(path_depth / max_depth, 1.0)
            scores["depth_penalty"] = (
                -self.scoring_weights["depth_penalty"] * depth_factor
            )
            total_score += scores["depth_penalty"]

            # Recency boost
            scores["recency_boost"] = 0
            if "last_modified" in doc.metadata:
                try:
                    last_modified = float(doc.metadata["last_modified"])
                    days_ago = (time.time() - last_modified) / (24 * 3600)
                    if days_ago < 30:  # 30-day window for recency
                        recency_factor = 1 - (days_ago / 30)
                        scores["recency_boost"] = (
                            self.scoring_weights["recency_boost"] * recency_factor
                        )
                        total_score += scores["recency_boost"]
                except (ValueError, TypeError):
                    logger.debug(
                        f"Invalid last_modified timestamp: {doc.metadata['last_modified']}"
                    )

        # Log scoring breakdown if debug is enabled
        if debug and logger.isEnabledFor(logging.DEBUG):
            source = doc.metadata.get("source", "unknown")
            logger.debug(f"\nScoring breakdown for {source}:")
            for factor, score in scores.items():
                logger.debug(f"  {factor:15}: {score:+.3f}")
            logger.debug(f"  {'total':15}: {total_score:.3f}")

        return total_score, scores

    def _matches_paths(self, doc: Document, paths: list[Path]) -> bool:
        """Check if document matches any of the given paths."""
        source = doc.metadata.get("source", "")
        if not source:
            return False
        source_path = Path(source)
        return any(
            path.resolve() in source_path.parents or path.resolve() == source_path
            for path in paths
        )

    def search(
        self,
        query: str,
        paths: list[Path] | None = None,
        n_results: int = 5,
        where: dict | None = None,
        group_chunks: bool = True,
        max_attempts: int = 3,
        explain: bool = False,
    ) -> tuple[list[Document], list[float], list[dict[str, Any]] | None]:
        """Search for documents similar to the query."""
        # Get more results than needed to allow for filtering
        query_n_results = n_results * 3 if group_chunks else n_results

        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=query_n_results,
            where=where,
        )

        if not results["ids"][0]:
            return [], [], [] if explain else None

        # Process results
        if group_chunks:
            # Group by source document
            docs_by_source: dict[str, tuple[Document, float]] = {}
            for i, doc_id in enumerate(results["ids"][0]):
                source_id = doc_id.split("#chunk")[0]
                if source_id not in docs_by_source:
                    doc = Document(
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i],
                        doc_id=doc_id,
                    )
                    if not paths or self._matches_paths(doc, paths):
                        docs_by_source[source_id] = (doc, results["distances"][0][i])

            # Take top n results
            sorted_docs = sorted(docs_by_source.values(), key=lambda x: x[1])[
                :n_results
            ]
            documents, distances = zip(*sorted_docs) if sorted_docs else ([], [])
        else:
            # Process individual chunks
            documents, distances, _ = self._process_individual_chunks(
                results, paths, n_results, explain
            )

        # Add explanations if requested
        if explain:
            explanations = []
            for doc, distance in zip(documents, distances):
                score, score_breakdown = self.compute_relevance_score(
                    doc, distance, query, debug=explain
                )
                explanations.append(
                    self.explain_scoring(query, doc, distance, score_breakdown)
                )
            return list(documents), list(distances), explanations

        return list(documents), list(distances), None

    def _process_individual_chunks(
        self,
        results: dict,
        paths: list[Path] | None,
        n_results: int,
        explain: bool,
    ) -> tuple[list[Document], list[float], list[dict]]:
        """Process search results as individual chunks."""
        documents: list[Document] = []
        distances: list[float] = []
        explanations: list[dict] = []
        seen_ids = set()

        result_distances = results["distances"][0] if "distances" in results else []

        for i, doc_id in enumerate(results["ids"][0]):
            if len(documents) >= n_results or doc_id in seen_ids:
                break

            doc = Document(
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                doc_id=doc_id,
            )

            if paths and not self._matches_paths(doc, paths):
                continue

            documents.append(doc)
            distances.append(result_distances[i])
            seen_ids.add(doc_id)

        return documents, distances, explanations

    def list_documents(self, group_by_source: bool = True) -> list[Document]:
        """List all documents in the index.

        Args:
            group_by_source: Whether to group chunks from the same document

        Returns:
            List of documents
        """
        # Get all documents from collection
        results = self.collection.get()
        logger.debug("ChromaDB returned %d documents", len(results["ids"]))
        if results["ids"]:
            logger.debug("First document metadata: %s", results["metadatas"][0])

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

    def get_document_chunks(self, base_doc_id: str) -> list[Document]:
        """Get all chunks for a document.

        Args:
            base_doc_id: Base document ID (without chunk suffix)

        Returns:
            List of document chunks, ordered by chunk index
        """
        # Get all documents from collection
        all_docs = self.collection.get()

        # Filter chunks belonging to this document
        chunks = []
        for i, doc_id in enumerate(all_docs["ids"]):
            if doc_id.startswith(base_doc_id):
                chunk = Document(
                    content=all_docs["documents"][i],
                    metadata=all_docs["metadatas"][i],
                    doc_id=doc_id,
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
                results, _, _ = self.search(
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

    def explain_scoring(
        self, query: str, doc: Document, distance: float, scores: dict[str, float]
    ) -> dict:
        """Explain the scoring breakdown for a document.

        Args:
            query: The search query
            doc: The document being scored
            distance: The embedding distance from ChromaDB
            scores: Score breakdown from compute_relevance_score

        Returns:
            dict: Detailed scoring breakdown with explanations
        """
        explanations = {}

        # Base similarity score
        explanations["base"] = f"Embedding similarity: {scores['base']:.3f}"

        # Term overlap
        query_terms = set(query.lower().split())
        content_terms = set(doc.content.lower().split())
        term_overlap = len(query_terms & content_terms) / len(query_terms)
        explanations["term_overlap"] = (
            f"Term overlap {term_overlap:.1%}: +{scores['term_overlap']:.3f}"
        )

        # Path depth
        if "depth_penalty" in scores:
            path_depth = len(Path(doc.metadata.get("source", "")).parts)
            explanations["depth_penalty"] = (
                f"Path depth {path_depth}: {scores['depth_penalty']:.3f}"
            )

        # Recency
        if "recency_boost" in scores:
            if doc.metadata and "last_modified" in doc.metadata:
                try:
                    last_modified = float(doc.metadata["last_modified"])
                    days_ago = (time.time() - last_modified) / (24 * 3600)
                    if days_ago < 30:
                        explanations["recency_boost"] = (
                            f"Modified {days_ago:.1f} days ago: +{scores['recency_boost']:.3f}"
                        )
                    else:
                        explanations["recency_boost"] = (
                            f"Modified {days_ago:.1f} days ago: +0"
                        )
                except (ValueError, TypeError):
                    explanations["recency_boost"] = "Invalid modification time: +0"

        return {
            "scores": scores,
            "explanations": explanations,
            "total_score": sum(scores.values()),
            "weights": self.scoring_weights,
        }

    def get_status(self) -> dict:
        """Get status information about the index.

        Returns:
            dict: Status information including:
                - collection_name: Name of the collection
                - storage_type: "persistent" or "in-memory"
                - persist_directory: Path to persist directory (if persistent)
                - document_count: Number of unique source documents
                - chunk_count: Total number of chunks
                - source_stats: Statistics about document sources
                - config: Basic configuration information
        """
        # Get all documents to analyze
        results = self.collection.get()

        # Count unique source documents
        sources = set()
        source_stats: dict[str, int] = {}  # Extension -> count

        for metadata in results["metadatas"]:
            if metadata and "source" in metadata:
                sources.add(metadata["source"])
                # Get file extension statistics
                ext = Path(metadata["source"]).suffix
                source_stats[ext] = source_stats.get(ext, 0) + 1

        status = {
            "collection_name": self.collection_name,
            "storage_type": "persistent" if self.is_persistent else "in-memory",
            "document_count": len(sources),
            "chunk_count": len(results["ids"]) if results["ids"] else 0,
            "source_stats": source_stats,
            "config": {
                "chunk_size": self.processor.chunk_size,
                "chunk_overlap": self.processor.chunk_overlap,
            },
        }

        if self.is_persistent and self.persist_directory:
            status["persist_directory"] = str(self.persist_directory)

        return status

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

    def _get_valid_files(
        self, path: Path, glob_pattern: str = "**/*.*", file_limit: int = 1000
    ) -> set[Path]:
        """Get valid files for indexing from a path.

        Args:
            path: Path to scan (file or directory)
            glob_pattern: Pattern to match files (only used for directories)
            file_limit: Maximum number of files to return

        Returns:
            Set of valid file paths
        """
        valid_files = set()
        path = path.resolve()  # Resolve path first

        # If it's a file, just validate it
        if path.is_file():
            valid_files.add(path)
            return valid_files

        # For directories, use git ls-files if possible
        try:
            # Check if directory is in a git repo
            subprocess.run(
                ["git", "-C", str(path), "rev-parse", "--git-dir"],
                capture_output=True,
                check=True,
            )

            # Get list of tracked files
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(path),
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            files = [path / line for line in result.stdout.splitlines()]
            gitignore_patterns = None  # No need for gitignore in git mode
            logger.debug("Using git ls-files for file listing")
        except subprocess.CalledProcessError:
            # Not a git repo or git not available, fall back to glob
            files = list(path.glob(glob_pattern))
            gitignore_patterns = self._load_gitignore(path)

        for f in files:
            if not f.is_file():
                continue

            # Check gitignore patterns if in glob mode
            if gitignore_patterns and self._is_ignored(f, gitignore_patterns):
                continue

            # Filter by glob pattern if it's not from git ls-files
            if gitignore_patterns:  # Only check pattern if using glob
                rel_path = str(f.relative_to(path))
                # Convert glob pattern to fnmatch pattern
                fnmatch_pattern = glob_pattern.replace("**/*", "*")
                if not fnmatch_path(rel_path, fnmatch_pattern):
                    continue

            # Resolve symlinks to target
            try:
                resolved = f.resolve()
                valid_files.add(resolved)
            except Exception as e:
                logger.warning(f"Error resolving symlink: {f} -> {e}")

        # Check file limit
        if len(valid_files) >= file_limit:
            logger.warning(
                f"File limit ({file_limit}) reached, was {len(valid_files)}. Consider adding patterns to .gitignore "
                f"or using a more specific glob pattern than '{glob_pattern}' to exclude unwanted files."
            )
            valid_files = set(list(valid_files)[:file_limit])

        return valid_files

    def _normalize_timestamp(self, timestamp: str | float | int | None) -> str:
        """Normalize timestamp to ISO format string."""
        if timestamp is None:
            return datetime.fromtimestamp(0).isoformat()
        try:
            if isinstance(timestamp, int | float):
                return datetime.fromtimestamp(float(timestamp)).isoformat()
            # If it's already an ISO string, validate and return
            if isinstance(timestamp, str):
                datetime.fromisoformat(timestamp)  # Validate format
                return timestamp
            raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
        except (ValueError, TypeError) as e:
            logger.warning("Invalid timestamp format: %s (%s)", timestamp, e)
            return datetime.fromtimestamp(0).isoformat()

    def _compare_timestamps(self, stored: str, current: float) -> bool:
        """Compare stored ISO timestamp with current Unix timestamp.

        Returns True if current is newer than stored."""
        try:
            stored_ts = datetime.fromisoformat(stored).timestamp()
            # Round to seconds for comparison
            return int(current) > int(stored_ts)
        except (ValueError, TypeError) as e:
            logger.warning("Error comparing timestamps: %s", e)
            return True  # If we can't compare, assume modified

    def _get_stored_timestamps(self) -> dict[str, str]:
        """Get stored timestamps for all indexed files."""
        stored = {}
        for doc in self.get_all_documents():
            if "source" in doc.metadata:
                abs_path = str(Path(doc.metadata["source"]).resolve())
                timestamp = self._normalize_timestamp(doc.metadata.get("last_modified"))
                stored[abs_path] = timestamp
                logger.debug("Stored timestamp for %s: %s", abs_path, timestamp)
        return stored

    def collect_documents(
        self, path: Path, glob_pattern: str = "**/*.*", check_modified: bool = True
    ) -> list[Document]:
        """Collect documents from a file or directory without processing them.

        Args:
            path: Path to collect documents from
            glob_pattern: Pattern to match files (only used for directories)
            check_modified: Whether to check for modifications (skip unchanged files)

        Returns:
            List of documents ready for processing
        """
        documents: list[Document] = []
        valid_files = self._get_valid_files(path, glob_pattern)

        if not valid_files:
            logger.debug(f"No valid files found in {path}")
            return documents

        if check_modified:
            stored_timestamps = self._get_stored_timestamps()

        # Process files in order (least deep first)
        for file_path in sorted(valid_files, key=lambda x: len(x.parts)):
            abs_path = str(file_path.resolve())

            if check_modified:
                current_mtime = os.path.getmtime(file_path)
                stored_timestamp = stored_timestamps.get(abs_path)

                if stored_timestamp and not self._compare_timestamps(
                    stored_timestamp, current_mtime
                ):
                    logger.debug("Skipping unchanged file: %s", abs_path)
                    continue

                if not stored_timestamp:
                    logger.debug("New file: %s", abs_path)
                else:
                    logger.debug(
                        "Modified file: %s (current: %s, stored: %s)",
                        abs_path,
                        self._normalize_timestamp(current_mtime),
                        stored_timestamp,
                    )

            logger.debug(f"Processing {file_path}")
            documents.extend(Document.from_file(file_path, processor=self.processor))

        return documents

    def index_file(self, path: Path) -> int:
        """Index a single file.

        Args:
            path: Path to the file to index

        Returns:
            Number of documents indexed
        """
        documents = self.collect_documents(path)
        if documents:
            self.add_documents(documents)
            return len(documents)
        return 0

    def get_all_documents(self) -> list[Document]:
        """Get all documents from the index.

        Returns:
            List of all documents in the index, including all chunks.
        """
        logger.debug("Getting all documents from index")
        docs = self.list_documents(group_by_source=False)
        return docs
