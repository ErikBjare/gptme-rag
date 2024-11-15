from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from collections.abc import Generator

from .document_processor import DocumentProcessor


@dataclass
class Document:
    """Represents a document to be indexed."""

    content: str
    metadata: dict[str, Any]
    source_path: Path | None = None
    doc_id: str | None = None
    embedding: list[float] | None = None
    last_modified: datetime | None = None
    chunk_index: int | None = None

    @classmethod
    def from_file(
        cls,
        path: Path,
        processor: DocumentProcessor | None = None,
    ) -> Generator["Document", None, None]:
        """Create Document(s) from a file.

        Args:
            path: Path to the file
            processor: Optional DocumentProcessor for chunking

        Yields:
            Document instances, either a single document or multiple chunks
        """
        last_modified = datetime.fromtimestamp(path.stat().st_mtime)
        base_metadata = {
            "source": str(path),
            "filename": path.name,
            "extension": path.suffix,
            "last_modified": last_modified.isoformat(),
        }

        if processor is None:
            # If no processor provided, create single document
            content = path.read_text()
            yield cls(
                content=content,
                metadata=base_metadata.copy(),
                source_path=path,
                last_modified=last_modified,
            )
        else:
            # Process file in chunks
            base_id = str(path.absolute())
            for chunk in processor.process_file(path):
                # Merge base metadata with chunk metadata
                chunk_metadata = {
                    **base_metadata,
                    **chunk["metadata"],
                }
                # Ensure unique chunk IDs by using both index and position
                chunk_id = f"{base_id}#chunk{chunk['metadata']['chunk_index']}-{chunk['metadata']['chunk_start']}"
                yield cls(
                    content=chunk["text"],
                    metadata=chunk_metadata,
                    source_path=path,
                    doc_id=chunk_id,
                    last_modified=last_modified,
                    chunk_index=chunk["metadata"]["chunk_index"],
                )

    @property
    def is_chunk(self) -> bool:
        """Check if this document is a chunk of a larger document.

        Returns:
            True if this is a chunk, False otherwise
        """
        return self.chunk_index is not None

    def get_chunk_info(self) -> tuple[int, int]:
        """Get information about the chunk.

        Returns:
            Tuple of (chunk_index, total_chunks) if this is a chunk,
            otherwise (0, 1)
        """
        if not self.is_chunk or self.chunk_index is None:
            return (0, 1)
        chunk_index = self.chunk_index  # Now we know it's not None
        total_chunks = self.metadata.get("total_chunks", chunk_index + 1)
        return (chunk_index, total_chunks)
