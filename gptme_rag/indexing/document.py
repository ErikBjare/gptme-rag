from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """Represents a document to be indexed."""

    content: str
    metadata: dict[str, Any]
    source_path: Path | None = None
    doc_id: str | None = None
    embedding: list[float] | None = None
    last_modified: datetime | None = None

    @classmethod
    def from_file(cls, path: Path) -> "Document":
        """Create a Document from a file."""
        content = path.read_text()
        last_modified = datetime.fromtimestamp(path.stat().st_mtime)
        metadata = {
            "source": str(path),
            "filename": path.name,
            "extension": path.suffix,
            "last_modified": last_modified.isoformat(),  # Convert to ISO format string
        }
        return cls(
            content=content,
            metadata=metadata,
            source_path=path,
            last_modified=last_modified,
        )
