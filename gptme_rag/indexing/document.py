from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

@dataclass
class Document:
    """Represents a document to be indexed."""
    content: str
    metadata: Dict[str, Any]
    source_path: Optional[Path] = None
    doc_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    last_modified: Optional[datetime] = None
    
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
