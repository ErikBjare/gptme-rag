"""File watcher for automatic index updates."""

import logging
import time
from pathlib import Path
from typing import List, Optional, Set

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .indexer import Indexer

logger = logging.getLogger(__name__)


class IndexEventHandler(FileSystemEventHandler):
    """Handle file system events for index updates."""

    def __init__(self, indexer: Indexer, pattern: str = "**/*.*", ignore_patterns: Optional[List[str]] = None):
        """Initialize the event handler.

        Args:
            indexer: The indexer to update
            pattern: Glob pattern for files to index
            ignore_patterns: List of glob patterns to ignore
        """
        self.indexer = indexer
        self.pattern = pattern
        self.ignore_patterns = ignore_patterns or [".git", "__pycache__", "*.pyc"]
        self._pending_updates: Set[Path] = set()
        self._last_update = time.time()
        self._update_delay = 1.0  # seconds

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if not event.is_directory and self._should_process(event.src_path):
            self._queue_update(Path(event.src_path))

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if not event.is_directory and self._should_process(event.src_path):
            self._queue_update(Path(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events."""
        if not event.is_directory and self._should_process(event.src_path):
            # TODO: Implement document removal in Indexer
            logger.info(f"File deleted: {event.src_path}")

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move events."""
        if not event.is_directory:
            if self._should_process(event.src_path):
                # TODO: Implement document removal in Indexer
                logger.info(f"File moved from: {event.src_path}")
            if self._should_process(event.dest_path):
                self._queue_update(Path(event.dest_path))

    def _should_process(self, path: str) -> bool:
        """Check if a file should be processed based on pattern and ignore patterns."""
        path_obj = Path(path)
        return (
            path_obj.match(self.pattern)
            and not any(path_obj.match(pattern) for pattern in self.ignore_patterns)
        )

    def _queue_update(self, path: Path) -> None:
        """Queue a file for update, applying debouncing."""
        self._pending_updates.add(path)
        current_time = time.time()
        
        # In test environments, process immediately
        if self._update_delay == 0:
            self._process_updates()
        # Otherwise apply debouncing
        elif current_time - self._last_update >= self._update_delay:
            self._process_updates()

    def _process_updates(self) -> None:
        """Process all pending updates."""
        if not self._pending_updates:
            return

        # Sort updates by modification time to ensure we process the latest version
        updates = sorted(
            [p for p in self._pending_updates if p.exists()],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Process only the latest version of each file
        processed_files = set()
        for path in updates:
            try:
                # Skip if we've already processed a newer version of this file
                if path.name in processed_files:
                    continue
                    
                # Skip binary and non-text files
                if not path.is_file() or path.suffix in {'.sqlite3', '.db', '.bin', '.pyc'}:
                    continue

                self.indexer.index_file(path)
                processed_files.add(path.name)
            except Exception as e:
                logger.error(f"Failed to update index for {path}: {e}")

        self._pending_updates.clear()
        self._last_update = time.time()


class FileWatcher:
    """Watch files and update the index automatically."""

    def __init__(
        self,
        indexer: Indexer,
        paths: List[str],
        pattern: str = "**/*.*",
        ignore_patterns: Optional[List[str]] = None,
        update_delay: float = 1.0,
    ):
        """Initialize the file watcher.

        Args:
            indexer: The indexer to update
            paths: List of paths to watch
            pattern: Glob pattern for files to index
            ignore_patterns: List of glob patterns to ignore
            update_delay: Delay between updates (0 for immediate updates in tests)
        """
        self.indexer = indexer
        self.paths = [Path(p) for p in paths]
        self.event_handler = IndexEventHandler(indexer, pattern, ignore_patterns)
        self.event_handler._update_delay = update_delay  # Set the update delay
        self.observer = Observer()

    def start(self) -> None:
        """Start watching for file changes."""
        # First index existing files
        for path in self.paths:
            if not path.exists():
                logger.warning(f"Watch path does not exist: {path}")
                continue
            # Index existing files
            self.indexer.index_directory(path, self.event_handler.pattern)
            # Set up watching
            self.observer.schedule(self.event_handler, str(path), recursive=True)
        
        self.observer.start()
        logger.info(f"Started watching paths: {', '.join(str(p) for p in self.paths)}")

    def stop(self) -> None:
        """Stop watching for file changes."""
        self.observer.stop()
        self.observer.join()
        logger.info("Stopped file watcher")

    def __enter__(self) -> 'FileWatcher':
        """Start watching when used as context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop watching when exiting context manager."""
        self.stop()
