"""File watcher for automatic index updates."""

import logging
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .indexer import Indexer

logger = logging.getLogger(__name__)


class IndexEventHandler(FileSystemEventHandler):
    """Handle file system events for index updates."""

    def __init__(
        self,
        indexer: Indexer,
        pattern: str = "**/*.*",
        ignore_patterns: list[str] | None = None,
    ):
        """Initialize the event handler.

        Args:
            indexer: The indexer to update
            pattern: Glob pattern for files to index
            ignore_patterns: List of glob patterns to ignore
        """
        self.indexer = indexer
        self.pattern = pattern
        self.ignore_patterns = ignore_patterns or [".git", "__pycache__", "*.pyc"]
        self._pending_updates: set[Path] = set()
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

    def _should_process(self, path: str) -> bool:
        """Check if a file should be processed based on pattern and ignore patterns."""
        path_obj = Path(path)
        return path_obj.match(self.pattern) and not any(
            path_obj.match(pattern) for pattern in self.ignore_patterns
        )

    def _queue_update(self, path: Path) -> None:
        """Queue a file for update, applying debouncing."""
        logger.debug(f"Queueing update for {path}")
        self._pending_updates.add(path)

        # Always process updates after a delay to ensure file is written
        time.sleep(self._update_delay)
        self._process_updates()
        logger.debug(f"Processed update for {path}")

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move events."""
        if not event.is_directory:
            logger.info(f"File moved: {event.src_path} -> {event.dest_path}")
            src_path = Path(event.src_path).absolute()
            dest_path = Path(event.dest_path).absolute()

            # Remove old file from index if it was being tracked
            if self._should_process(event.src_path):
                logger.debug(f"Removing old path from index: {src_path}")
                old_docs = self.indexer.search(
                    "", n_results=100, where={"source": str(src_path)}
                )[0]
                for doc in old_docs:
                    if doc.doc_id is not None:
                        self.indexer.delete_document(doc.doc_id)
                        logger.debug(f"Deleted old document: {doc.doc_id}")

            # Index the file at its new location if it matches our patterns
            if self._should_process(event.dest_path):
                logger.info(f"Indexing moved file at new location: {dest_path}")

                # Wait for the file to be fully moved and readable
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        # Try to read the file to ensure it's ready
                        content = dest_path.read_text()
                        # Index the file
                        self.indexer.index_file(dest_path)

                        # Verify the update with content-based search
                        results, _, _ = self.indexer.search(
                            content[:50]
                        )  # Search by content prefix
                        if results and any(
                            str(dest_path) == doc.metadata.get("source")
                            for doc in results
                        ):
                            logger.info(
                                f"Successfully verified moved file: {dest_path}"
                            )
                            break
                        elif attempt < max_attempts - 1:
                            logger.warning(
                                f"Verification failed, retrying... ({attempt + 1}/{max_attempts})"
                            )
                            time.sleep(0.5)  # Wait before retry
                        else:
                            logger.error(
                                f"Failed to verify moved file after {max_attempts} attempts: {dest_path}"
                            )
                    except Exception as e:
                        if attempt < max_attempts - 1:
                            logger.warning(
                                f"Error processing moved file (attempt {attempt + 1}): {e}"
                            )
                            time.sleep(0.5)  # Wait before retry
                        else:
                            logger.error(
                                f"Failed to process moved file after {max_attempts} attempts: {e}"
                            )

    def _should_skip_file(self, path: Path, processed_paths: set[str]) -> bool:
        """Check if a file should be skipped during processing."""
        canonical_path = str(path.resolve())
        if (
            canonical_path in processed_paths
            or not path.is_file()
            or path.suffix in {".sqlite3", ".db", ".bin", ".pyc"}
        ):
            logger.debug(f"Skipping {canonical_path} (already processed or binary)")
            return True
        return False

    def _update_index_with_retries(
        self, path: Path, content: str, max_attempts: int = 3
    ) -> bool:
        """Update index for a file with retries."""
        canonical_path = str(path.resolve())

        # Delete old versions
        self.indexer.delete_documents({"source": canonical_path})
        logger.debug(f"Cleared old versions for: {canonical_path}")

        # Try indexing with verification
        for attempt in range(max_attempts):
            logger.info(f"Indexing attempt {attempt + 1} for {path}")
            self.indexer.index_file(path)

            if self.indexer.verify_document(path, content=content):
                logger.info(f"Successfully verified index update for {path}")
                return True

            if attempt < max_attempts - 1:
                logger.warning(
                    f"Verification failed, retrying... ({attempt + 1}/{max_attempts})"
                )
                time.sleep(0.5)

        logger.error(
            f"Failed to verify index update after {max_attempts} attempts for {path}"
        )
        return False

    def _process_single_update(self, path: Path, processed_paths: set[str]) -> None:
        """Process a single file update.

        Args:
            path: Path to the file to process
            processed_paths: Set of already processed canonical paths
        """
        """Process a single file update."""
        if self._should_skip_file(path, processed_paths):
            return

        # Wait to ensure file is fully written
        time.sleep(0.2)

        try:
            if path.exists():
                # Read current content for verification
                current_content = path.read_text()

                # Update index
                if self._update_index_with_retries(path, current_content):
                    processed_paths.add(str(path.resolve()))

        except Exception as e:
            logger.error(f"Error processing update for {path}: {e}", exc_info=True)

    def _process_updates(self) -> None:
        """Process all pending updates."""
        if not self._pending_updates:
            logger.debug("No pending updates to process")
            return

        # Get all pending updates that still exist
        existing_updates = [p for p in self._pending_updates if p.exists()]
        if not existing_updates:
            logger.debug("No existing files to process")
            return

        logger.info(f"Processing {len(existing_updates)} updates")

        # Sort updates by modification time to get latest versions
        updates = sorted(
            existing_updates, key=lambda p: p.stat().st_mtime, reverse=True
        )
        logger.debug(f"Sorted updates: {[str(p) for p in updates]}")

        # Process only the latest version of each file
        processed_paths: set[str] = set()
        for path in updates:
            self._process_single_update(path, processed_paths)

        self._pending_updates.clear()
        self._last_update = time.time()
        logger.info("Finished processing updates")


class FileWatcher:
    """Watch files and update the index automatically."""

    def __init__(
        self,
        indexer: Indexer,
        paths: list[str],
        pattern: str = "**/*.*",
        ignore_patterns: list[str] | None = None,
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
        # Reset collection once before starting
        self.indexer.reset_collection()
        logger.debug("Reset collection before starting watcher")

        # First index existing files
        for path in self.paths:
            if not path.exists():
                logger.warning(f"Watch path does not exist: {path}")
                continue

            # Index existing files
            try:
                self.indexer.index_directory(path, self.event_handler.pattern)
                logger.debug(f"Indexed existing files in {path}")
            except Exception as e:
                logger.error(f"Error indexing directory {path}: {e}", exc_info=True)

            # Set up watching
            try:
                self.observer.schedule(self.event_handler, str(path), recursive=True)
                logger.debug(f"Scheduled observer for {path}")
            except Exception as e:
                logger.error(
                    f"Error scheduling observer for {path}: {e}", exc_info=True
                )

        # Start the observer
        try:
            self.observer.start()
            # Wait a bit to ensure the observer is ready
            time.sleep(0.5)  # Increased wait time for better stability
            logger.info(
                f"Started watching paths: {', '.join(str(p) for p in self.paths)}"
            )
        except Exception as e:
            logger.error(f"Error starting observer: {e}", exc_info=True)
            raise

    def stop(self) -> None:
        """Stop watching for file changes."""
        self.observer.stop()
        self.observer.join()
        logger.info("Stopped file watcher")

    def __enter__(self) -> "FileWatcher":
        """Start watching when used as context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop watching when exiting context manager."""
        self.stop()
