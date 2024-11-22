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
                        results, _ = self.indexer.search(
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
        processed_paths = set()
        for path in updates:
            try:
                canonical_path = str(path.resolve())
                logger.debug(f"Processing update for {canonical_path}")

                # Skip if already processed or if it's a binary file
                if (
                    canonical_path in processed_paths
                    or not path.is_file()
                    or path.suffix in {".sqlite3", ".db", ".bin", ".pyc"}
                ):
                    logger.debug(
                        f"Skipping {canonical_path} (already processed or binary)"
                    )
                    continue

                # Wait to ensure file is fully written
                time.sleep(0.2)

                # Get current content to ensure we have the latest version
                if path.exists():
                    try:
                        # Read current content for verification
                        current_content = path.read_text()

                        # Clear old versions and index new version atomically
                        try:
                            # Delete old versions
                            self.indexer.delete_documents({"source": canonical_path})
                            logger.debug(f"Cleared old versions for: {canonical_path}")

                            # Index the new version immediately to maintain atomicity
                            max_attempts = 3
                            for attempt in range(max_attempts):
                                logger.info(
                                    f"Indexing attempt {attempt + 1} for {path}"
                                )
                                self.indexer.index_file(path)

                                # Verify the update
                                if self.indexer.verify_document(
                                    path, content=current_content
                                ):
                                    processed_paths.add(canonical_path)
                                    logger.info(
                                        f"Successfully verified index update for {path}"
                                    )
                                    break
                                elif attempt < max_attempts - 1:
                                    logger.warning(
                                        f"Verification failed, retrying... ({attempt + 1}/{max_attempts})"
                                    )
                                    time.sleep(0.5)  # Wait before retry
                                else:
                                    logger.error(
                                        f"Failed to verify index update after {max_attempts} attempts for {path}"
                                    )
                        except Exception as e:
                            logger.error(
                                f"Error updating index for {path}: {e}", exc_info=True
                            )
                        for attempt in range(max_attempts):
                            logger.info(f"Indexing attempt {attempt + 1} for {path}")
                            self.indexer.index_file(path)

                            # Verify the update
                            if self.indexer.verify_document(
                                path, content=current_content
                            ):
                                processed_paths.add(canonical_path)
                                logger.info(
                                    f"Successfully verified index update for {path}"
                                )
                                break
                            elif attempt < max_attempts - 1:
                                logger.warning(
                                    f"Verification failed, retrying... ({attempt + 1}/{max_attempts})"
                                )
                                time.sleep(0.5)  # Wait before retry
                            else:
                                logger.error(
                                    f"Failed to verify index update after {max_attempts} attempts for {path}"
                                )

                    except Exception as e:
                        logger.error(
                            f"Error updating index for {path}: {e}", exc_info=True
                        )
                        continue

            except Exception as e:
                logger.error(f"Error processing update for {path}: {e}", exc_info=True)

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
        # First index existing files
        for path in self.paths:
            if not path.exists():
                logger.warning(f"Watch path does not exist: {path}")
                continue
            # Reset collection before starting
            self.indexer.reset_collection()
            logger.debug("Reset collection before starting watcher")

            # Index existing files
            self.indexer.index_directory(path, self.event_handler.pattern)
            logger.debug(f"Indexed existing files in {path}")
            # Set up watching
            self.observer.schedule(self.event_handler, str(path), recursive=True)

        self.observer.start()
        # Wait a bit to ensure the observer is ready
        time.sleep(0.2)
        logger.info(f"Started watching paths: {', '.join(str(p) for p in self.paths)}")

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
