#!/usr/bin/env python3
"""Simple example of file watching with gptme-rag."""

import sys
from pathlib import Path

from gptme_rag.indexing.indexer import Indexer
from gptme_rag.indexing.watcher import FileWatcher
from rich.console import Console

console = Console()


def main():
    """Run file watcher."""
    try:
        # Create indexer
        indexer = Indexer(persist_directory=Path("watch_index"))
        
        # Get current directory to watch
        watch_dir = Path.cwd()
        
        console.print(f"Watching directory: {watch_dir}")
        console.print("Try:")
        console.print("- Adding .md files")
        console.print("- Modifying existing files")
        console.print("- Searching using another terminal")
        console.print("\nPress Ctrl+C to stop")

        # Start watching
        with FileWatcher(
            indexer=indexer,
            paths=[str(watch_dir)],
            pattern="**/*.md",
        ) as watcher:
            try:
                # Keep running until interrupted
                import signal
                try:
                    signal.pause()
                except AttributeError:  # Windows
                    while True:
                        import time
                        time.sleep(1)
            except KeyboardInterrupt:
                console.print("\nStopping watcher...")

        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
