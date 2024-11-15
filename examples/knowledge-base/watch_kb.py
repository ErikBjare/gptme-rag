#!/usr/bin/env python3
"""Knowledge base watcher script demonstrating live indexing."""

import sys
from pathlib import Path

from gptme_rag.indexing.indexer import Indexer
from gptme_rag.indexing.watcher import FileWatcher
from rich.console import Console

console = Console()


def main():
    """Run the knowledge base watcher."""
    try:
        # Get knowledge base directory
        kb_dir = Path(__file__).parent / "content"
        if not kb_dir.exists():
            console.print(f"[red]Error: Knowledge base directory not found: {kb_dir}")
            return 1

        # Create indexer with chunking enabled
        index_dir = Path("kb_index")
        indexer = Indexer(
            persist_directory=index_dir,
            chunk_size=500,
            chunk_overlap=50,
        )

        # Initial indexing
        console.print(f"Performing initial indexing of {kb_dir}")
        with console.status("Indexing..."):
            indexer.index_directory(kb_dir, glob_pattern="**/*.md")

        # Start file watcher
        console.print("\nWatching for changes... (Ctrl+C to exit)")
        with FileWatcher(
            indexer=indexer,
            paths=[str(kb_dir)],
            pattern="**/*.md",
            ignore_patterns=[".git", "__pycache__", "*.pyc"],
        ) as _watcher:
            try:
                # Keep running until interrupted
                import signal

                signal.pause()
            except (KeyboardInterrupt, AttributeError):
                # Handle both Ctrl+C and Windows (no signal.pause)
                import time

                while True:
                    try:
                        time.sleep(1)
                    except KeyboardInterrupt:
                        break

        console.print("\nStopping file watcher...")
        return 0

    except Exception as e:
        console.print(f"[red]Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
