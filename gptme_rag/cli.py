import click
from pathlib import Path
from rich.console import Console
from rich.table import Table

from .indexing.indexer import Indexer
from .query.context_assembler import ContextAssembler

console = Console()

@click.group()
def cli():
    """RAG implementation for gptme context management."""
    pass

@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--pattern', '-p', default="**/*.*", help="Glob pattern for files to index")
@click.option('--persist-dir', type=click.Path(path_type=Path), help="Directory to persist the index")
def index(directory: Path, pattern: str, persist_dir: Optional[Path]):
    """Index documents in a directory."""
    try:
        indexer = Indexer(persist_directory=persist_dir)
        with console.status(f"Indexing {directory}..."):
            indexer.index_directory(directory, pattern)
        console.print(f"✅ Successfully indexed {directory}", style="green")
    except Exception as e:
        console.print(f"❌ Error indexing directory: {e}", style="red")

@cli.command()
@click.argument('query')
@click.option('--n-results', '-n', default=5, help="Number of results to return")
@click.option('--persist-dir', type=click.Path(path_type=Path), help="Directory to persist the index")
@click.option('--max-tokens', default=4000, help="Maximum tokens in context window")
def search(query: str, n_results: int, persist_dir: Optional[Path], max_tokens: int):
    """Search the index and assemble context."""
    try:
        indexer = Indexer(persist_directory=persist_dir)
        assembler = ContextAssembler(max_tokens=max_tokens)
        
        # Search for relevant documents
        with console.status("Searching..."):
            documents = indexer.search(query, n_results=n_results)
        
        # Create result table
        table = Table(title="Search Results")
        table.add_column("Source")
        table.add_column("Content Preview")
        table.add_column("Score", justify="right")
        
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
            score = doc.metadata.get("score", "N/A")
            table.add_row(source, preview, str(score))
            
        console.print(table)
        
        # Assemble context window
        context = assembler.assemble_context(documents, user_query=query)
        console.print("\n[bold]Assembled Context:[/bold]")
        console.print(f"Total tokens: {context.total_tokens}")
        console.print(f"Truncated: {context.truncated}")
        
        if click.confirm("Show full context?"):
            console.print(context.content)
            
    except Exception as e:
        console.print(f"❌ Error searching index: {e}", style="red")

if __name__ == '__main__':
    cli()
