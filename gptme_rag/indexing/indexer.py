from pathlib import Path
from typing import List, Optional, Iterator
import chromadb
from chromadb.config import Settings
from chromadb.api import Collection

from .document import Document

class Indexer:
    """Handles document indexing and embedding storage."""
    
    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: str = "gptme_docs"
    ):
        if persist_directory:
            persist_directory = Path(persist_directory).resolve()
            persist_directory.mkdir(parents=True, exist_ok=True)
            print(f"Using persist directory: {persist_directory}")
            
        settings = Settings()
        if persist_directory:
            settings.persist_directory = str(persist_directory)
            settings.allow_reset = True  # Allow resetting for testing
            settings.is_persistent = True
            
        print(f"Creating ChromaDB client with settings: {settings}")
        self.client = chromadb.PersistentClient(path=str(persist_directory) if persist_directory else None)
        
        try:
            print(f"Getting or creating collection: {collection_name}")
            self.collection: Collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Collection created/retrieved. Count: {self.collection.count()}")
        except Exception as e:
            print(f"Error creating collection, resetting: {e}")
            self.client.reset()
            self.collection: Collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
    def __del__(self):
        """Cleanup when the indexer is destroyed."""
        try:
            self.client.reset()
        except:
            pass
    
    def add_document(self, document: Document) -> None:
        """Add a single document to the index."""
        if not document.doc_id:
            document.doc_id = str(hash(document.content))
            
        self.collection.add(
            documents=[document.content],
            metadatas=[document.metadata],
            ids=[document.doc_id]
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add multiple documents to the index."""
        contents = []
        metadatas = []
        ids = []
        
        for doc in documents:
            if not doc.doc_id:
                doc.doc_id = str(hash(doc.content))
            contents.append(doc.content)
            metadatas.append(doc.metadata)
            ids.append(doc.doc_id)
            
        self.collection.add(
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
    
    def index_directory(
        self,
        directory: Path,
        glob_pattern: str = "**/*.*"
    ) -> None:
        """Index all files in a directory matching the glob pattern."""
        directory = directory.resolve()  # Convert to absolute path
        files = list(directory.glob(glob_pattern))
        print(f"Found {len(files)} files in {directory}:")
        for f in files:
            print(f"  {f.relative_to(directory)}")
        
        documents = [Document.from_file(f) for f in files if f.is_file()]
        if not documents:
            raise ValueError(f"No documents found in {directory} with pattern {glob_pattern}")
        self.add_documents(documents)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None
    ) -> tuple[List[Document], dict]:
        """Search for documents similar to the query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        documents = []
        for i, doc_id in enumerate(results["ids"][0]):
            doc = Document(
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                doc_id=doc_id
            )
            documents.append(doc)
            
        return documents, results
