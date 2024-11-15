"""Streaming document processor for efficient handling of large documents."""

import logging
from pathlib import Path
from typing import Generator, List, Optional, Union

import tiktoken

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process documents in a streaming fashion to manage memory efficiently."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_chunks: Optional[int] = None,
        encoding_name: str = "cl100k_base",
    ):
        """Initialize the document processor.

        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            max_chunks: Maximum number of chunks to process (None for unlimited)
            encoding_name: Name of the tiktoken encoding to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks
        self.encoding = tiktoken.get_encoding(encoding_name)

    def process_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> Generator[dict, None, None]:
        """Process text into chunks with metadata.

        Args:
            text: Text to process
            metadata: Optional metadata to include with each chunk

        Yields:
            Dict containing chunk text and metadata
        """
        # Encode text to tokens
        tokens = self.encoding.encode(text)
        
        # Process in chunks
        chunk_start = 0
        chunk_count = 0
        
        while chunk_start < len(tokens):
            # Check if we've reached max chunks
            if self.max_chunks and chunk_count >= self.max_chunks:
                break
                
            # Calculate chunk end
            chunk_end = min(chunk_start + self.chunk_size, len(tokens))
            
            # Decode chunk
            chunk_tokens = tokens[chunk_start:chunk_end]
            chunk_text = self.encoding.decode(chunk_tokens)
            
        # Skip empty text
        if not text.strip():
            return
            
        try:
            # Encode text to tokens
            tokens = self.encoding.encode(text)
            
            # Calculate total chunks
            total_chunks = max(1, self.estimate_chunks(len(tokens)))
            
            # If text is short enough, yield as single chunk
            if len(tokens) <= self.chunk_size:
                yield {
                    "text": text,
                    "metadata": {
                        **(metadata or {}),
                        "chunk_index": 0,
                        "token_count": len(tokens),
                        "total_chunks": 1,
                        "chunk_start": 0,
                        "chunk_end": len(tokens),
                    }
                }
            else:
                # Process in chunks
                chunk_start = 0
                chunk_count = 0
                
                while chunk_start < len(tokens):
                    # Calculate chunk end
                    chunk_end = min(chunk_start + self.chunk_size, len(tokens))
                    
                    # Decode chunk
                    chunk_tokens = tokens[chunk_start:chunk_end]
                    chunk_text = self.encoding.decode(chunk_tokens)
                    
                    # Create chunk metadata
                    chunk_metadata = {
                        **(metadata or {}),
                        "chunk_index": chunk_count,
                        "token_count": len(chunk_tokens),
                        "total_chunks": total_chunks,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                    }
                    
                    yield {
                        "text": chunk_text,
                        "metadata": chunk_metadata,
                    }
                    
                    # Move to next chunk
                    chunk_start = chunk_end - self.chunk_overlap
                    chunk_count += 1
                    
                    # Stop if we've reached max chunks
                    if self.max_chunks and chunk_count >= self.max_chunks:
                        break
                    
                    # Stop if remaining text is too small
                    if len(tokens) - chunk_start <= self.chunk_overlap:
                        break
                    
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            # Yield the entire text as a single chunk if processing fails
            yield {
                "text": text,
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": 0,
                    "token_count": len(text),
                    "total_chunks": 1,
                    "chunk_start": 0,
                    "chunk_end": len(text),
                    "error": str(e),
                }
            }
            
            yield {
                "text": chunk_text,
                "metadata": chunk_metadata,
            }
            
            # Move to next chunk
            chunk_start = chunk_end - self.chunk_overlap
            chunk_count += 1
            
            # Check if we're near the end
            remaining_tokens = len(tokens) - chunk_start
            
            if remaining_tokens <= self.chunk_overlap:
                # We've processed everything
                break
            elif remaining_tokens < self.chunk_size:
                # Create final chunk with remaining tokens
                chunk_end = len(tokens)
                chunk_tokens = tokens[chunk_start:chunk_end]
                chunk_text = self.encoding.decode(chunk_tokens)
                
                chunk_metadata = {
                    **(metadata or {}),
                    "chunk_index": chunk_count,
                    "token_count": len(chunk_tokens),
                    "total_chunks": total_chunks,
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "is_final_chunk": True,
                }
                
                yield {
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                }
                break

    def is_binary_file(self, file_path: Path) -> bool:
        """Check if a file is binary.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file appears to be binary
        """
        # Check file extension first
        binary_extensions = {'.db', '.sqlite3', '.bin', '.pyc', '.so', '.dll', '.exe'}
        if file_path.suffix.lower() in binary_extensions:
            return True
            
        # Read first chunk and check for null bytes
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\x00' in chunk
        except Exception:
            return True

    def process_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[dict] = None,
        encoding: str = "utf-8",
    ) -> Generator[dict, None, None]:
        """Process a file in chunks.

        Args:
            file_path: Path to the file
            metadata: Optional metadata to include with each chunk
            encoding: File encoding

        Yields:
            Dict containing chunk text and metadata
        """
        path = Path(file_path)
        
        # Skip binary files
        if self.is_binary_file(path):
            return
            
        file_metadata = {
            "filename": path.name,
            "file_path": str(path.absolute()),
            "file_type": path.suffix.lstrip("."),
            **(metadata or {}),
        }
        
        try:
            # Read the entire file content
            content = path.read_text(encoding=encoding)
            
            # Skip empty files
            if not content.strip():
                return
                
            # Process the content
            yield from self.process_text(content, file_metadata)
        except UnicodeDecodeError:
            # Skip files that can't be decoded as text
            return

    def estimate_token_count(self, text: str) -> int:
        """Estimate the number of tokens in a text.

        Args:
            text: Text to estimate

        Returns:
            Estimated number of tokens
        """
        return len(self.encoding.encode(text))

    def estimate_chunks(self, token_count: int) -> int:
        """Estimate the number of chunks for a given token count.

        Args:
            token_count: Number of tokens

        Returns:
            Estimated number of chunks
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        effective_chunk_size = self.chunk_size - self.chunk_overlap
        chunks = (token_count + effective_chunk_size - 1) // effective_chunk_size
        
        if self.max_chunks:
            chunks = min(chunks, self.max_chunks)
        
        return chunks

    def get_optimal_chunk_size(
        self,
        target_chunks: int,
        token_count: int,
        min_chunk_size: int = 100,
    ) -> int:
        """Calculate optimal chunk size to achieve target number of chunks.

        Args:
            target_chunks: Desired number of chunks
            token_count: Total number of tokens
            min_chunk_size: Minimum chunk size

        Returns:
            Optimal chunk size
        """
        if target_chunks <= 0:
            raise ValueError("Target chunks must be positive")
        
        # Calculate base chunk size
        base_size = token_count // target_chunks
        
        # Adjust for overlap
        if self.chunk_overlap > 0:
            # Solve: (token_count) / (chunk_size - overlap) = target_chunks
            adjusted_size = (token_count // target_chunks) + self.chunk_overlap
        else:
            adjusted_size = base_size
        
        # Ensure minimum size
        return max(adjusted_size, min_chunk_size)
