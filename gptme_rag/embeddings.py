from chromadb.api.types import Documents, EmbeddingFunction
from sentence_transformers import SentenceTransformer


class ModernBERTEmbedding(EmbeddingFunction):
    def __init__(
        self, model_name: str = "joe32140/ModernBERT-base-msmarco", device: str = "cpu"
    ):
        """Initialize ModernBERT embedding function.

        Args:
            model_name: Name of the ModernBERT model to use. Options:
                - "joe32140/ModernBERT-base-msmarco" (default, optimized for retrieval)
                  Best for search/retrieval tasks, trained with contrastive learning on MS MARCO.
                  Recommended chunk size: 512-1024 tokens for general text, 256-512 for code.
                - "answerdotai/ModernBERT-base" (general purpose)
                  Better for tasks requiring deeper semantic understanding.
                  Can handle longer chunks (up to 8192 tokens).
            device: Device to run the model on (defaults to 'cpu')

        Note:
            The msmarco variant is specifically optimized for retrieval tasks and should give
            better results for search/similarity use cases. It works best with smaller chunk
            sizes as it's trained on passage-level data.
        """
        self.model_name = model_name
        self.is_msmarco = "msmarco" in model_name.lower()
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, texts: Documents) -> list[list[float]]:
        """Generate embeddings for the input texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        # Batch inputs for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=32,  # Adjust based on GPU memory
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
        ).tolist()
        return embeddings
