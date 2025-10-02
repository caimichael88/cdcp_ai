from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class EmbeddingService:
    """
    Service for generating embeddings from text chunks using SentenceTransformers.
    Uses CPU mode to avoid MPS segfaults on Apple Silicon.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

        # Force CPU mode to avoid segfault on Apple Silicon
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

        print(f"Loading embedding model: {model_name} (CPU mode)")

        # Use SentenceTransformer instead of raw transformers
        self.model = SentenceTransformer(model_name, device='cpu')

        print(f"✓ Embedding model loaded successfully")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of text documents."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.model.encode(query, convert_to_numpy=True)
