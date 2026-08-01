"""
Shared embedding model registry — single SentenceTransformer instance
shared across EmbeddingStore, QueryClassifier, QueryCache, and QueryRewriter.

Avoids loading the same model 3× (saving ~940MB RAM/VRAM).
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

_embedding_model = None


def get_embedding_model():
    """Return the shared SentenceTransformer instance (lazy init, thread-safe enough for single-worker)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        import torch
        import os

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        local_only = os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "false").lower() == "true"

        logger.info("[MODEL-REGISTRY] Loading shared embedding model on %s: %s", device, model_name)
        _embedding_model = SentenceTransformer(model_name, device=device, local_files_only=local_only)

    return _embedding_model


class SharedEmbeddings:
    """
    LangChain-compatible embeddings wrapper around the shared SentenceTransformer.

    Drop-in replacement for ``HuggingFaceEmbeddings`` so ChromaDB can use it.
    """

    def __init__(self):
        self._model = get_embedding_model()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(
            [text],
            normalize_embeddings=True,
        )[0].tolist()
