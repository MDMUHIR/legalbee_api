"""Embedding service using BGE-M3 for query vector generation."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from app.config import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings for queries using BGE-M3."""

    def __init__(self, model_name: str = ""):
        self.model_name = model_name or config.embedding_model
        self._model: Optional[object] = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
        return self._model

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=len(texts),
        )
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
