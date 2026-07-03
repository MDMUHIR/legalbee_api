"""BGE-M3 embedding generation with batching, retry, and failure handling.

Uses sentence-transformers for BAAI/bge-m3 which produces 1024-dimensional
dense vectors optimized for multilingual (Bengali + English) legal text.

The BGE-M3 tokenizer handles up to 8192 tokens per input.
"""

import time
import logging
from typing import Optional, Iterator

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0


class EmbeddingGenerator:
    """Generate embeddings using BAAI/bge-m3 via sentence-transformers.

    Features:
      - Lazy model loading (only when first embedding is requested)
      - Configurable batch size for memory-efficient processing
      - Retry logic with exponential backoff on transient failures
      - Progress reporting during batch embedding
      - Safe handling of empty inputs
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model: Optional[object] = None

    @property
    def model(self):
        """Load model on first access (lazy initialization)."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        """Load BGE-M3 model with optional GPU device selection."""
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self.model_name)
        start = time.time()

        kwargs = {}
        if self.device:
            kwargs["device"] = self.device

        model = SentenceTransformer(self.model_name, trust_remote_code=True, **kwargs)

        elapsed = time.time() - start
        logger.info(
            "Model loaded in %.1fs | dim=%s | device=%s",
            elapsed,
            model.get_sentence_embedding_dimension(),
            str(model.device),
        )
        return model

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for a list of text strings.

        Args:
            texts: List of strings to embed.

        Returns:
            List of numpy arrays, each of shape (1024,).

        Raises:
            RuntimeError: If embedding fails after all retries.
        """
        return list(self.embed_generator(texts))

    def embed_generator(self, texts: list[str]) -> Iterator[np.ndarray]:
        """Yield embeddings one batch at a time for memory efficiency.

        Args:
            texts: List of strings to embed.

        Yields:
            Numpy array per text, preserving input order.
        """
        if not texts:
            return

        total = len(texts)
        batched = 0

        for batch_idx in range(0, total, self.batch_size):
            batch = texts[batch_idx : batch_idx + self.batch_size]
            valid_batch = [t if t else " " for t in batch]

            embeddings = self._embed_with_retry(valid_batch)

            for emb in embeddings:
                yield np.array(emb, dtype=np.float32)

            batched += len(batch)
            pct = min(100, int(batched / total * 100))
            logger.debug("Embedding progress: %d/%d (%d%%)", batched, total, pct)

    def _embed_with_retry(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch with retry logic for transient failures.

        BGE-M3 can be sensitive to very long inputs (>8192 tokens).
        If a batch fails, we try with individually truncated texts.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                embeddings = self.model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=len(texts),
                )
                return [np.array(e, dtype=np.float32) for e in embeddings]

            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY_SECONDS * attempt
                    logger.warning(
                        "Embedding batch failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt,
                        MAX_RETRIES,
                        e,
                        delay,
                    )
                    time.sleep(delay)

                    texts = self._truncate_texts(texts)
                else:
                    logger.error("Embedding failed after %d attempts: %s", MAX_RETRIES, e)

        raise RuntimeError(
            f"Embedding failed after {MAX_RETRIES} attempts. Last error: {last_error}"
        )

    def _truncate_texts(self, texts: list[str], max_chars: int = 6000) -> list[str]:
        """Truncate texts to avoid tokenizer overflow in retry scenario."""
        return [t[:max_chars] if len(t) > max_chars else t for t in texts]

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        result = list(self.embed_generator([text]))
        return result[0] if result else np.array([], dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (1024 for BGE-M3)."""
        return self.model.get_sentence_embedding_dimension()
