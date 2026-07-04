"""Qdrant cloud service wrapper."""

from __future__ import annotations

import logging
from typing import Optional
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText

from app.config import config

logger = logging.getLogger(__name__)


class QdrantService:
    """Singleton wrapper around QdrantClient for the bangladesh_laws collection."""

    def __init__(self):
        self.url = config.qdrant_url
        self.api_key = config.qdrant_api_key
        self.collection = config.collection_name
        self.vector_name = config.dense_vector_name
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=self.url, api_key=self.api_key)
        return self._client

    def search(
        self,
        vector: list[float],
        limit: int = 8,
        score_threshold: float = 0.3,
        query_filter: Optional[Filter] = None,
    ) -> list[dict]:
        try:
            from qdrant_client.models import QueryRequest
            results = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                using=self.vector_name,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )
            return [
                {
                    "id": str(r.id),
                    "score": r.score,
                    "payload": r.payload or {},
                }
                for r in results.points
            ]
        except Exception as e:
            logger.error("Qdrant search failed: %s", e)
            raise

    def search_with_filter(
        self,
        vector: list[float],
        filter_dict: dict,
        limit: int = 8,
        score_threshold: float = 0.3,
    ) -> list[dict]:
        conditions = []
        for key, value in filter_dict.items():
            if key.startswith("hierarchy."):
                conditions.append(
                    FieldCondition(key=f"metadata.{key}", match=MatchValue(value=str(value)))
                )
            elif key in ("act_name", "year", "document_type", "language", "source_pdf", "chunk_type"):
                field = f"metadata.{key}" if key != "chunk_type" else "chunk_type"
                if key == "year":
                    conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=int(value)))
                    )
                else:
                    conditions.append(
                        FieldCondition(key=field, match=MatchText(text=str(value)))
                    )

        query_filter = Filter(must=conditions) if conditions else None
        return self.search(vector, limit=limit, score_threshold=score_threshold, query_filter=query_filter)

    def collection_info(self) -> dict:
        try:
            info = self.client.get_collection(self.collection)
            return {
                "name": self.collection,
                "points_count": getattr(info, "points_count", 0),
                "status": str(getattr(info, "status", "unknown")),
            }
        except Exception as e:
            logger.error("Failed to get collection info: %s", e)
            return {"error": str(e)}

    def close(self):
        if self._client:
            self._client.close()
            self._client = None


@lru_cache(maxsize=1)
def get_qdrant_service() -> QdrantService:
    return QdrantService()
