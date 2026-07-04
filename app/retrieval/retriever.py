"""Qdrant retriever: dense search with optional metadata filtering."""

from __future__ import annotations

import logging
import time
from typing import Optional

from app.config import config
from app.services.qdrant_service import get_qdrant_service
from app.services.embedding_service import get_embedding_service
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class LegalRetriever:
    """Retrieve legal chunks from Qdrant using BGE-M3 dense search."""

    def __init__(self):
        self.qdrant = get_qdrant_service()
        self.embedder = get_embedding_service()
        self.top_k = config.top_k
        self.score_threshold = config.score_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 0,
        score_threshold: float = 0.0,
        metadata_filter: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        t0 = time.time()

        if top_k <= 0:
            top_k = self.top_k
        if score_threshold <= 0:
            score_threshold = self.score_threshold

        vector = self.embedder.embed(query)

        if metadata_filter:
            results = self.qdrant.search_with_filter(
                vector=vector,
                filter_dict=metadata_filter,
                limit=top_k,
                score_threshold=score_threshold,
            )
        else:
            results = self.qdrant.search(
                vector=vector,
                limit=top_k,
                score_threshold=score_threshold,
            )

        elapsed_ms = (time.time() - t0) * 1000
        logger.info(
            "Retrieved %d chunks for query '%s...' in %.0f ms",
            len(results),
            query[:80],
            elapsed_ms,
        )

        return [self._to_chunk(r) for r in results]

    def _to_chunk(self, result: dict) -> RetrievedChunk:
        p = result.get("payload", {})
        meta = p.get("metadata", {})

        return RetrievedChunk(
            text=p.get("text", "") or p.get("page_content", ""),
            chunk_id=result.get("id", "") or p.get("chunk_id", ""),
            score=result.get("score", 0.0),
            chunk_type=p.get("chunk_type", ""),
            citation=p.get("citation", ""),
            act_name=meta.get("act_name", "") or meta.get("bangla_name", ""),
            year=meta.get("year", 0),
            hierarchy=meta.get("hierarchy", {}),
            references=p.get("references", []),
        )

    def retrieve_by_act(
        self,
        act_name: str,
        top_k: int = 0,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks from a specific act."""
        return self.retrieve(
            query=act_name,
            top_k=top_k or self.top_k * 2,
            metadata_filter={"act_name": act_name},
        )

    def retrieve_by_section(
        self,
        section_number: str,
        act_name: str = "",
        top_k: int = 0,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks for a specific section, optionally within an act."""
        filters: dict = {"hierarchy.section": section_number}
        if act_name:
            filters["act_name"] = act_name
        return self.retrieve(
            query=f"Section {section_number} {act_name}",
            top_k=top_k or self.top_k,
            metadata_filter=filters,
        )
