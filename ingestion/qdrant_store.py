"""Qdrant vector store integration for the Bangladeshi law ingestion pipeline.

Manages collection lifecycle (create/ensure), payload indexing, and batch upsert
with clean payload structure optimized for legal RAG retrieval.
"""

import os
import uuid
import logging
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "bangladesh_laws"
DENSE_VECTOR_NAME = "law_dense_vector"
VECTOR_SIZE = 1024


class QdrantStore:
    """Manage Qdrant collection and batch upsert for law document chunks.

    Creates a collection with:
      - 1024-dim dense vectors (BGE-M3 output)
      - Sparse vector placeholder for hybrid search compatibility
      - Payload indexes on key metadata fields for filtered queries
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = COLLECTION_NAME,
        vector_size: int = VECTOR_SIZE,
    ):
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        self.vector_size = vector_size

        if not self.url or not self.api_key:
            raise ValueError(
                "Qdrant credentials required. "
                "Set QDRANT_URL and QDRANT_API_KEY environment variables."
            )

        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        logger.info("QdrantStore: connected to %s", self.url)

    def ensure_collection(self) -> None:
        """Create the collection if it does not exist.

        Configures dense vectors only (1024-dim, Cosine) for BGE-M3 embeddings.
        Payload indexes on metadata fields use dot-notation for nested access.
        """
        if self.client.collection_exists(self.collection_name):
            logger.info("Collection '%s' already exists", self.collection_name)
            self._ensure_payload_indexes()
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            },
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=2,
            ),
        )

        self._ensure_payload_indexes()
        logger.info(
            "Collection '%s' created | vectors=%d dim | dense=%s",
            self.collection_name,
            self.vector_size,
            DENSE_VECTOR_NAME,
        )

    def _ensure_payload_indexes(self) -> None:
        """Create payload indexes for efficient filtered searches.

        Uses dot-notation for nested metadata fields (e.g., 'metadata.act_name').
        """
        indexed_fields: list[tuple[str, PayloadSchemaType]] = [
            ("metadata.act_name", PayloadSchemaType.KEYWORD),
            ("metadata.bangla_name", PayloadSchemaType.KEYWORD),
            ("metadata.year", PayloadSchemaType.INTEGER),
            ("metadata.act_number", PayloadSchemaType.KEYWORD),
            ("metadata.document_type", PayloadSchemaType.KEYWORD),
            ("metadata.source_pdf", PayloadSchemaType.KEYWORD),
            ("metadata.language", PayloadSchemaType.KEYWORD),
            ("metadata.hierarchy.chapter", PayloadSchemaType.KEYWORD),
            ("metadata.hierarchy.article", PayloadSchemaType.KEYWORD),
            ("metadata.hierarchy.section", PayloadSchemaType.KEYWORD),
            ("metadata.hierarchy.clause", PayloadSchemaType.KEYWORD),
            ("metadata.hierarchy.sub_clause", PayloadSchemaType.KEYWORD),
            ("chunk_type", PayloadSchemaType.KEYWORD),
        ]

        for field_name, schema_type in indexed_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                    wait=False,
                )
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.debug("Payload index '%s': %s", field_name, e)

    def upsert_chunks(
        self,
        chunk_texts: list[str],
        chunk_metadata: list[dict[str, Any]],
        chunk_types: list[str],
        citations: list[str],
        references_list: list[list[dict[str, str]]],
        embeddings: list[list[float]],
        validations: list[dict[str, bool]] | None = None,
        batch_size: int = 64,
    ) -> int:
        """Upsert chunks with clean payload structure into Qdrant.

        Payload per point:
          {
            "text": <chunk text>,
            "chunk_type": <section|article|clause|preamble|...>,
            "citation": <auto-generated legal citation>,
            "metadata": {<canonical document + hierarchy metadata>},
            "validation": {<boundary and quality indicators>},
            "references": [<structured reference objects>]
          }

        Embeddings go into Qdrant's vector field, NOT duplicated in payload.
        """
        if not chunk_texts:
            logger.warning("upsert_chunks: no data to upsert")
            return 0

        assert len(chunk_texts) == len(chunk_metadata) == len(embeddings), (
            f"Length mismatch: texts={len(chunk_texts)}, "
            f"metadata={len(chunk_metadata)}, embeddings={len(embeddings)}"
        )

        total = len(chunk_texts)
        upserted = 0

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            points: list[PointStruct] = []

            for i in range(batch_start, batch_end):
                point_id = str(uuid.uuid4())

                payload: dict[str, Any] = {
                    "text": chunk_texts[i],
                    "chunk_type": chunk_types[i] if i < len(chunk_types) else "section",
                    "citation": citations[i] if i < len(citations) else "",
                    "metadata": self._sanitize_metadata(chunk_metadata[i]),
                    "references": references_list[i] if i < len(references_list) else [],
                    "validation": validations[i] if validations and i < len(validations) else {},
                }

                point = PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR_NAME: embeddings[i],
                    },
                    payload=payload,
                )
                points.append(point)

            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                upserted += len(points)
                logger.debug("Upserted batch: %d/%d points", upserted, total)
            except Exception as e:
                logger.error("Failed to upsert batch at offset %d: %s", batch_start, e)
                raise

        logger.info("Upsert complete: %d points in '%s'", upserted, self.collection_name)
        return upserted

    @staticmethod
    def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
        """Clean metadata dict for Qdrant — flatten/stringify any non-serializable values.

        Qdrant accepts nested JSON in payload, so hierarchy is preserved.
        Only non-serializable types (sets, custom objects) are converted.
        """
        clean: dict[str, Any] = {}
        for key, value in meta.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, dict):
                clean[key] = QdrantStore._sanitize_metadata(value)
            elif isinstance(value, list):
                clean_list = []
                for v in value:
                    if isinstance(v, (str, int, float, bool)):
                        clean_list.append(v)
                    elif isinstance(v, dict):
                        clean_list.append(QdrantStore._sanitize_metadata(v))
                    elif v is not None:
                        clean_list.append(str(v))
                clean[key] = clean_list
            else:
                clean[key] = str(value)
        return clean

    def collection_info(self) -> dict[str, Any]:
        """Return collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": getattr(info, "points_count", 0),
                "status": str(getattr(info, "status", "unknown")),
            }
        except Exception as e:
            logger.error("Failed to get collection info: %s", e)
            return {"error": str(e)}

    def delete_all_points(self) -> None:
        """Remove all points from the collection (for re-ingestion)."""
        from qdrant_client.models import Filter, FilterSelector

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=Filter()),
        )
        logger.info("Deleted all points from '%s'", self.collection_name)

    def delete_points_by_source(self, source_pdf: str) -> None:
        """Delete all chunks from a specific source PDF (useful for re-processing)."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source_pdf",
                            match=MatchValue(value=source_pdf),
                        )
                    ]
                )
            ),
        )
        logger.info("Deleted points for source '%s'", source_pdf)
