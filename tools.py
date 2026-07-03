"""
tools.py — Search tools for the Legal Bee RAG system.

Improvements over original:
  - Uses new 'legal-bee-db' collection with Hybrid Search
  - Dense + Sparse vector names match the Qdrant Cloud UI configuration
  - Metadata formatted properly: law_name, law_year, section_number, chapter
  - Added filtered search by law name or year
  - Proper LangChain @tool decorators (no SimpleNamespace hack)
  - Single shared embedding model & client (no repeated loading)
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText

load_dotenv()

# ---------------------------------------------------------------------------
# Shared singletons — loaded once, reused across all tool calls
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_embed():
    # FastEmbedEmbeddings with 384 dimensions to match Qdrant database
    # BAAI/bge-small-en-v1.5 is optimized for multilingual support (Bengali+English)
    return FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
    )


@lru_cache(maxsize=1)
def _get_store():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise ValueError(
            "Missing Qdrant credentials. "
            "Set QDRANT_URL and QDRANT_API_KEY in your .env file."
        )

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    return QdrantVectorStore(
        client=client,
        collection_name="legal-bee-db",          # ← new collection
        embedding=_get_embed(),
        vector_name="law_dense_vector",           # ← must match Qdrant UI
        sparse_vector_name="law_sparse_vector",   # ← must match Qdrant UI
        retrieval_mode=RetrievalMode.HYBRID,      # ← Dense + Sparse together
    )


# ---------------------------------------------------------------------------
# Metadata formatter
# ---------------------------------------------------------------------------

def _format_chunk(doc, score: float | None = None) -> str:
    m = doc.metadata
    law = m.get("law_name", "Unknown Law")
    year = m.get("law_year", "")
    section = m.get("section_number", "")
    chapter = m.get("chapter", "")

    header_parts = [law]
    if year:
        header_parts.append(f"({year})")
    if section:
        header_parts.append(f"ধারা {section}")
    if chapter:
        header_parts.append(chapter)
    if score is not None:
        header_parts.append(f"relevance={score:.2f}")

    header = " | ".join(header_parts)
    return f"[{header}]\n{doc.page_content.strip()}"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_bd_law(query: str) -> str:
    """Search Bangladeshi laws for relevant legal provisions using hybrid
    semantic + keyword search. Use this for any legal question or topic.
    Input: a legal question, topic, or case description in Bengali or English."""
    try:
        store = _get_store()
        results = store.similarity_search_with_score(query, k=6)

        if not results:
            return "প্রদত্ত ডাটাবেসে কোনো প্রাসঙ্গিক আইন পাওয়া যায়নি।"

        chunks = [_format_chunk(doc, score) for doc, score in results]
        return "\n\n---\n\n".join(chunks)

    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def find_sections_by_law(law_name: str) -> str:
    """Retrieve sections from a specific Bangladeshi Act by its name.
    Use this when the user mentions a specific law by name.
    Input: Act name in Bengali or English, e.g. 'মানব পাচার আইন' or 'Animal Slaughter Act'."""
    try:
        store = _get_store()
        client = store.client

        # Use Qdrant payload filter to get only chunks from this law
        results = client.query_points(
            collection_name="legal-bee-db",
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="law_name",
                        match=MatchText(text=law_name),
                    )
                ]
            ),
            limit=8,
            with_payload=True,
        )

        if not results.points:
            # Fallback: semantic search with law name
            docs = store.similarity_search(law_name, k=8)
            if not docs:
                return f"'{law_name}' সংক্রান্ত কোনো আইন ডাটাবেসে পাওয়া যায়নি।"
            chunks = [_format_chunk(doc) for doc in docs]
        else:
            from langchain_core.documents import Document
            chunks = []
            for point in results.points:
                p = point.payload or {}
                doc = Document(
                    page_content=p.get("page_content", ""),
                    metadata=p,
                )
                chunks.append(_format_chunk(doc))

        return "\n\n---\n\n".join(chunks)

    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def search_by_section(section_number: str, law_name: str = "") -> str:
    """Find the exact text of a specific section number from a Bangladeshi law.
    Use this when the user asks about a specific section like 'ধারা ১৮' or 'section 5'.
    Input: section_number as string (e.g. '18'), optional law_name to narrow down."""
    try:
        store = _get_store()
        client = store.client

        must_conditions = [
            FieldCondition(
                key="section_number",
                match=MatchValue(value=str(section_number)),
            )
        ]
        if law_name:
            must_conditions.append(
                FieldCondition(
                    key="law_name",
                    match=MatchText(text=law_name),
                )
            )

        results = client.query_points(
            collection_name="legal-bee-db",
            query_filter=Filter(must=must_conditions),
            limit=5,
            with_payload=True,
        )

        if not results.points:
            return f"ধারা {section_number} ডাটাবেসে পাওয়া যায়নি।"

        from langchain_core.documents import Document
        chunks = []
        for point in results.points:
            p = point.payload or {}
            doc = Document(page_content=p.get("page_content", ""), metadata=p)
            chunks.append(_format_chunk(doc))

        return "\n\n---\n\n".join(chunks)

    except Exception as e:
        return f"Search error: {str(e)}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_tools() -> list:
    """Return all available legal search tools."""
    return [search_bd_law, find_sections_by_law, search_by_section]