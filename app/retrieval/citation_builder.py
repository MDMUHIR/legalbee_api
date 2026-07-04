"""Citation builder: extracts structured citations from retrieved chunk metadata."""

from __future__ import annotations

from app.models.schemas import RetrievedChunk


class CitationBuilder:
    """Build legal citations from retrieved chunk metadata.

    Uses the pre-built citation field from ingestion whenever possible.
    Falls back to constructing from hierarchy metadata.
    """

    def build(self, chunks: list[RetrievedChunk]) -> list[str]:
        seen: set[str] = set()
        citations: list[str] = []

        for chunk in chunks:
            if chunk.citation and chunk.citation not in seen:
                citations.append(chunk.citation)
                seen.add(chunk.citation)
                continue

            citation = self._build_from_hierarchy(chunk)
            if citation and citation not in seen:
                citations.append(citation)
                seen.add(citation)

        return citations

    def _build_from_hierarchy(self, chunk: RetrievedChunk) -> str:
        parts = [chunk.act_name] if chunk.act_name else []
        h = chunk.hierarchy
        if h.get("article"):
            parts.append(f"Article {h['article']}")
        if h.get("section"):
            parts.append(f"Section {h['section']}")
        if h.get("clause"):
            parts.append(f"Clause ({h['clause']})")
        if h.get("sub_clause"):
            parts.append(f"Sub-clause ({h['sub_clause']})")
        return ", ".join(parts) if parts else ""

    def build_context_string(self, chunks: list[RetrievedChunk], max_chunks: int = 8) -> str:
        """Build a formatted context string for the LLM prompt."""
        parts: list[str] = []
        for i, chunk in enumerate(chunks[:max_chunks]):
            header = f"[Chunk {i + 1}]"
            if chunk.citation:
                header += f" | {chunk.citation}"
            if chunk.score > 0:
                header += f" | relevance={chunk.score:.2f}"
            parts.append(f"{header}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    def build_references(self, chunks: list[RetrievedChunk]) -> list[dict]:
        seen: set[str] = set()
        refs: list[dict] = []
        for chunk in chunks:
            for ref in chunk.references:
                key = f"{ref.get('type', '')}:{ref.get('target', '')}"
                if key not in seen:
                    refs.append(ref)
                    seen.add(key)
        return refs
