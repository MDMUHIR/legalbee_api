"""Legal-aware chunking: respect section boundaries, split long sections at clause level."""

import re
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ingestion.utils import estimate_tokens
from ingestion.structure_parser import ParsedDocument, LegalSection, LegalClause
from ingestion.metadata import LawMetadata

logger = logging.getLogger(__name__)

BENGALI_SENTENCE_END = "।"

RE_SENTENCE = re.compile(r"([^।।\n]+[।।])")


@dataclass
class Chunk:
    """A legal text chunk with metadata ready for embedding and storage."""

    chunk_id: str
    text: str
    token_count: int
    chunk_type: str = "section"
    citation: str = ""
    hierarchy: dict[str, str] = field(default_factory=dict)
    references: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class LegalChunker:
    """Convert parsed legal structure into semantic chunks ready for embedding.

    Chunking rules:
      1. One section = one chunk (if it fits)
      2. Long sections → split at clause/sub-clause boundaries
      3. Oversized single clauses → split at sentence boundaries (`।` or `. `)
      4. NEVER combine text from different sections into one chunk
      5. OVERLAP only when splitting within a section (100 token equivalent)

    Token targets:
      - Target: 600-900 tokens
      - Maximum: 1000 tokens
      - Overlap: ~100 tokens (character equivalent)
    """

    TARGET_TOKENS_MIN = 600
    TARGET_TOKENS_MAX = 900
    MAX_TOKENS = 1000
    OVERLAP_TOKENS = 100

    OVERLAP_CHARS = 280

    def __init__(
        self,
        target_min: int = TARGET_TOKENS_MIN,
        target_max: int = TARGET_TOKENS_MAX,
        max_tokens: int = MAX_TOKENS,
        overlap_tokens: int = OVERLAP_TOKENS,
    ):
        self.target_min = target_min
        self.target_max = target_max
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._char_per_token = 3.0

    def chunk(
        self, parsed_doc: ParsedDocument, metadata: LawMetadata, page_count: int = 1
    ) -> list[Chunk]:
        """Create semantic chunks from a parsed legal document.

        Args:
            parsed_doc: Parsed document with sections and clauses.
            metadata: Extracted document-level metadata.
            page_count: Total pages in source PDF.

        Returns:
            List of Chunk objects with text and per-chunk metadata.
        """
        chunks: list[Chunk] = []

        if parsed_doc.preamble and len(parsed_doc.preamble.strip()) > 50:
            pre_chunks = self._chunk_preamble(parsed_doc.preamble, metadata)
            chunks.extend(pre_chunks)

        for section in parsed_doc.sections:
            section_chunks = self._chunk_section(
                section, metadata, page_count, chunk_index_offset=len(chunks)
            )
            chunks.extend(section_chunks)

        logger.info("Chunker: %d chunks from %d sections", len(chunks), len(parsed_doc.sections))
        return chunks

    def _chunk_preamble(self, text: str, metadata: LawMetadata) -> list[Chunk]:
        """Chunk the preamble (text before first section)."""
        token_count = estimate_tokens(text)
        if token_count <= self.target_max:
            chunk = self._make_chunk(
                text=text,
                metadata=metadata,
                section=0,
                clause="",
                chunk_idx=0,
                total_in_section=1,
            )
            return [chunk]

        parts = self._split_at_sentences(text)
        return self._build_chunks_from_parts(parts, metadata, section_num="0")

    def _chunk_section(
        self,
        section: LegalSection,
        metadata: LawMetadata,
        page_count: int,
        chunk_index_offset: int = 0,
    ) -> list[Chunk]:
        """Chunk a single legal section, respecting clause boundaries.

        Strategy:
          1. If section has clauses, try to group clauses into chunks.
          2. If section is short enough, keep as one chunk.
          3. Otherwise, split at sentence boundaries.
        """
        if section.clauses:
            return self._chunk_section_by_clauses(
                section, metadata, page_count, chunk_index_offset
            )

        return self._chunk_section_text(
            section.content,
            metadata,
            section.section_number,
            section.chapter,
            section.article,
            section.section_title,
            chunk_index_offset,
        )

    def _chunk_section_by_clauses(
        self,
        section: LegalSection,
        metadata: LawMetadata,
        page_count: int,
        chunk_index_offset: int,
    ) -> list[Chunk]:
        """Group clauses into chunks that fit the target token range.

        The section intro text (before first clause) is prepended to the first chunk.
        Clauses are never split mid-content unless they individually exceed max_tokens.
        """
        chunks: list[Chunk] = []
        intro_end = 0

        if section.clauses:
            intro_end = section.clauses[0].start_char
            intro_text = section.content[:intro_end].strip()
        else:
            intro_text = ""

        intro_tokens = estimate_tokens(intro_text) if intro_text else 0

        current_group: list[LegalClause] = []
        current_tokens = intro_tokens

        def flush_group() -> None:
            nonlocal current_tokens, intro_tokens
            if not current_group:
                return
            group_text_parts = []
            if intro_text and len(chunks) == 0:
                group_text_parts.append(intro_text)

            clause_refs: list[str] = []
            sub_clause_refs: list[str] = []
            for cl in current_group:
                group_text_parts.append(f"({cl.identifier}) {cl.text}")
                clause_refs.append(cl.identifier)
                for sub in cl.children:
                    sub_clause_refs.append(sub.identifier)

            combined = "\n".join(group_text_parts)

            chunk_idx = chunk_index_offset + len(chunks)
            total = (len(section.clauses) // max(1, len(current_group))) + 2

            chunks.append(
                self._make_chunk(
                    text=combined,
                    metadata=metadata,
                    section=section.section_number,
                    chapter=section.chapter,
                    article=section.article,
                    section_title=section.section_title,
                    clause=", ".join(clause_refs),
                    sub_clause=", ".join(sub_clause_refs),
                    chunk_idx=chunk_idx,
                    total_in_section=total,
                )
            )
            current_group.clear()
            current_tokens = 0

        for clause in section.clauses:
            clause_text = f"({clause.identifier}) {clause.text}"
            clause_tokens = estimate_tokens(clause_text)

            if clause_tokens > self.max_tokens:
                flush_group()
                clause_chunks = self._chunk_section_text(
                    clause_text,
                    metadata,
                    section.section_number,
                    section.chapter,
                    section.article,
                    section.section_title,
                    chunk_index_offset + len(chunks),
                    clause_ref=clause.identifier,
                )
                chunks.extend(clause_chunks)
                continue

            if current_tokens + clause_tokens > self.max_tokens and current_group:
                flush_group()

            current_group.append(clause)
            current_tokens += clause_tokens

        flush_group()

        if not chunks:
            chunks = self._chunk_section_text(
                section.content,
                metadata,
                section.section_number,
                section.chapter,
                section.article,
                section.section_title,
                chunk_index_offset,
            )

        return chunks

    def _chunk_section_text(
        self,
        text: str,
        metadata: LawMetadata,
        section_num: str,
        chapter: str = "",
        article: str = "",
        section_title: str = "",
        chunk_index_offset: int = 0,
        clause_ref: str = "",
    ) -> list[Chunk]:
        """Chunk plain section text (no clause structure detected).

        Uses sentence boundaries for splitting, ensuring chunks stay within
        token limits.
        """
        token_count = estimate_tokens(text)

        if token_count <= self.target_max:
            chunk = self._make_chunk(
                text=text,
                metadata=metadata,
                section=section_num,
                chapter=chapter,
                article=article,
                section_title=section_title,
                clause=clause_ref,
                chunk_idx=chunk_index_offset,
                total_in_section=1,
            )
            return [chunk]

        sentences = self._split_at_sentences(text)
        return self._build_chunks_from_parts(
            sentences,
            metadata,
            section_num=section_num,
            chapter=chapter,
            article=article,
            section_title=section_title,
            clause_ref=clause_ref,
            offset=chunk_index_offset,
        )

    def _split_at_sentences(self, text: str) -> list[str]:
        """Split text at sentence boundaries (Bangla `।` and English `. `)."""
        parts = RE_SENTENCE.findall(text)
        if parts:
            remaining = RE_SENTENCE.sub("", text).strip()
            if remaining:
                parts.append(remaining)
            return [p.strip() for p in parts if p.strip()]

        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _build_chunks_from_parts(
        self,
        parts: list[str],
        metadata: LawMetadata,
        section_num: str = "",
        chapter: str = "",
        article: str = "",
        section_title: str = "",
        clause_ref: str = "",
        offset: int = 0,
    ) -> list[Chunk]:
        """Build chunks by accumulating sentence parts until target token range."""
        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_tokens = 0

        total_est = sum(estimate_tokens(p) for p in parts)
        est_chunks = max(1, total_est // self.target_max)
        target_per_chunk = min(self.max_tokens, self.target_max)

        def make_chunk(buf: list[str], idx: int) -> Chunk:
            return self._make_chunk(
                text=" ".join(buf),
                metadata=metadata,
                section=section_num,
                chapter=chapter,
                article=article,
                section_title=section_title,
                clause=clause_ref,
                chunk_idx=offset + idx,
                total_in_section=est_chunks,
            )

        for i, part in enumerate(parts):
            pt = estimate_tokens(part)

            if pt > self.max_tokens:
                if buffer:
                    chunks.append(make_chunk(buffer, len(chunks)))
                    buffer.clear()
                    buffer_tokens = 0

                sub_parts = self._force_split_long_part(part)
                for sp in sub_parts:
                    chunks.append(
                        self._make_chunk(
                            text=sp,
                            metadata=metadata,
                            section=section_num,
                            chapter=chapter,
                            article=article,
                            section_title=section_title,
                            clause=clause_ref,
                            chunk_idx=offset + len(chunks),
                            total_in_section=est_chunks + len(sub_parts),
                        )
                    )
                continue

            if buffer_tokens + pt > target_per_chunk and buffer:
                chunks.append(make_chunk(buffer, len(chunks)))
                overlap_part = self._get_overlap(buffer[-1])
                buffer = [overlap_part] if overlap_part else []
                buffer_tokens = estimate_tokens(overlap_part) if overlap_part else 0

            buffer.append(part)
            buffer_tokens += pt

        if buffer:
            chunks.append(make_chunk(buffer, len(chunks)))

        return chunks

    def _force_split_long_part(self, text: str) -> list[str]:
        """Brute-force split a very long sentence at word boundaries."""
        words = text.split()
        if len(words) < 10:
            return [text]

        result: list[str] = []
        buf: list[str] = []
        buf_tokens = 0

        for w in words:
            w_tokens = estimate_tokens(w)
            if buf_tokens + w_tokens > self.max_tokens and buf:
                result.append(" ".join(buf))
                buf = [w]
                buf_tokens = w_tokens
            else:
                buf.append(w)
                buf_tokens += w_tokens

        if buf:
            result.append(" ".join(buf))
        return result or [text]

    def _get_overlap(self, last_sentence: str) -> str:
        """Get the overlap fragment from the last sentence for context continuity."""
        words = last_sentence.split()
        if len(words) <= 5:
            return last_sentence

        overlap_words = max(3, int(self.OVERLAP_CHARS / (self._char_per_token)))
        overlap_words = min(overlap_words, len(words) - 1)
        return " ".join(words[-overlap_words:])

    def _make_chunk(
        self,
        text: str,
        metadata: LawMetadata,
        section: str = "",
        chapter: str = "",
        article: str = "",
        section_title: str = "",
        clause: str = "",
        sub_clause: str = "",
        chunk_idx: int = 0,
        total_in_section: int = 1,
    ) -> Chunk:
        """Create a Chunk with canonical metadata, hierarchy, citation and references.

        The payload is structured for optimal RAG retrieval:
          - No duplicate fields (act_name, not law_name)
          - hierarchy groups the legal position
          - citation is auto-generated for the LLM
          - references are structured objects
        """
        chunk_id = f"{metadata.source_pdf}:s{section}:c{chunk_idx}"

        hierarchy = {
            "chapter": chapter or "",
            "article": article or "",
            "section": str(section) if section else "",
            "clause": str(clause) if clause else "",
            "sub_clause": str(sub_clause) if sub_clause else "",
        }

        chunk_type = self._determine_chunk_type(
            section=section,
            chapter=chapter,
            article=article,
            clause=clause,
            sub_clause=sub_clause,
            text=text,
        )

        citation = self._build_citation(metadata, hierarchy)

        refs = self._build_references(metadata)

        act_name = metadata.act_name or metadata.bangla_name

        doc_meta = {
            "act_name": act_name,
            "bangla_name": metadata.bangla_name,
            "act_number": metadata.act_number,
            "year": metadata.act_year,
            "publication_date": metadata.publication_date,
            "document_type": metadata.document_type,
            "source_pdf": metadata.source_pdf,
            "volume": metadata.volume,
            "pdf_page": 1,
            "language": metadata.language,
            "section_title": section_title or "",
            "hierarchy": hierarchy,
            "chunk_index": chunk_idx,
        }

        return Chunk(
            chunk_id=chunk_id,
            text=text.strip(),
            token_count=estimate_tokens(text),
            chunk_type=chunk_type,
            citation=citation,
            hierarchy=hierarchy,
            references=refs,
            metadata=doc_meta,
        )

    @staticmethod
    def _determine_chunk_type(
        section: str,
        chapter: str,
        article: str,
        clause: str,
        sub_clause: str,
        text: str,
    ) -> str:
        """Classify the chunk into one of the legal provision types.

        Priority: sub_clause > clause > section > article > chapter > preamble
        Special types are detected from text content (schedule, definition, amendment).
        """
        if not section and not article and not chapter:
            stripped = text.lower()
            if any(kw in stripped[:200] for kw in ["schedule", "তফসিল", "পরিশিষ্ট"]):
                return "schedule"
            if "appendix" in stripped[:200]:
                return "appendix"
            if any(kw in stripped[:200] for kw in ["সংক্ষিপ্ত শিরোনাম", "preamble", "প্রস্তাবনা", "যেহেতু"]):
                return "preamble"
            return "preamble"

        if sub_clause:
            return "clause"
        if clause:
            return "clause"
        if article:
            return "article"
        if section:
            return "section"
        if chapter:
            return "chapter"
        return "act"

    @staticmethod
    def _build_citation(metadata: LawMetadata, hierarchy: dict[str, str]) -> str:
        """Build a human-readable legal citation string.

        Examples:
          'সরকারি চাকরি (সংশোধন) আইন, ২০২৬, ধারা ২'
          'Representation of the People (Amendment) Act, 2026, Section 35, Clause (2)'
        """
        name = metadata.act_name or metadata.bangla_name
        parts = [name] if name else []

        if hierarchy.get("chapter"):
            parts.append(f"Chapter {hierarchy['chapter']}")
        if hierarchy.get("article"):
            parts.append(f"Article {hierarchy['article']}")
        if hierarchy.get("section"):
            parts.append(f"Section {hierarchy['section']}")
        if hierarchy.get("clause"):
            parts.append(f"Clause ({hierarchy['clause']})")
        if hierarchy.get("sub_clause"):
            parts.append(f"Sub-clause ({hierarchy['sub_clause']})")

        return ", ".join(parts) if parts else ""

    @staticmethod
    def _build_references(metadata: LawMetadata) -> list[dict[str, str]]:
        """Build structured references from metadata.

        The amendment_of field becomes a reference of type 'amends'.
        Cross-references to other laws become 'references' entries.
        """
        refs: list[dict[str, str]] = []

        if metadata.amendment_of:
            refs.append({"type": "amends", "target": metadata.amendment_of})

        for cross_ref in metadata.cross_references:
            if cross_ref.strip():
                refs.append({"type": "references", "target": cross_ref.strip()})

        return refs
