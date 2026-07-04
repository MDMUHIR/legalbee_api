"""Legal-aware chunking: respect section boundaries, split long sections at clause level.

Every chunk is self-contained — section context is prepended to ALL chunks
from the same section, not just the first one. This ensures each chunk is
independently retrievable and understandable without its neighbours.

Citations reference the primary legal provision only — never concatenate
unrelated clause numbers. Non-legal identifiers like (review), (GEMS), (PMIS)
are excluded by the structure parser's clause validator.
"""

import re
import uuid
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional

from ingestion.utils import estimate_tokens, BENGALI_DIGITS
from ingestion.structure_parser import ParsedDocument, LegalSection, LegalClause
from ingestion.metadata import LawMetadata

logger = logging.getLogger(__name__)

RE_SENTENCE = re.compile(r"([^।।\n]+[।।])")


class ChunkType(str, Enum):
    ACT = "act"
    CHAPTER = "chapter"
    PART = "part"
    ARTICLE = "article"
    SECTION = "section"
    CLAUSE = "clause"
    SUB_CLAUSE = "sub_clause"
    SCHEDULE = "schedule"
    APPENDIX = "appendix"
    EXPLANATION = "explanation"
    DEFINITION = "definition"
    PREAMBLE = "preamble"
    AMENDMENT = "amendment"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    token_count: int
    chunk_type: ChunkType = ChunkType.SECTION
    citation: str = ""
    hierarchy: dict[str, str] = field(default_factory=dict)
    references: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    validation: dict[str, bool] = field(default_factory=lambda: {
        "validated": True,
        "starts_at_boundary": True,
        "ends_at_boundary": True,
        "is_complete_chunk": True,
    })
    validation_errors: list[str] = field(default_factory=list)


class LegalChunker:
    """Convert parsed legal structure into self-contained semantic chunks.

    Rules:
      1. Every chunk includes section context (never starts mid-provision).
      2. Long sections split at clause boundaries only.
      3. Single oversized clauses split at sentence boundaries.
      4. NEVER combine different sections into one chunk.
      5. OVERLAP only when forced to split within a section.

    Token targets:
      - Target: 600-900 tokens
      - Maximum: 1000 tokens
    """

    TARGET_TOKENS_MIN = 600
    TARGET_TOKENS_MAX = 900
    MAX_TOKENS = 1000

    def __init__(
        self,
        target_min: int = TARGET_TOKENS_MIN,
        target_max: int = TARGET_TOKENS_MAX,
        max_tokens: int = MAX_TOKENS,
    ):
        self.target_min = target_min
        self.target_max = target_max
        self.max_tokens = max_tokens

    def chunk(
        self, parsed_doc: ParsedDocument, metadata: LawMetadata, page_count: int = 1
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        if parsed_doc.preamble and len(parsed_doc.preamble.strip()) > 50:
            pre_chunks = self._chunk_preamble(parsed_doc.preamble, metadata)
            chunks.extend(pre_chunks)

        for section in parsed_doc.sections:
            section_chunks = self._chunk_section(
                section, metadata, chunk_index_offset=len(chunks)
            )
            chunks.extend(section_chunks)

        validated = [c for c in chunks if self._validate(c)]
        dropped = len(chunks) - len(validated)
        if dropped:
            logger.warning("Chunker: %d/%d chunks dropped by validation", dropped, len(chunks))

        logger.info("Chunker: %d chunks from %d sections", len(validated), len(parsed_doc.sections))
        return validated

    # ── preamble ──────────────────────────────────────────────────────

    def _chunk_preamble(self, text: str, metadata: LawMetadata) -> list[Chunk]:
        token_count = estimate_tokens(text)
        if token_count <= self.max_tokens:
            chunk = self._make_chunk(
                text=text,
                metadata=metadata,
                section="",
                chunk_type=ChunkType.PREAMBLE,
                chunk_idx=0,
            )
            return [chunk]

        parts = self._split_at_sentences(text)
        result: list[Chunk] = []
        for i, p in enumerate(parts):
            result.append(self._make_chunk(
                text=p,
                metadata=metadata,
                section="",
                chunk_type=ChunkType.PREAMBLE,
                chunk_idx=i,
            ))
        return result

    # ── section chunking ──────────────────────────────────────────────

    def _chunk_section(
        self,
        section: LegalSection,
        metadata: LawMetadata,
        chunk_index_offset: int = 0,
    ) -> list[Chunk]:
        if section.clauses:
            return self._chunk_by_clauses(
                section, metadata, chunk_index_offset
            )

        return self._chunk_plain_section(
            section, metadata, chunk_index_offset
        )

    def _chunk_by_clauses(
        self,
        section: LegalSection,
        metadata: LawMetadata,
        chunk_index_offset: int,
    ) -> list[Chunk]:
        """Group clauses into chunks. Every chunk gets the section header prepended."""
        chunks: list[Chunk] = []

        if not section.clauses:
            return chunks

        intro_end = section.clauses[0].start_char
        intro_text = section.content[:intro_end].strip()

        current_group: list[LegalClause] = []
        current_tokens = estimate_tokens(intro_text)
        # Section header is added to every chunk below
        header_tokens = estimate_tokens(intro_text)

        def flush_group() -> None:
            if not current_group:
                return
            group_parts: list[str] = []
            clause_ids: list[str] = []
            sub_clause_ids: list[str] = []

            for cl in current_group:
                group_parts.append(f"({cl.identifier}) {cl.text}")
                clause_ids.append(cl.identifier)
                for sub in cl.children:
                    sub_clause_ids.append(sub.identifier)

            body = "\n".join(group_parts)

            primary_clause = clause_ids[0] if clause_ids else ""
            primary_sub = sub_clause_ids[0] if sub_clause_ids else ""

            chunk_type = ChunkType.CLAUSE
            if not primary_clause:
                chunk_type = ChunkType.SECTION
            if primary_sub:
                chunk_type = ChunkType.SUB_CLAUSE

            chunks.append(
                self._make_chunk(
                    text=body,
                    metadata=metadata,
                    section=section.section_number,
                    chapter=section.chapter,
                    article=section.article,
                    section_title=section.section_title,
                    clause=primary_clause,
                    sub_clause=primary_sub,
                    inserted_section=section.inserted_section_number,
                    chunk_type=chunk_type,
                    chunk_idx=chunk_index_offset + len(chunks),
                )
            )
            current_group.clear()

        for clause in section.clauses:
            clause_text = f"({clause.identifier}) {clause.text}"
            clause_tokens = estimate_tokens(clause_text)

            if clause_tokens > self.max_tokens:
                flush_group()
                sub_chunks = self._chunk_plain_text(
                    clause_text,
                    metadata,
                    section.section_number,
                    section.chapter,
                    section.article,
                    section.section_title,
                    chunk_index_offset + len(chunks),
                    clause_ref=clause.identifier,
                )
                chunks.extend(sub_chunks)
                continue

            if current_tokens + clause_tokens > self.max_tokens and current_group:
                flush_group()
                current_tokens = header_tokens

            current_group.append(clause)
            current_tokens += clause_tokens

        flush_group()

        if not chunks:
            return self._chunk_plain_section(section, metadata, chunk_index_offset)

        for c in chunks:
            c.text = f"{intro_text}\n\n{c.text}".strip()
            c.token_count = estimate_tokens(c.text)

        return chunks

    def _chunk_plain_section(
        self,
        section: LegalSection,
        metadata: LawMetadata,
        chunk_index_offset: int,
    ) -> list[Chunk]:
        return self._chunk_plain_text(
            section.content,
            metadata,
            section.section_number,
            section.chapter,
            section.article,
            section.section_title,
            chunk_index_offset,
            chunk_type=ChunkType.SECTION,
        )

    def _chunk_plain_text(
        self,
        text: str,
        metadata: LawMetadata,
        section_num: str,
        chapter: str = "",
        article: str = "",
        section_title: str = "",
        chunk_index_offset: int = 0,
        clause_ref: str = "",
        chunk_type: ChunkType = ChunkType.SECTION,
    ) -> list[Chunk]:
        token_count = estimate_tokens(text)
        if token_count <= self.max_tokens:
            chunk = self._make_chunk(
                text=text,
                metadata=metadata,
                section=section_num,
                chapter=chapter,
                article=article,
                section_title=section_title,
                clause=clause_ref,
                chunk_type=chunk_type,
                chunk_idx=chunk_index_offset,
            )
            return [chunk]

        sentences = self._split_at_sentences(text)
        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_tokens = 0

        for part in sentences:
            pt = estimate_tokens(part)

            if pt > self.max_tokens:
                if buffer:
                    chunks.append(self._make_chunk(
                        text=" ".join(buffer),
                        metadata=metadata,
                        section=section_num, chapter=chapter, article=article,
                        section_title=section_title, clause=clause_ref,
                        chunk_type=chunk_type,
                        chunk_idx=chunk_index_offset + len(chunks),
                    ))
                    buffer.clear()
                    buffer_tokens = 0

                words = part.split()
                sub_buf: list[str] = []
                sub_tokens = 0
                for w in words:
                    wt = estimate_tokens(w)
                    if sub_tokens + wt > self.max_tokens and sub_buf:
                        chunks.append(self._make_chunk(
                            text=" ".join(sub_buf),
                            metadata=metadata,
                            section=section_num, chapter=chapter, article=article,
                            section_title=section_title, clause=clause_ref,
                            chunk_type=chunk_type,
                            chunk_idx=chunk_index_offset + len(chunks),
                        ))
                        sub_buf = [w]
                        sub_tokens = wt
                    else:
                        sub_buf.append(w)
                        sub_tokens += wt
                if sub_buf:
                    chunks.append(self._make_chunk(
                        text=" ".join(sub_buf),
                        metadata=metadata,
                        section=section_num, chapter=chapter, article=article,
                        section_title=section_title, clause=clause_ref,
                        chunk_type=chunk_type,
                        chunk_idx=chunk_index_offset + len(chunks),
                    ))
                continue

            if buffer_tokens + pt > self.max_tokens and buffer:
                chunks.append(self._make_chunk(
                    text=" ".join(buffer),
                    metadata=metadata,
                    section=section_num, chapter=chapter, article=article,
                    section_title=section_title, clause=clause_ref,
                    chunk_type=chunk_type,
                    chunk_idx=chunk_index_offset + len(chunks),
                ))
                buffer.clear()
                buffer_tokens = 0

            buffer.append(part)
            buffer_tokens += pt

        if buffer:
            chunks.append(self._make_chunk(
                text=" ".join(buffer),
                metadata=metadata,
                section=section_num, chapter=chapter, article=article,
                section_title=section_title, clause=clause_ref,
                chunk_type=chunk_type,
                chunk_idx=chunk_index_offset + len(chunks),
            ))

        return chunks

    # ── sentence splitting ────────────────────────────────────────────

    def _split_at_sentences(self, text: str) -> list[str]:
        parts = RE_SENTENCE.findall(text)
        if parts:
            remaining = RE_SENTENCE.sub("", text).strip()
            if remaining:
                parts.append(remaining)
            return [p.strip() for p in parts if p.strip()]

        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    # ── chunk factory ─────────────────────────────────────────────────

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
        inserted_section: str = "",
        chunk_type: ChunkType = ChunkType.SECTION,
        chunk_idx: int = 0,
    ) -> Chunk:
        chunk_id = f"{metadata.source_pdf}:s{section}:c{chunk_idx}"

        hierarchy = {
            "part": "",
            "chapter": chapter or "",
            "article": article or "",
            "section": str(section) if section else "",
            "clause": str(clause) if clause else "",
            "sub_clause": str(sub_clause) if sub_clause else "",
        }

        citation = self._build_citation(metadata, hierarchy, section_title, inserted_section)

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

    # ── classification ────────────────────────────────────────────────

    @staticmethod
    def _classify_chunk_type(
        section: str,
        chapter: str,
        article: str,
        clause: str,
        sub_clause: str,
        text: str,
    ) -> ChunkType:
        if not section and not article and not chapter:
            t = text.lower()
            if any(kw in t[:200] for kw in ["schedule", "তফসিল", "পরিশিষ্ট"]):
                return ChunkType.SCHEDULE
            if "appendix" in t[:200]:
                return ChunkType.APPENDIX
            if any(kw in t[:200] for kw in ["explanation", "ব্যাখ্যা"]):
                return ChunkType.EXPLANATION
            if any(kw in t[:200] for kw in ["definition", "সংজ্ঞা"]):
                return ChunkType.DEFINITION
            return ChunkType.PREAMBLE

        if sub_clause:
            return ChunkType.SUB_CLAUSE
        if clause:
            return ChunkType.CLAUSE
        if article:
            return ChunkType.ARTICLE
        if section:
            return ChunkType.SECTION
        if chapter:
            return ChunkType.CHAPTER
        return ChunkType.ACT

    # ── citation ──────────────────────────────────────────────────────

    @staticmethod
    def _build_citation(
        metadata: LawMetadata,
        hierarchy: dict[str, str],
        section_title: str = "",
        inserted_section: str = "",
    ) -> str:
        """Build a clean legal citation.

        For amendment acts, includes the inserted/amended provision:
          'সরকারি চাকরি (সংশোধন) আইন, ২০২৬, Section ২ (নতুন ধারা ৩৭ক)'
        For regular acts:
          'রাজস্ব আইন, ২০২৩, Section 35, Clause (2)'
        """
        name = metadata.act_name or metadata.bangla_name
        if not name:
            return ""

        parts: list[str] = [name]

        h = hierarchy
        if h.get("part"):
            parts.append(f"Part {h['part']}")
        if h.get("chapter"):
            parts.append(f"Chapter {h['chapter']}")
        if h.get("article"):
            parts.append(f"Article {h['article']}")
        if h.get("section"):
            sec_part = f"Section {h['section']}"
            if inserted_section:
                sec_part += f" (inserted Section {inserted_section})"
            elif section_title:
                clean_title = section_title.strip()
                if 3 < len(clean_title) < 100 and not any(
                    kw in clean_title for kw in ["P.O.", "সনের", "নং", "No.", "of"]
                ):
                    sec_part += f" ({clean_title})"
            parts.append(sec_part)
        if h.get("clause"):
            parts.append(f"Clause ({h['clause']})")
        if h.get("sub_clause"):
            parts.append(f"Sub-clause ({h['sub_clause']})")

        return ", ".join(parts)

    # ── references ────────────────────────────────────────────────────

    @staticmethod
    def _build_references(metadata: LawMetadata) -> list[dict[str, str]]:
        refs: list[dict[str, str]] = []

        if metadata.amendment_of:
            refs.append({"type": "amends", "target": metadata.amendment_of})

        for ref in metadata.references:
            if ref.target.strip():
                refs.append({"type": ref.ref_type, "target": ref.target})

        return refs

    # ── validation ────────────────────────────────────────────────────

    def _validate(self, chunk: Chunk) -> bool:
        """Validate chunk quality and populate validation metadata.

        Checks:
          - Text is non-empty and reasonable length
          - Does not start with a closing parenthesis
          - Required metadata fields present
          - Hierarchy consistent with chunk_type
          - Citation present for non-preamble chunks

        Sets chunk.validation with boundary/quality indicators.
        """
        errors: list[str] = []
        text = chunk.text.strip()
        v = chunk.validation

        if not text:
            errors.append("empty text")
            v["is_complete_chunk"] = False
        if len(text) < 20:
            errors.append(f"text too short ({len(text)} chars)")
            v["is_complete_chunk"] = False

        first_word = text[:80].lstrip()
        if first_word.startswith((")", "）")):
            errors.append(f"starts inside provision: '{text[:50]}'")
            v["starts_at_boundary"] = False

        last_line = text.split("\n")[-1].strip() if "\n" in text else text[-100:]
        if any(
            kw in last_line.lower()
            for kw in ["p.o. no.", "সংক্ষিপ্ত", "রহিতকরণ", "সনের", "নং আইন", "অধ্যায়"]
        ):
            pass
        elif not re.search(r"[।.!?]$|।[\"'""]$", last_line) and chunk.chunk_type not in (
            ChunkType.PREAMBLE, ChunkType.SCHEDULE
        ):
            paragraph_endings = r"(?:citation|amendment|provision|proviso|thereunder|accordingly|thereof)[\s.]*$"
            if not re.search(paragraph_endings, last_line.lower()):
                v["ends_at_boundary"] = False

        if not chunk.metadata.get("act_name") and not chunk.metadata.get("bangla_name"):
            errors.append("missing act name")
        if not chunk.metadata.get("year") or chunk.metadata.get("year") == 0:
            errors.append("missing or zero year")
        if chunk.chunk_type == ChunkType.SECTION and not (
            chunk.hierarchy.get("section") or chunk.metadata.get("hierarchy", {}).get("section")
        ):
            errors.append("section type but no section in hierarchy")
        if chunk.chunk_type == ChunkType.CLAUSE and not (
            chunk.hierarchy.get("clause") or chunk.metadata.get("hierarchy", {}).get("clause")
        ):
            errors.append("clause type but no clause in hierarchy")
        if chunk.citation == "" and chunk.chunk_type != ChunkType.PREAMBLE:
            errors.append("missing citation")
            v["is_complete_chunk"] = False

        if errors:
            chunk.valid = False
            chunk.validation_errors = errors
            v["validated"] = False
            v["starts_at_boundary"] = False
            v["ends_at_boundary"] = False
            v["is_complete_chunk"] = False
            logger.warning("Rejected chunk %s: %s", chunk.chunk_id, errors)
            return False

        chunk.valid = True
        v["validated"] = True
        return True
