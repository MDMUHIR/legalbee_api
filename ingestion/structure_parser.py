"""Legal structure parser for Bangladeshi law documents.

Detects the hierarchical structure:
  Act → Chapter → Section → Sub-section → Clause → Sub-clause

Handles both Bangla and mixed Bangla/English numbering schemes.
"""

import re
import logging
from dataclasses import dataclass, field

from ingestion.utils import BENGALI_DIGITS

logger = logging.getLogger(__name__)

RE_BENGALI_SECTION = re.compile(
    r"(?:^|\s)([{0}]{{1,2}})\s*[.।]".format(BENGALI_DIGITS)
)

RE_ARTICLE = re.compile(
    r"(?:^|\n|\s)Article\s+(\d+[A-Za-z]?)\s*[.।:)]?",
    re.IGNORECASE,
)

RE_CHAPTER = re.compile(
    r"(?:^|\n)\s*("
    r"প্রথম|দ্বিতীয়|তৃতীয়|চতুর্থ|পঞ্চম|ষষ্ঠ|সপ্তম|অষ্টম|নবম|দশম|"
    r"একাদশ|দ্বাদশ|ত্রয়োদশ|চতুর্দশ|পঞ্চদশ|ষোড়শ|সপ্তদশ|অষ্টাদশ|"
    r"ঊনবিংশ|বিংশ|একবিংশ"
    r")\s*অধ্যায়\s*[\n:]?",
)

RE_CLAUSE = re.compile(
    r"\s*[(（]\s*"
    r"([^{)）\s]+)"
    r"\s*[)）]\s*"
)

VALID_CLAUSE_SPECIAL = frozenset({"xial", "xiaa", "bis", "ter"})


def _is_valid_clause_identifier(identifier: str) -> bool:
    """Return True only if the identifier looks like a genuine legal clause marker.

    Legal clause markers in Bangladeshi law:
      - (১), (12), (1)   — digits, 1-3 chars
      - (ক), (খ)          — single Bangla consonant
      - (a), (i), (aa), (ii) — 1-3 lowercase ASCII letters
      - (xial), (xiaa)    — special known identifiers

    Explicitly EXCLUDED (common false positives in parenthetical prose):
      - English words: (review), (Amendment), (Act), (except)
      - Acronyms: (GEMS), (PMIS), (EVM)
      - Any identifier longer than 4 chars
      - Any identifier containing uppercase after the first char
    """
    if not identifier or len(identifier) > 4:
        return False

    if identifier.lower() in VALID_CLAUSE_SPECIAL:
        return True

    if all(c in (BENGALI_DIGITS + "0123456789") for c in identifier):
        return len(identifier) <= 3

    if len(identifier) == 1 and "\u0995" <= identifier <= "\u09B9":
        return True

    if (
        identifier.isascii()
        and identifier.isalpha()
        and identifier.islower()
        and len(identifier) <= 3
    ):
        return True

    return False

RE_CHAPTER_EN = re.compile(
    r"(?:^|\n)\s*CHAPTER\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE
)

RE_INSERTED_SECTION = re.compile(
    r"(?:ধারা|Section|Article)\s*"
    r"([{0}]+[ক-হ]?|\d+[A-Za-z]?)".format(BENGALI_DIGITS)
)

RE_VAGUE_SECTION = re.compile(
    r"(?:^|\n)\s*(\d+[A-Za-z]?)\s*[.．。]",
)

SECTION_TITLE_SEPARATORS = re.compile(r"[.。:：\-–—\n]")


@dataclass
class LegalClause:
    """Represents a sub-section, clause, or sub-clause within a section."""

    identifier: str = ""
    text: str = ""
    level: int = 1
    children: list["LegalClause"] = field(default_factory=list)
    start_char: int = 0
    end_char: int = 0


@dataclass
class LegalSection:
    """Represents a single legal section (ধারা) with its clauses.

    For amendment acts, `inserted_section_number` captures the section
    number being inserted into the target act (e.g., section 2 of act-1630
    inserts section 37A into the Government Service Act).
    """

    section_number: str = ""
    section_title: str = ""
    inserted_section_number: str = ""
    chapter: str = ""
    article: str = ""
    content: str = ""
    clauses: list[LegalClause] = field(default_factory=list)
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0


@dataclass
class ParsedDocument:
    """Full parsed structure of a law document."""

    preamble: str = ""
    chapters: list[str] = field(default_factory=list)
    sections: list[LegalSection] = field(default_factory=list)
    clauses: list[LegalClause] = field(default_factory=list)


class StructureParser:
    """Parse Bangladeshi law PDF text into hierarchical legal structure.

    The parser detects:
      - Chapters (অধ্যায় / CHAPTER)
      - Sections (ধারা) numbered with Bengali numerals (১।, ২।)
      - Articles (for English/Constitution-style documents)
      - Sub-sections / Clauses / Sub-clauses with nested numbering
    """

    def __init__(self):
        pass

    def parse(self, text: str) -> ParsedDocument:
        """Parse cleaned text into a ParsedDocument with hierarchical structure.

        Args:
            text: Cleaned, header-free document text.

        Returns:
            ParsedDocument with detected chapters, sections, and clauses.
        """
        doc = ParsedDocument()

        text = text.strip()
        if not text:
            logger.warning("StructureParser: empty input text")
            return doc

        section_spans = self._find_section_boundaries(text)

        current_chapter = ""
        current_article = ""
        preamble_end = 0

        for i, (sec_num, sec_title, start_pos, _) in enumerate(section_spans):
            if i == 0 and start_pos > 0:
                doc.preamble = text[:start_pos].strip()
                preamble_end = start_pos

            end_pos = section_spans[i + 1][2] if i + 1 < len(section_spans) else len(text)
            section_text = text[start_pos:end_pos]

            chapter, sec_title_clean = self._extract_chapter_from_title(sec_title)
            if chapter:
                current_chapter = chapter
                if chapter not in doc.chapters:
                    doc.chapters.append(chapter)

            article_num = self._extract_article(sec_title_clean or "")
            if not article_num:
                article_num = self._extract_article(section_text[:300])

            inserted_num = self._extract_inserted_section(section_text)

            section = LegalSection(
                section_number=sec_num,
                section_title=sec_title_clean,
                inserted_section_number=inserted_num,
                chapter=current_chapter,
                article=article_num,
                content=section_text.strip(),
                start_char=start_pos,
                end_char=end_pos,
            )

            section.clauses = self._parse_clauses(section_text, sec_num)
            doc.sections.append(section)

            if not current_article and article_num:
                current_article = article_num

        if doc.sections and doc.sections[0].start_char > 0:
            potential_preamble = text[:doc.sections[0].start_char].strip()
            if potential_preamble and len(potential_preamble) > 20:
                doc.preamble = potential_preamble

        logger.info(
            "StructureParser: %d sections, %d chapters found",
            len(doc.sections),
            len(doc.chapters),
        )
        return doc

    def _find_section_boundaries(
        self, text: str
    ) -> list[tuple[str, str, int, int]]:
        """Find all section markers and return (number, title, start_pos, end_pos).

        Section markers are identified by:
          - Bengali numerals followed by `।` (at line start or preceded by space)
          - 'Article N' pattern
          - English numerals followed by `.` (fallback)
        """
        boundaries: list[tuple[str, str, int, int]] = []

        for m in RE_BENGALI_SECTION.finditer(text):
            sec_num = m.group(1).strip()

            title = self._extract_section_title(text, m.start())

            boundaries.append((sec_num, title, m.start(), m.end()))

        if not boundaries:
            for m in RE_ARTICLE.finditer(text):
                sec_num = m.group(1)
                title = self._extract_section_title(text, m.start())
                boundaries.append((sec_num, title, m.start(), m.end()))

        if not boundaries:
            for m in RE_CHAPTER_EN.finditer(text):
                boundaries.append((m.group(1), "", m.start(), m.end()))

        if not boundaries:
            for m in RE_VAGUE_SECTION.finditer(text):
                num_str = m.group(1)
                if num_str.isdigit() and int(num_str) <= 200:
                    boundaries.append((num_str, "", m.start(), m.end()))

        boundaries.sort(key=lambda x: x[2])
        return boundaries

    def _extract_section_title(self, text: str, marker_pos: int) -> str:
        """Extract the section title preceding the section marker.

        On bdlaws PDFs the title is often on the same line as the marker,
        separated by whitespace (e.g., 'স�?ক্ষিপ্ত শিরোনাম    ১।').

        Only returns text that looks like a genuine title — short, meaningful,
        and directly preceding the section number. Returns empty string
        if nothing plausible is found.
        """
        line_start = text.rfind("\n", 0, marker_pos)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1

        same_line_prefix = text[line_start:marker_pos].strip()

        skip_keywords = [
            "আইন", "সংশোধন", "সনের", "নং", "অধ্যাদেশ",
            "P.O.", "President", "Order", "No.", "of",
        ]

        contains_skip = any(kw in same_line_prefix for kw in skip_keywords)

        title_parts: list[str] = []

        if contains_skip:
            return ""

        if same_line_prefix and len(same_line_prefix) > 3 and len(same_line_prefix) < 200:
            title_parts.append(same_line_prefix)

        if line_start > 1:
            prev_line_end = max(0, line_start - 2)
            prev_line_start = text.rfind("\n", 0, prev_line_end)
            if prev_line_start == -1:
                prev_line_start = 0
            else:
                prev_line_start += 1
            prev_line = text[prev_line_start:prev_line_end].strip()

            if prev_line and 3 < len(prev_line) < 200:
                has_section_num = re.match(
                    r"^\s*[{0}]+\s*[.।]".format(BENGALI_DIGITS), prev_line
                )
                has_article = re.match(r"^\s*Article\s+\d+", prev_line, re.IGNORECASE)
                has_skip_kw = any(kw in prev_line for kw in skip_keywords)

                if not has_section_num and not has_article and not has_skip_kw:
                    title_parts.insert(0, prev_line)

        title = " ".join(title_parts).strip()

        if len(title) < 3 or len(title) > 200:
            return ""

        return title

    def _extract_chapter_from_title(self, title: str) -> tuple[str, str]:
        """If the title contains a chapter name, extract it and return
        (chapter_name, remaining_title)."""
        m = RE_CHAPTER.search(title)
        if m:
            chapter = m.group(1).strip() + " অধ্যায়"
            remaining = RE_CHAPTER.sub("", title).strip()
            return chapter, remaining
        return "", title

    def _extract_article(self, title_or_text: str) -> str:
        """Extract article number from the first line of the section description.

        Amendment acts reference the article they modify in their first sentence,
        e.g., 'উক্ত Order এর Article 8 এর clause (1) এ...'.
        We capture the article number for hierarchy metadata.
        """
        first_200 = title_or_text[:200] if title_or_text else ""

        m = RE_ARTICLE.search(first_200)
        if m:
            return m.group(1)
        return ""

    def _extract_inserted_section(self, section_text: str) -> str:
        """Extract the section number being inserted or amended into the target act.

        In amendment acts, each section inserts/replaces a specific provision
        in the target law. For example:
          - 'ধারা ৩৭ এর পর নিম্নরূপ নূতন ধারা ৩৭ক সন্নিবেশিত'
          - 'এর ধারা ৩৭ এর পর নিম্নরূপ নূতন ধারা'
        We detect the section being inserted (e.g., 37A).
        """
        patterns = [
            r"নূতন\s+ধারা\s+([{0}]+[ক-হ]?)".format(BENGALI_DIGITS),
            r"নতুন\s+ধারা\s+([{0}]+[ক-হ]?)".format(BENGALI_DIGITS),
            r"ধারা\s+([{0}]+[ক-হ]?)\s+সন্নিবেশ".format(BENGALI_DIGITS),
            r"(?:নূতন|নতুন)\s*Article\s+(\d+[A-Za-z]*)",
        ]

        for pattern in patterns:
            m = re.search(pattern, section_text[:500])
            if m:
                return m.group(1)
        return ""

    def _parse_clauses(self, section_text: str, section_num: str) -> list[LegalClause]:
        """Parse sub-sections, clauses, and sub-clauses within a section.

        Builds a tree of LegalClause objects where nesting reflects the legal hierarchy:
          (১) → sub-section (level 1)
            (ক) → clause (level 2)
              (অ) or (aa) → sub-clause (level 3)

        Only genuine legal clause identifiers are accepted.
        English words in parentheses like (review), (GEMS) are skipped.
        """
        raw_matches = list(RE_CLAUSE.finditer(section_text))
        valid_matches = [(m, m.group(1).strip()) for m in raw_matches]
        valid_matches = [
            (m, ident) for m, ident in valid_matches
            if _is_valid_clause_identifier(ident)
        ]

        if len(valid_matches) < 2:
            return []

        clauses: list[LegalClause] = []
        clause_stack: list[LegalClause] = []

        for i, (m, identifier) in enumerate(valid_matches):
            clause_start = m.end()

            clause_end = (
                valid_matches[i + 1][0].start()
                if i + 1 < len(valid_matches)
                else len(section_text)
            )

            clause_text = section_text[clause_start:clause_end].strip()

            level = self._determine_clause_level(identifier, clause_stack)

            clause = LegalClause(
                identifier=identifier,
                text=clause_text,
                level=level,
                start_char=m.start(),
                end_char=clause_end,
            )

            while clause_stack and clause_stack[-1].level >= level:
                clause_stack.pop()

            if clause_stack:
                clause_stack[-1].children.append(clause)
            else:
                clauses.append(clause)

            clause_stack.append(clause)

        return clauses

    def _determine_clause_level(
        self, identifier: str, clause_stack: list[LegalClause]
    ) -> int:
        """Determine nesting level of a clause identifier, using stack context.

        Level 1: (১), (1)  — sub-section (numeric)
        Level 2: (ক), (a)  — clause (single char, Bangla consonant or ASCII letter)
        Level 3: (অ), (aa), (i), (ii) — sub-clause (inside a clause)
        """
        is_digits = all(c in (BENGALI_DIGITS + "0123456789") for c in identifier)
        if is_digits:
            return 1

        is_bangla_consonant = (
            len(identifier) == 1 and "\u0995" <= identifier <= "\u09B9"
        )

        is_inside_clause = clause_stack and any(c.level >= 2 for c in clause_stack)

        if is_inside_clause:
            if is_bangla_consonant:
                return 3
            return 3

        if is_bangla_consonant:
            return 2

        if len(identifier) == 1 and identifier.isascii() and identifier.isalpha():
            return 2 if identifier.islower() else 3

        return 3
