"""Metadata extraction from Bangladeshi law PDF text."""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from ingestion.utils import bengali_to_int, is_mostly_bangla

logger = logging.getLogger(__name__)

BENGALI_DIGITS = "০১২৩৪৫৬৭৮৯"
ENGLISH_DIGITS = "0123456789"

RE_ACT_YEAR_NUMBER = re.compile(
    r"[(（]\s*([১২][০-৯]{3})\s*সনের?\s*([০-৯]+)\s*নং\s*(?:আইন|অধ্যাদেশ)\s*[)）]"
)

RE_ENGLISH_ACT = re.compile(
    r"Act\s+No\.?\s*(\d+)\s*of\s*(\d{4})",
    re.IGNORECASE,
)

RE_PUBLICATION_DATE = re.compile(
    r"\[\s*([০-৯]+)\s+([\u0980-\u09FF]+),\s*([১২][০-৯]{3})\s*\]"
)

RE_ENGLISH_DATE = re.compile(
    r"\[\s*(\d{1,2})(?:st|nd|rd|th)?\s+(\w+),?\s*(\d{4})\s*\]"
)

RE_AMENDMENT_OF = re.compile(
    r"(?:of|এর)\s+(?:the\s+)?(.+?)\s+(?:সংশোধনকল্পে|সংশোধন)"
    r"(?:কল্পে\s+প্রণীত\s+আইন)?$",
    re.DOTALL,
)

RE_AMENDMENT_ALT = re.compile(
    r"(.+?)\s+এর\s+অধিকতর\s+সংশোধনকল্পে\s+প্রণীত\s+আইন"
)

RE_CROSS_REFERENCE = re.compile(
    r"([১২][০-৯]{3})\s*সনের?\s*([০-৯]+)\s*নং\s*(আইন|অধ্যাদেশ)"
)

RE_ORDINANCE = re.compile(
    r"[(（]\s*([১২][০-৯]{3})\s*সনের?\s*([০-৯]+)\s*নং\s*অধ্যাদেশ\s*[)）]"
)

RE_VOLUME_PREFIX = re.compile(r"act-print-(\d+)")


@dataclass
class LawMetadata:
    """Structured metadata for a Bangladeshi law document."""

    act_name: str = ""
    bangla_name: str = ""
    act_number: str = ""
    act_year: int = 0
    publication_date: str = ""
    publication_date_en: str = ""
    language: str = "mixed"
    document_type: str = "Act"
    amendment_of: str = ""
    cross_references: list[str] = field(default_factory=list)
    source_pdf: str = ""
    volume: str = ""


class MetadataExtractor:
    """Extract structured metadata from the first page(s) of a Bangladeshi law PDF.

    Handles both pure-Bangla and English-mixed documents.
    Extracts: act name, number, year, date, document type, amendment info,
    cross-references, and language classification.
    """

    def __init__(self):
        pass

    def extract(self, text: str, filename: str, first_page_text: str = "") -> LawMetadata:
        """Extract all available metadata from document text.

        Args:
            text: Full cleaned text of the document.
            filename: Source PDF filename.
            first_page_text: Optional first-page-only text for better header extraction.

        Returns:
            LawMetadata dataclass with all extracted fields.
        """
        meta = LawMetadata()

        header_text = (first_page_text or text)[:2000]

        meta.source_pdf = filename
        meta.volume = self._extract_volume(filename)

        meta.language = self._detect_language(header_text)

        meta.act_year, meta.act_number = self._extract_act_year_number(header_text)
        meta.publication_date, meta.publication_date_en = self._extract_date(header_text)

        meta.bangla_name = self._extract_bangla_name(header_text)
        meta.act_name = self._extract_english_name(header_text, meta.bangla_name)

        meta.amendment_of = self._extract_amendment_of(header_text)

        meta.document_type = self._extract_document_type(header_text, meta.act_name)

        meta.cross_references = self._extract_cross_refs(text)

        logger.info(
            "Metadata: '%s' (%d/%s) | type=%s lang=%s",
            meta.act_name or meta.bangla_name[:60],
            meta.act_year,
            meta.act_number,
            meta.document_type,
            meta.language,
        )

        return meta

    def _detect_language(self, text: str) -> str:
        """Classify document language as 'bn', 'en', or 'mixed'."""
        if is_mostly_bangla(text, 0.8):
            return "bn"
        if is_mostly_bangla(text, 0.15):
            return "mixed"
        return "en"

    def _extract_act_year_number(self, text: str) -> tuple[int, str]:
        """Extract act year and number from patterns like:
        '( ২০২৬ সনের ০১ নং আইন )' or 'Act No. 3 of 2026'.
        """
        m = RE_ACT_YEAR_NUMBER.search(text)
        if m:
            return bengali_to_int(m.group(1)), str(bengali_to_int(m.group(2)))

        m2 = RE_ENGLISH_ACT.search(text)
        if m2:
            return int(m2.group(2)), m2.group(1)

        return 0, ""

    def _extract_date(self, text: str) -> tuple[str, str]:
        """Extract publication date in Bangla and English formats."""
        bangla_date = ""
        english_date = ""

        m = RE_PUBLICATION_DATE.search(text)
        if m:
            bangla_date = f"{m.group(1)} {m.group(2)}, {m.group(3)}"

        m2 = RE_ENGLISH_DATE.search(text)
        if m2:
            english_date = f"{m2.group(1)} {m2.group(2)}, {m2.group(3)}"

        return bangla_date, english_date

    def _extract_bangla_name(self, text: str) -> str:
        """Extract the Bangla act name from the header lines.

        The Bangla name typically appears as a title line containing
        'আইন' or 'অধ্যাদেশ' followed by the year.
        """
        lines = text.split("\n")
        candidates = []
        for line in lines[:30]:
            stripped = self._strip_date_prefix(line.strip())
            if not stripped or len(stripped) < 10:
                continue
            if any(word in stripped for word in ["আইন", "অধ্যাদেশ", "বিধিমালা"]):
                if is_mostly_bangla(stripped, 0.6):
                    candidates.append(stripped)

        for cand in candidates:
            if re.search(r"[১২][০-৯]{3}", cand):
                return re.sub(r"\s{2,}", " ", cand).strip()

        return candidates[0] if candidates else ""

    def _extract_english_name(self, text: str, bangla_name: str) -> str:
        """Extract the English act name from header lines.

        English titles appear in mixed-language documents alongside Bangla names,
        e.g., 'Representation of the People (Amendment) Act, 2026'.
        """
        lines = text.split("\n")
        for line in lines[:20]:
            stripped = self._strip_date_prefix(line.strip())
            if not stripped or len(stripped) < 10:
                continue
            if not is_mostly_bangla(stripped, 0.2):
                if re.search(r"Act[,，]\s*(19|20)\d{2}|Act\s*$", stripped):
                    return re.sub(r"\s{2,}", " ", stripped).strip()

        for line in lines[:20]:
            stripped = self._strip_date_prefix(line.strip())
            if len(stripped) > 20 and not is_mostly_bangla(stripped, 0.2):
                if "Act" in stripped or "Ordinance" in stripped:
                    if re.search(r"(19|20)\d{2}", stripped):
                        return re.sub(r"\s{2,}", " ", stripped).strip()

        return bangla_name

    @staticmethod
    def _strip_date_prefix(text: str) -> str:
        """Remove leading date pattern like '04/07/2026' from a line."""
        return re.sub(r"^\s*\d{2}/\d{2}/\d{4}\s+", "", text)

    def _extract_amendment_of(self, text: str) -> str:
        """Extract which law this document amends.

        Patterns:
          - '... এর অধিকতর সংশোধনকল্পে প্রণীত আইন'
          - '... সংশোধনকল্পে প্রণীত আইন'
          - 'সরকারি চাকরি আইন, ২০১৮ (২০১৮ সনের ৫৭ নং আইন) সংশোধনকল্পে প্রণীত আইন'
        """
        m = RE_AMENDMENT_ALT.search(text)
        if m:
            return re.sub(r"\s{2,}", " ", m.group(1)).strip()

        for line in text.split("\n")[:15]:
            stripped = line.strip()
            if "সংশোধন" in stripped and ("আইন" in stripped or "অধ্যাদেশ" in stripped):
                if not re.search(r"[(（].*?সনের.*?নং.*?আইন.*?[)）]", stripped):
                    continue
                before = re.split(r"\s*সংশোধন", stripped)[0]
                if before:
                    return re.sub(r"\s{2,}", " ", before).strip()

        return ""

    def _extract_document_type(self, text: str, english_name: str) -> str:
        """Determine document type: Act, Ordinance, Rule, Amendment.

        Amendment acts are detected by the presence of 'সংশোধন' in the
        description or amendment reference.
        """
        header = text[:1000]

        if "Amendment" in english_name or "সংশোধন" in header:
            if "আইন" in header:
                return "Amendment Act"
            if "অধ্যাদেশ" in header:
                return "Amendment Ordinance"

        if "অধ্যাদেশ" in header:
            return "Ordinance"
        if "বিধিমালা" in header:
            return "Rule"
        if "Ordinance" in text[:500]:
            return "Ordinance"
        if "Act" in text[:500]:
            return "Act"

        return "Act"

    def _extract_cross_refs(self, text: str) -> list[str]:
        """Extract references to other laws mentioned in this document."""
        refs: list[str] = []
        seen: set[str] = set()

        for m in RE_CROSS_REFERENCE.finditer(text):
            year = str(bengali_to_int(m.group(1)))
            number = str(bengali_to_int(m.group(2)))
            law_type = m.group(3)
            ref = f"{year} সনের {number} নং {law_type}"
            if ref not in seen:
                refs.append(ref)
                seen.add(ref)

        return refs

    @staticmethod
    def _extract_volume(filename: str) -> str:
        """Extract volume identifier from filename pattern 'act-print-XXXX.pdf'."""
        m = RE_VOLUME_PREFIX.search(filename)
        if m:
            return m.group(1)
        return ""
