"""Text cleaning: removes repeated page headers, footers, noise, and normalizes Unicode."""

import re
import logging
from dataclasses import dataclass, field
from collections import Counter

from ingestion.utils import (
    normalize_unicode,
    fix_hyphenated_words,
    collapse_whitespace,
    is_noise_line,
    is_date_header,
    is_footer_url_line,
    is_copyright_line,
    extract_page_number_from_footer,
)

logger = logging.getLogger(__name__)


@dataclass
class CleanResult:
    """Result of cleaning a document's text."""

    cleaned_text: str = ""
    removed_headers: list[str] = field(default_factory=list)
    removed_footers: list[str] = field(default_factory=list)
    removed_noise_count: int = 0
    source_pages: int = 0


class TextCleaner:
    """Detect and remove repeated page artifacts from Bangladeshi law PDFs.

    The bdlaws PDFs have a consistent structure:
      - Top of every page: access date, act name, garbled diacritics
      - Bottom of every page: URL, page number (X/Y)
      - Copyright notice on the last page
      - Isolated Bangla vowel signs from font rendering

    The cleaner:
      1. Identifies repetitive header/footer lines by cross-page analysis
      2. Removes them from pages 2+ while preserving the first occurrence
      3. Strips noise lines (isolated diacritics, vowel signs)
      4. Normalizes whitespace and Unicode
      5. Rejoins hyphenated line-break words
    """

    HEADER_WINDOW = 8
    FOOTER_WINDOW = 5
    MIN_REPEAT_RATIO = 0.6

    def __init__(self, preserve_first_header: bool = True):
        self.preserve_first_header = preserve_first_header

    def clean(self, pages: list[str]) -> CleanResult:
        """Clean a list of page texts by removing headers, footers, and noise.

        Args:
            pages: List of text strings, one per PDF page.

        Returns:
            CleanResult with cleaned concatenated text and removal stats.
        """
        if not pages:
            return CleanResult()

        result = CleanResult(source_pages=len(pages))

        normalized_pages = [normalize_unicode(p) for p in pages]
        header_patterns = self._detect_repeating_headers(normalized_pages)
        footer_patterns = self._detect_repeating_footers(normalized_pages)

        cleaned_pages: list[str] = []
        for page_idx, page_text in enumerate(normalized_pages):
            lines = page_text.split("\n")
            cleaned_lines = []
            skip_count = 0

            # Track how far into header we are
            header_lines_seen = 0
            footer_lines_to_skip = set()

            # Identify footer lines to remove (from the bottom up)
            if footer_patterns:
                total = len(lines)
                for rev_idx, line in enumerate(reversed(lines)):
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    if self._match_any_pattern(line_stripped, footer_patterns):
                        footer_lines_to_skip.add(total - 1 - rev_idx)

            for line_idx, line in enumerate(lines):
                stripped = line.strip()

                stripped = re.sub(r"^\s*\d{2}/\d{2}/\d{4}\s+", "", stripped)

                if not stripped:
                    cleaned_lines.append("")
                    continue

                if line_idx in footer_lines_to_skip:
                    result.removed_footers.append(stripped)
                    continue

                if is_noise_line(stripped):
                    result.removed_noise_count += 1
                    continue

                if is_copyright_line(stripped):
                    continue

                if is_footer_url_line(stripped) or extract_page_number_from_footer(stripped):
                    result.removed_footers.append(stripped)
                    continue

                if header_patterns:
                    matched_header = False
                    for pat in header_patterns:
                        if pat in stripped or stripped in pat:
                            if page_idx >= 1 or not self.preserve_first_header:
                                if header_lines_seen < self.HEADER_WINDOW:
                                    header_lines_seen += 1
                                    matched_header = True
                                    skip_count += 1
                                    break
                    if matched_header:
                        continue

                if is_date_header(stripped) and page_idx >= 1:
                    continue

                cleaned_lines.append(stripped)

            cleaned_pages.append("\n".join(cleaned_lines))

        result.removed_headers = list(header_patterns)

        combined = "\n".join(cleaned_pages)
        combined = fix_hyphenated_words(combined)
        combined = collapse_whitespace(combined)
        combined = self._remove_empty_page_breaks(combined)
        result.cleaned_text = combined

        logger.info(
            "Cleaner: %d pages → %d chars cleaned | "
            "headers=%d footers=%d noise=%d",
            result.source_pages,
            len(result.cleaned_text),
            len(result.removed_headers),
            len(result.removed_footers),
            result.removed_noise_count,
        )

        return result

    def _detect_repeating_headers(self, pages: list[str]) -> set[str]:
        """Identify lines that appear across multiple pages near the top."""
        candidates: Counter = Counter()
        for page_text in pages:
            lines = [ln.strip() for ln in page_text.split("\n")[:self.HEADER_WINDOW]]
            seen = set()
            for ln in lines:
                if not ln or len(ln) < 3 or is_noise_line(ln):
                    continue
                if ln not in seen:
                    candidates[ln] += 1
                    seen.add(ln)

        threshold = max(2, int(len(pages) * self.MIN_REPEAT_RATIO))
        return {line for line, count in candidates.items() if count >= threshold}

    def _detect_repeating_footers(self, pages: list[str]) -> set[str]:
        """Identify lines that appear across multiple pages near the bottom."""
        candidates: Counter = Counter()
        for page_text in pages:
            lines = [ln.strip() for ln in page_text.split("\n")[-self.FOOTER_WINDOW:]]
            seen = set()
            for ln in lines:
                if not ln or len(ln) < 3:
                    continue
                if is_footer_url_line(ln):
                    candidates[ln] += 1
                    continue
                if extract_page_number_from_footer(ln):
                    candidates[ln] += 1
                    continue
                if ln not in seen:
                    candidates[ln] += 1
                    seen.add(ln)

        threshold = max(2, int(len(pages) * self.MIN_REPEAT_RATIO))
        return {line for line, count in candidates.items() if count >= threshold}

    @staticmethod
    def _match_any_pattern(line: str, patterns: set[str]) -> bool:
        for pat in patterns:
            if pat in line or line in pat:
                return True
        return False

    @staticmethod
    def _remove_empty_page_breaks(text: str) -> str:
        """Remove blank page-break markers left after header/footer removal."""
        text = re.sub(r"\n{4,}", "\n\n", text)
        return text.strip()
