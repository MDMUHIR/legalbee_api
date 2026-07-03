"""Shared utilities: digit conversion, Unicode helpers, token estimation."""

import re
import unicodedata
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

BENGALI_DIGITS = "০১২৩৪৫৬৭৮৯"

BENGALI_RANGE_START = 0x0980
BENGALI_RANGE_END = 0x09FF

BENGALI_VOWEL_SIGNS: set[int] = {
    0x09BE,  # া
    0x09BF,  # ি
    0x09C0,  # ী
    0x09C1,  # ু
    0x09C2,  # ূ
    0x09C3,  # ৃ
    0x09C4,  # ৄ
    0x09C7,  # ে
    0x09C8,  # ৈ
    0x09CB,  # ো
    0x09CC,  # ৌ
}


def bengali_to_int(s: str) -> int:
    """Convert Bengali digit string like '২০২৬' to integer 2026."""
    result = ""
    for ch in str(s):
        idx = BENGALI_DIGITS.find(ch)
        result += str(idx) if idx >= 0 else ch
    try:
        return int(result)
    except ValueError:
        return 0


def int_to_bengali(n: int) -> str:
    """Convert integer to Bengali digit string."""
    return "".join(BENGALI_DIGITS[int(d)] for d in str(n))


def is_bangla_char(ch: str) -> bool:
    """Check if character is within the Bengali Unicode block."""
    return BENGALI_RANGE_START <= ord(ch) <= BENGALI_RANGE_END


def is_bangla_vowel_sign(ch: str) -> bool:
    """Check if character is a Bengali vowel sign (dependent vowel)."""
    return ord(ch) in BENGALI_VOWEL_SIGNS


def is_mostly_bangla(text: str, threshold: float = 0.3) -> bool:
    """Return True if proportion of Bangla characters exceeds threshold."""
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return False
    bangla = sum(1 for c in text if is_bangla_char(c))
    return bangla / alpha > threshold


def is_noise_line(line: str) -> bool:
    """Detect if a line is mostly noise (isolated diacritics / vowel signs)."""
    stripped = line.strip()
    if not stripped:
        return False

    chars = [c for c in stripped if not c.isspace()]
    if not chars:
        return True

    vowel_signs = sum(1 for c in chars if is_bangla_vowel_sign(c))
    total = len(chars)

    if total <= 4 and vowel_signs >= total * 0.5:
        return True

    bangla = sum(1 for c in chars if is_bangla_char(c))
    non_bangla_alpha = sum(1 for c in chars if c.isalpha() and not is_bangla_char(c))

    if total <= 6 and bangla > 0 and non_bangla_alpha == 0 and vowel_signs >= 2:
        return True

    return False


def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC form and fix common encoding issues."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", text)
    return text


def fix_hyphenated_words(text: str) -> str:
    """Rejoin hyphenated words that were split across line breaks in OCR/PDF output."""
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single spaces (preserve paragraph breaks)."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n +", "\n", text)
    text = text.strip()
    return text


@lru_cache(maxsize=1)
def _get_tokenizer():
    """Lazy-load the BGE-M3 tokenizer for token counting."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("BAAI/bge-m3", trust_remote_code=True)
    except Exception as e:
        logger.warning("Could not load BGE-M3 tokenizer: %s. Using character estimate.", e)
        return None


def estimate_tokens(text: str) -> int:
    """Estimate token count for BGE-M3 using the actual tokenizer if available."""
    tk = _get_tokenizer()
    if tk is not None:
        try:
            return len(tk.encode(text, add_special_tokens=False))
        except Exception:
            pass

    bangla_chars = sum(1 for c in text if is_bangla_char(c))
    other_chars = len(text) - bangla_chars
    return int(bangla_chars / 2.5 + other_chars / 4.0)


def extract_page_number_from_footer(line: str) -> Optional[int]:
    """Extract current page number from footer like '1/4' or '2/21'."""
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", line.strip())
    if m:
        return int(m.group(1))
    return None


def is_footer_url_line(line: str) -> bool:
    """Check if a line is the bdlaws URL footer."""
    return bool(re.search(r"bdlaws\.minlaw\.gov\.bd", line))


def is_date_header(line: str) -> bool:
    """Check if a line is just a date like '04/07/2026'."""
    return bool(re.match(r"^\s*\d{2}/\d{2}/\d{4}\s*$", line.strip()))


def is_copyright_line(line: str) -> bool:
    """Check if a line is a copyright notice."""
    stripped = line.strip()
    return (
        "Copyright" in stripped
        or "Legislative and Parliamentary Affairs Division" in stripped
        or "Ministry of Law, Justice and Parliamentary Affairs" in stripped
    )
