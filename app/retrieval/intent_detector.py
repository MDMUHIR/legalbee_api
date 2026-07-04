"""Legal intent detector: classifies query type and legal domain."""

from __future__ import annotations

import re
import logging

from app.config import config

logger = logging.getLogger(__name__)

QUERY_TYPE_PATTERNS = {
    "law_search": [
        r"(?:show|find|search|what is|where is|কী\s*(?:হলো|বলে)|দেখাও|খুঁজ)\s*(?:me\s+)?(?:Article|ধারা|Section|অনুচ্ছেদ)\s*\d+",
        r"^(?:article|section|ধারা|অনুচ্ছেদ)\s*\d+[A-Za-z]*",
    ],
    "act_summary": [
        r"summar[iy]ze|সারসংক্ষেপ|explain\s+(?:the\s+)?(?:Act|আইন)",
        r"what\s+(?:does|is)\s+the\s+.+\s+(?:Act|আইন)\s+(?:say|about)",
        r"give\s+me\s+(?:a\s+)?summary|overview\s+of",
    ],
    "amendment_question": [
        r"amendment|সংশোধন",
        r"what\s+changed|কী\s+পরিবর্তন|what\s+is\s+new",
        r"before\s+and\s+after|compare|তুলনা",
    ],
    "fact_analysis": [
        r"my\s+(?:landlord|employer|boss|tenant|neighbor|husband|wife)",
        r"আমার\s+(?:বাড়িওয়ালা|মনিব|প্রতিবেশী|স্বামী|স্ত্রী)",
        r"happened\s+to\s+me|what\s+can\s+I\s+do|what\s+are\s+my\s+rights",
        r"আমি\s+কী\s+করতে\s+পারি|আমার\s+কী\s+অধিকার",
        r"someone\s+(?:stole|took|damaged|did)|কে\s+(?:নিয়ে|করেছে)",
    ],
}

LEGAL_DOMAIN_PATTERNS = {
    "criminal": [r"punishment|শাস্তি|crime|অপরাধ|theft|চুরি|murder|হত্যা|offence"],
    "election": [r"election|নির্বাচন|vote|ভোট|candidate|প্রার্থী|nomination|মনোনয়ন"],
    "employment": [r"employment|চাকরি|service|employee|কর্মচারী|government\s+servant"],
    "constitution": [r"constitution|সংবিধান|fundamental\s+right|মৌলিক\s+অধিকার"],
    "administrative": [r"administrative|প্রশাসনিক|government|সরকারি|authority|কর্তৃপক্ষ"],
    "tax": [r"tax|কর|income|আয়|revenue|রাজস্ব|vat|ভ্যাট"],
    "civil": [r"property|সম্পত্তি|contract|চুক্তি|land|জমি|family|পারিবারিক"],
}


class IntentDetector:
    """Classify a legal query by type and domain using pattern matching."""

    def detect_query_type(self, question: str, retrieved_count: int = -1) -> str:
        text = question.lower()
        for qtype, patterns in QUERY_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return qtype
        if retrieved_count == 0:
            return "no_results"
        return "legal_question"

    def detect_legal_domain(self, question: str) -> str:
        text = question.lower()
        for domain, patterns in LEGAL_DOMAIN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return domain
        return "general"

    def detect_language(self, text: str) -> str:
        bengali_chars = sum(1 for c in text if "\u0980" <= c <= "\u09ff")
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars == 0:
            return "en"
        return "bn" if bengali_chars / alpha_chars > 0.3 else "en"

    def needs_rewrite(self, question: str) -> bool:
        words = question.strip().split()
        return len(words) < 5 and "article" not in question.lower() and "section" not in question.lower() and "ধারা" not in question
