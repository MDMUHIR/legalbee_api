"""Query rewriter: expands vague questions using LLM for better retrieval."""

from __future__ import annotations

import logging

from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)

REWRITE_PROMPT_EN = """Rewrite the following legal question to be more specific and searchable for a Bangladeshi law database.

Include:
- The relevant area of law (criminal, civil, constitutional, election, employment, tax)
- Specific legal terms
- If the user mentions a section number, article number, or act name, preserve it

Original: {question}
Rewritten:"""

REWRITE_PROMPT_BN = """নিম্নলিখিত আইনি প্রশ্নটি বাংলাদেশী আইন ডাটাবেসের জন্য আরও নির্দিষ্ট এবং অনুসন্ধানযোগ্য করে পুনর্লিখন করুন।

অন্তর্ভুক্ত করুন:
- সংশ্লিষ্ট আইনের ক্ষেত্র (ফৌজদারি, দেওয়ানি, সাংবিধানিক, নির্বাচন, কর্মসংস্থান, কর)
- নির্দিষ্ট আইনি পরিভাষা
- ব্যবহারকারী ধারা নম্বর, অনুচ্ছেদ নম্বর বা আইনের নাম উল্লেখ করলে তা সংরক্ষণ করুন

মূল: {question}
পুনর্লিখিত:"""


class QueryRewriter:
    """Rewrite vague queries into specific, retrievable legal questions."""

    def __init__(self):
        self.llm_service = get_llm_service()

    def rewrite(self, question: str, language: str = "en") -> str:
        prompt_template = REWRITE_PROMPT_BN if language == "bn" else REWRITE_PROMPT_EN
        prompt = prompt_template.format(question=question)

        try:
            llm = self.llm_service.llm
            result = llm.invoke(prompt, max_tokens=256)
            rewritten = result.content.strip() if hasattr(result, "content") else str(result).strip()
            if rewritten and len(rewritten) > len(question) * 0.3:
                logger.info("Query rewritten: '%s' -> '%s'", question[:60], rewritten[:120])
                return rewritten
        except Exception as e:
            logger.warning("Query rewrite failed: %s", e)

        return question
