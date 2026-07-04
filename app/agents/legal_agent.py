"""Legal Bee RAG Agent — main entry point for legal queries."""

from __future__ import annotations

import time
import logging
from typing import Optional

from app.agents.workflow import create_workflow, AgentState
from app.retrieval.intent_detector import IntentDetector
from app.retrieval.retriever import LegalRetriever
from app.retrieval.citation_builder import CitationBuilder
from app.retrieval.query_rewriter import QueryRewriter
from app.prompts.answer_prompt import get_answer_prompt
from app.services.llm_service import get_llm_service
from app.models.schemas import AgentResponse, RetrievedChunk

logger = logging.getLogger(__name__)


class LegalBeeAgent:
    """Production Legal RAG agent for Bangladeshi law.

    Usage:
        agent = LegalBeeAgent()
        response = agent.chat("What is the punishment for theft?")
        response = agent.search("Article 90E", filters={"act_name": "..."})
        response = agent.analyze("My landlord increased rent...")
        response = agent.summarize("Penal Code, 1860")
    """

    def __init__(self):
        self.intent_detector = IntentDetector()
        self.retriever = LegalRetriever()
        self.citation_builder = CitationBuilder()
        self.rewriter = QueryRewriter()
        self.workflow = create_workflow()

    def chat(
        self,
        question: str,
        language: str = "",
        user_type: str = "general",
    ) -> AgentResponse:
        t0 = time.time()

        if not language:
            language = self.intent_detector.detect_language(question)

        state: AgentState = {
            "question": question,
            "language": language,
            "user_type": user_type,
            "rewritten_question": "",
            "query_type": "",
            "legal_domain": "",
            "retrieved_chunks": [],
            "context": "",
            "answer": "",
            "citations": [],
            "references": [],
            "confidence": "medium",
            "needs_rewrite": False,
            "execution_time_ms": 0,
            "token_usage": {},
            "error": None,
        }

        try:
            result = self.workflow.invoke(state)
        except Exception as e:
            logger.exception("Workflow failed for: %s", question[:80])
            return AgentResponse(
                question=question,
                answer=f"An error occurred: {str(e)}",
                language_detected=language,
                error=str(e),
            )

        execution_ms = (time.time() - t0) * 1000

        chunks_raw = result.get("retrieved_chunks", [])
        # Convert dicts back to RetrievedChunk if needed
        retrieved_chunks: list[RetrievedChunk] = []
        for c in chunks_raw:
            if isinstance(c, RetrievedChunk):
                retrieved_chunks.append(c)
            elif isinstance(c, dict):
                retrieved_chunks.append(RetrievedChunk(**c))

        return AgentResponse(
            question=question,
            answer=result.get("answer", ""),
            answer_markdown=result.get("answer", ""),
            language_detected=language,
            query_type=result.get("query_type", "legal_question"),
            confidence=result.get("confidence", "medium"),
            citations=result.get("citations", []),
            retrieved_chunks=retrieved_chunks,
            references=result.get("references", []),
            execution_time_ms=execution_ms,
            token_usage=result.get("token_usage", {}),
            error=result.get("error"),
        )

    def search(
        self,
        query: str,
        language: str = "",
        filters: Optional[dict] = None,
        top_k: int = 8,
    ) -> AgentResponse:
        t0 = time.time()
        if not language:
            language = self.intent_detector.detect_language(query)

        chunks = self.retriever.retrieve(query, top_k=top_k, metadata_filter=filters)

        citations = self.citation_builder.build(chunks)
        references = self.citation_builder.build_references(chunks)
        context = self.citation_builder.build_context_string(chunks)

        confidence = "high" if len(chunks) >= 5 else "medium" if len(chunks) >= 2 else "low"

        return AgentResponse(
            question=query,
            answer=context,
            answer_markdown=context,
            language_detected=language,
            query_type="law_search",
            confidence=confidence,
            citations=citations,
            retrieved_chunks=chunks,
            references=references,
            execution_time_ms=(time.time() - t0) * 1000,
        )

    def analyze(
        self,
        facts: str,
        language: str = "",
    ) -> AgentResponse:
        t0 = time.time()
        if not language:
            language = self.intent_detector.detect_language(facts)

        rewritten = self.rewriter.rewrite(facts, language)
        chunks = self.retriever.retrieve(rewritten, top_k=12)
        citations = self.citation_builder.build(chunks)
        references = self.citation_builder.build_references(chunks)
        context = self.citation_builder.build_context_string(chunks)

        prompt = get_answer_prompt("analysis", language, facts=facts, context=context)

        if not chunks:
            return AgentResponse(
                question=facts,
                answer="I could not find relevant Bangladeshi law for this scenario." if language == "en"
                else "এই পরিস্থিতির জন্য কোনো প্রাসঙ্গিক বাংলাদেশী আইন পাওয়া যায়নি।",
                language_detected=language,
                query_type="fact_analysis",
                confidence="low",
                execution_time_ms=(time.time() - t0) * 1000,
            )

        try:
            llm = get_llm_service().llm
            response = llm.invoke(prompt, max_tokens=2048)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            answer = f"Analysis error: {str(e)}"

        return AgentResponse(
            question=facts,
            answer=answer,
            answer_markdown=answer,
            language_detected=language,
            query_type="fact_analysis",
            confidence="medium" if len(chunks) >= 3 else "low",
            citations=citations,
            retrieved_chunks=chunks,
            references=references,
            execution_time_ms=(time.time() - t0) * 1000,
        )

    def summarize(
        self,
        act_name: str,
        language: str = "",
    ) -> AgentResponse:
        t0 = time.time()
        if not language:
            language = self.intent_detector.detect_language(act_name)

        chunks = self.retriever.retrieve_by_act(act_name, top_k=24)
        citations = self.citation_builder.build(chunks)
        references = self.citation_builder.build_references(chunks)
        context = self.citation_builder.build_context_string(chunks, max_chunks=20)

        if not chunks:
            return AgentResponse(
                question=f"Summarize: {act_name}",
                answer=f"No information found for '{act_name}'." if language == "en"
                else f"'{act_name}' এর জন্য কোনো তথ্য পাওয়া যায়নি।",
                language_detected=language,
                query_type="act_summary",
                confidence="low",
                execution_time_ms=(time.time() - t0) * 1000,
            )

        prompt = get_answer_prompt("summary", language, act_name=act_name, context=context)

        try:
            llm = get_llm_service().llm
            response = llm.invoke(prompt, max_tokens=3072)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            answer = f"Summary error: {str(e)}"

        return AgentResponse(
            question=f"Summarize: {act_name}",
            answer=answer,
            answer_markdown=answer,
            language_detected=language,
            query_type="act_summary",
            confidence="high" if len(chunks) >= 10 else "medium",
            citations=citations,
            retrieved_chunks=chunks,
            references=references,
            execution_time_ms=(time.time() - t0) * 1000,
        )
