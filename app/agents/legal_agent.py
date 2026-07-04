"""Legal Bee RAG Agent — multi-agent entry point for legal queries.

Routes queries through a multi-agent LangGraph workflow with specialized agents:
  - Law Search Agent     → specific provisions
  - Legal Analysis Agent → fact scenarios
  - Act Summary Agent    → summarize acts
  - Amendment Compare    → what changed
  - Legal QA Agent       → general questions (default)
"""

from __future__ import annotations

import time
import logging
from typing import Optional

from app.agents.workflow import create_workflow, AgentState
from app.retrieval.intent_detector import IntentDetector
from app.retrieval.retriever import LegalRetriever
from app.retrieval.citation_builder import CitationBuilder
from app.models.schemas import AgentResponse, RetrievedChunk

logger = logging.getLogger(__name__)


class LegalBeeAgent:
    """Multi-agent Legal RAG system for Bangladeshi law.

    All methods route through the LangGraph workflow which
    auto-selects the appropriate specialized agent.

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
        self.workflow = create_workflow()

    # ── Main chat: through multi-agent workflow ────────────────────────

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
            "routed_agent": "",
            "agent_output": "",
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

    # ── Search: direct retrieval (no LLM) ──────────────────────────────

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

    # ── Analyze: via workflow with intent override ─────────────────────

    def analyze(
        self,
        facts: str,
        language: str = "",
    ) -> AgentResponse:
        t0 = time.time()
        if not language:
            language = self.intent_detector.detect_language(facts)

        state: AgentState = {
            "question": facts,
            "language": language,
            "user_type": "general",
            "rewritten_question": facts,
            "query_type": "fact_analysis",
            "legal_domain": self.intent_detector.detect_legal_domain(facts),
            "routed_agent": "",
            "agent_output": "",
            "retrieved_chunks": [],
            "context": "",
            "answer": "",
            "citations": [],
            "references": [],
            "confidence": "medium",
            "needs_rewrite": True,
            "execution_time_ms": 0,
            "token_usage": {},
            "error": None,
        }

        try:
            result = self.workflow.invoke(state)
        except Exception as e:
            logger.exception("Analyze workflow failed")
            return AgentResponse(
                question=facts,
                answer=f"Analysis error: {str(e)}",
                language_detected=language,
                error=str(e),
            )

        chunks_raw = result.get("retrieved_chunks", [])
        retrieved_chunks: list[RetrievedChunk] = []
        for c in chunks_raw:
            if isinstance(c, RetrievedChunk):
                retrieved_chunks.append(c)
            elif isinstance(c, dict):
                retrieved_chunks.append(RetrievedChunk(**c))

        return AgentResponse(
            question=facts,
            answer=result.get("answer", ""),
            answer_markdown=result.get("answer", ""),
            language_detected=language,
            query_type="fact_analysis",
            confidence=result.get("confidence", "medium"),
            citations=result.get("citations", []),
            retrieved_chunks=retrieved_chunks,
            references=result.get("references", []),
            execution_time_ms=(time.time() - t0) * 1000,
            token_usage=result.get("token_usage", {}),
            error=result.get("error"),
        )

    # ── Summarize: via workflow with intent override ───────────────────

    def summarize(
        self,
        act_name: str,
        language: str = "",
    ) -> AgentResponse:
        t0 = time.time()
        if not language:
            language = self.intent_detector.detect_language(act_name)

        state: AgentState = {
            "question": act_name,
            "language": language,
            "user_type": "general",
            "rewritten_question": f"Summarize the {act_name}",
            "query_type": "act_summary",
            "legal_domain": "general",
            "routed_agent": "",
            "agent_output": "",
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
            logger.exception("Summary workflow failed")
            return AgentResponse(
                question=f"Summarize: {act_name}",
                answer=f"Summary error: {str(e)}",
                language_detected=language,
                error=str(e),
            )

        chunks_raw = result.get("retrieved_chunks", [])
        retrieved_chunks: list[RetrievedChunk] = []
        for c in chunks_raw:
            if isinstance(c, RetrievedChunk):
                retrieved_chunks.append(c)
            elif isinstance(c, dict):
                retrieved_chunks.append(RetrievedChunk(**c))

        return AgentResponse(
            question=f"Summarize: {act_name}",
            answer=result.get("answer", ""),
            answer_markdown=result.get("answer", ""),
            language_detected=language,
            query_type="act_summary",
            confidence=result.get("confidence", "medium"),
            citations=result.get("citations", []),
            retrieved_chunks=retrieved_chunks,
            references=result.get("references", []),
            execution_time_ms=(time.time() - t0) * 1000,
            token_usage=result.get("token_usage", {}),
            error=result.get("error"),
        )
