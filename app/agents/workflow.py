"""LangGraph workflow for the Legal Bee RAG agent.

Nodes:
  1. detect_intent    — classify query type, language, domain
  2. rewrite_query    — expand vague queries (optional)
  3. retrieve         — dense search in Qdrant
  4. build_context    — format retrieved chunks for LLM
  5. generate_answer  — LLM generates grounded answer
  6. build_citations  — extract citations from metadata

The graph is linear: detect → rewrite → retrieve → context → generate → citations
"""

from __future__ import annotations

import time
import logging
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from app.config import config
from app.services.llm_service import get_llm_service
from app.retrieval.retriever import LegalRetriever
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.citation_builder import CitationBuilder
from app.retrieval.intent_detector import IntentDetector
from app.prompts.system_prompt import get_system_prompt
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    language: str
    user_type: str
    rewritten_question: str
    query_type: str
    legal_domain: str
    retrieved_chunks: list[RetrievedChunk]
    context: str
    answer: str
    citations: list[str]
    references: list[dict]
    confidence: str
    needs_rewrite: bool
    execution_time_ms: float
    token_usage: dict
    error: Optional[str]


_retriever: Optional[LegalRetriever] = None
_rewriter: Optional[QueryRewriter] = None
_citation_builder: Optional[CitationBuilder] = None
_intent_detector: Optional[IntentDetector] = None


def _get_retriever() -> LegalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = LegalRetriever()
    return _retriever


def _get_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter


def _get_citation_builder() -> CitationBuilder:
    global _citation_builder
    if _citation_builder is None:
        _citation_builder = CitationBuilder()
    return _citation_builder


def _get_intent_detector() -> IntentDetector:
    global _intent_detector
    if _intent_detector is None:
        _intent_detector = IntentDetector()
    return _intent_detector


def detect_intent(state: AgentState) -> AgentState:
    detector = _get_intent_detector()
    question = state["question"]

    if not state.get("language"):
        state["language"] = detector.detect_language(question)

    state["query_type"] = detector.detect_query_type(question, retrieved_count=-1)
    state["legal_domain"] = detector.detect_legal_domain(question)
    state["needs_rewrite"] = detector.needs_rewrite(question)

    logger.info(
        "Intent: type=%s domain=%s lang=%s rewrite=%s",
        state["query_type"],
        state["legal_domain"],
        state["language"],
        state["needs_rewrite"],
    )
    return state


def rewrite_query(state: AgentState) -> AgentState:
    if not state.get("needs_rewrite"):
        state["rewritten_question"] = state["question"]
        return state

    rewriter = _get_rewriter()
    rewritten = rewriter.rewrite(state["question"], state.get("language", "en"))
    state["rewritten_question"] = rewritten
    return state


def retrieve(state: AgentState) -> AgentState:
    retriever = _get_retriever()
    query = state.get("rewritten_question") or state["question"]

    chunks = retriever.retrieve(query)

    if not chunks:
        chunks = retriever.retrieve(state["question"], score_threshold=0.2)

    state["retrieved_chunks"] = chunks
    logger.info("Retrieved %d chunks", len(chunks))
    return state


def build_context(state: AgentState) -> AgentState:
    builder = _get_citation_builder()
    chunks = state.get("retrieved_chunks", [])
    state["context"] = builder.build_context_string(chunks)
    return state


def generate_answer(state: AgentState) -> AgentState:
    t0 = time.time()
    llm_service = get_llm_service()
    llm = llm_service.llm

    context = state.get("context", "")
    language = state.get("language", "en")
    user_type = state.get("user_type", "general")
    question = state["question"]

    if not context.strip():
        if language == "bn":
            state["answer"] = "দুঃখিত, আমি উপলব্ধ বাংলাদেশী আইন ডাটাবেসে এই বিষয়ে কোনো প্রাসঙ্গিক তথ্য খুঁজে পাইনি।\n\nঅনুগ্রহ করে অন্য কোনো আইনি প্রশ্ন জিজ্ঞাসা করুন।"
        else:
            state["answer"] = "I could not find sufficient legal information in the available Bangladeshi law database for this query.\n\nPlease try rephrasing your question or ask about a different legal topic."
        state["confidence"] = "low"
        state["execution_time_ms"] = (time.time() - t0) * 1000
        state["token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0}
        return state

    system_msg = get_system_prompt(language, user_type)
    user_prompt = f"Question: {question}\n\n{system_msg}"

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [SystemMessage(content=system_msg), HumanMessage(content=question)]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        state["answer"] = content.strip()
        try:
            usage = response.response_metadata.get("token_usage", {})
            state["token_usage"] = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
        except Exception:
            state["token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0}

    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        if language == "bn":
            state["answer"] = f"একটি ত্রুটি ঘটেছে: {str(e)}"
        else:
            state["answer"] = f"An error occurred while generating the answer. Please try again."
        state["error"] = str(e)

    state["execution_time_ms"] = (time.time() - t0) * 1000

    chunk_count = len(state.get("retrieved_chunks", []))
    if chunk_count >= 5:
        state["confidence"] = "high"
    elif chunk_count >= 2:
        state["confidence"] = "medium"
    else:
        state["confidence"] = "low"

    return state


def build_citations(state: AgentState) -> AgentState:
    builder = _get_citation_builder()
    chunks = state.get("retrieved_chunks", [])
    state["citations"] = builder.build(chunks)
    state["references"] = builder.build_references(chunks)
    return state


def create_workflow() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("detect_intent", detect_intent)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("build_context", build_context)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("build_citations", build_citations)

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "build_context")
    graph.add_edge("build_context", "generate_answer")
    graph.add_edge("generate_answer", "build_citations")
    graph.add_edge("build_citations", END)

    return graph.compile()
