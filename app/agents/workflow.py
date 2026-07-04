"""Multi-agent LangGraph workflow for the Legal Bee RAG system.

Architecture:
  User Query
      │
      ▼
  detect_intent → rewrite_query → query_router
      │
      ├── Law Search Agent      (specific provisions)
      ├── Legal Analysis Agent  (fact scenarios)
      ├── Act Summary Agent     (summarize acts)
      ├── Amendment Compare Agent (what changed)
      └── Legal QA Agent        (general questions)
            │
            ▼
  verify_citations → format_response → END

Every agent uses shared retriever + LLM, but has its own:
  - Retrieval strategy (filters, top_k)
  - System prompt (specialized instructions)
  - Output format

This architecture is designed to be extended with new agents
(e.g., Case Law, Document Drafting, Multilingual Translation)
without modifying the core router or shared services.
"""

from __future__ import annotations

import time
import logging
from typing import TypedDict, Optional, Literal

from langgraph.graph import StateGraph, END

from app.services.llm_service import get_llm_service
from app.retrieval.retriever import LegalRetriever
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.citation_builder import CitationBuilder
from app.retrieval.intent_detector import IntentDetector
from app.prompts.system_prompt import (
    get_system_prompt,
    get_agent_prompt,
)
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

AgentName = Literal[
    "law_search", "legal_analysis", "act_summary",
    "amendment_compare", "legal_qa", "citation_verify",
]


class AgentState(TypedDict, total=False):
    question: str
    language: str
    user_type: str
    rewritten_question: str
    query_type: str
    legal_domain: str
    routed_agent: str
    agent_output: str
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


# ── Shared singletons ──────────────────────────────────────────────────

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


# ── Node: detect_intent ────────────────────────────────────────────────

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
        state["query_type"], state["legal_domain"],
        state["language"], state["needs_rewrite"],
    )
    return state


# ── Node: rewrite_query ────────────────────────────────────────────────

def rewrite_query(state: AgentState) -> AgentState:
    if not state.get("needs_rewrite"):
        state["rewritten_question"] = state["question"]
        return state
    rewriter = _get_rewriter()
    state["rewritten_question"] = rewriter.rewrite(
        state["question"], state.get("language", "en")
    )
    return state


# ── Node: query_router ─────────────────────────────────────────────────

def query_router(state: AgentState) -> AgentState:
    """Route query to the appropriate specialized agent."""
    qtype = state.get("query_type", "legal_question")

    routing = {
        "law_search": "law_search_agent",
        "fact_analysis": "legal_analysis_agent",
        "act_summary": "act_summary_agent",
        "amendment_question": "amendment_compare_agent",
    }

    agent = routing.get(qtype, "legal_qa_agent")
    state["routed_agent"] = agent
    logger.info("Router: %s → %s", qtype, agent)
    return state


# ── Shared retrieval helper ────────────────────────────────────────────

def _retrieve_and_build_context(
    state: AgentState, query: str, top_k: int = 8, filters: dict | None = None
) -> AgentState:
    retriever = _get_retriever()
    builder = _get_citation_builder()

    chunks = retriever.retrieve(query, top_k=top_k, metadata_filter=filters)
    if not chunks:
        chunks = retriever.retrieve(state["question"], score_threshold=0.15, top_k=top_k)

    state["retrieved_chunks"] = chunks
    state["context"] = builder.build_context_string(chunks)
    logger.info("Retrieved %d chunks for agent %s", len(chunks), state.get("routed_agent"))
    return state


# ── Shared LLM caller ──────────────────────────────────────────────────

def _call_llm(
    system_prompt: str,
    question: str,
    language: str,
) -> tuple[str, dict]:
    t0 = time.time()
    llm = get_llm_service().llm

    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]

    try:
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        usage = {}
        try:
            u = response.response_metadata.get("token_usage", {})
            usage = {"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0)}
        except Exception:
            pass
        return content.strip(), usage
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        if language == "bn":
            return f"একটি ত্রুটি ঘটেছে: {str(e)}", {}
        return f"An error occurred: {str(e)}", {}


# ── NO-OP: shared confidence scorer ────────────────────────────────────

def _score_confidence(chunks: list) -> str:
    n = len(chunks)
    if n >= 5:
        return "high"
    if n >= 2:
        return "medium"
    return "low"


# ═══════════════════════════════════════════════════════════════════════
# SPECIALIZED AGENT NODES
# ═══════════════════════════════════════════════════════════════════════

# ── Agent: Law Search ──────────────────────────────────────────────────

def law_search_agent(state: AgentState) -> AgentState:
    """Find specific legal provisions by section/article number."""
    query = state.get("rewritten_question") or state["question"]
    language = state.get("language", "en")
    user_type = state.get("user_type", "general")

    state = _retrieve_and_build_context(state, query, top_k=10)
    context = state.get("context", "")

    if not context.strip():
        state["answer"] = (
            "I could not find that specific legal provision. Please check the section/article number and try again."
            if language == "en"
            else "আমি সেই নির্দিষ্ট আইনি বিধানটি খুঁজে পাইনি। অনুগ্রহ করে ধারা/অনুচ্ছেদ নম্বর পরীক্ষা করে আবার চেষ্টা করুন।"
        )
        state["confidence"] = "low"
        return state

    prompt = get_agent_prompt("law_search", language, user_type, context)
    answer, usage = _call_llm(prompt, state["question"], language)

    state["answer"] = answer
    state["token_usage"] = usage
    state["confidence"] = _score_confidence(state.get("retrieved_chunks", []))
    return state


# ── Agent: Legal Analysis ─────────────────────────────────────────────

def legal_analysis_agent(state: AgentState) -> AgentState:
    """Analyze a factual scenario against Bangladeshi law."""
    query = state.get("rewritten_question") or state["question"]
    language = state.get("language", "en")
    user_type = state.get("user_type", "general")

    state = _retrieve_and_build_context(state, query, top_k=12)
    context = state.get("context", "")

    if not context.strip():
        state["answer"] = (
            "I could not find relevant Bangladeshi law for your situation. Please provide more details or consult a lawyer directly."
            if language == "en"
            else "আপনার পরিস্থিতির জন্য কোনো প্রাসঙ্গিক বাংলাদেশী আইন খুঁজে পাওয়া যায়নি। অনুগ্রহ করে আরও বিস্তারিত জানান বা সরাসরি একজন আইনজীবীর সাথে পরামর্শ করুন।"
        )
        state["confidence"] = "low"
        return state

    prompt = get_agent_prompt("legal_analysis", language, user_type, context)
    answer, usage = _call_llm(prompt, state["question"], language)

    state["answer"] = answer
    state["token_usage"] = usage
    state["confidence"] = _score_confidence(state.get("retrieved_chunks", []))
    return state


# ── Agent: Act Summary ─────────────────────────────────────────────────

def act_summary_agent(state: AgentState) -> AgentState:
    """Summarize a Bangladeshi legal act."""
    question = state["question"]
    language = state.get("language", "en")
    user_type = state.get("user_type", "general")

    retriever = _get_retriever()
    builder = _get_citation_builder()

    chunks = retriever.retrieve_by_act(question, top_k=20)
    if not chunks:
        chunks = retriever.retrieve(question, top_k=15)

    state["retrieved_chunks"] = chunks
    state["context"] = builder.build_context_string(chunks, max_chunks=15)

    context = state.get("context", "")
    if not context.strip():
        state["answer"] = (
            f"No information found for '{question}'. The act may not be in the database yet."
            if language == "en"
            else f"'{question}' এর জন্য কোনো তথ্য পাওয়া যায়নি। আইনটি ডাটাবেসে নাও থাকতে পারে।"
        )
        state["confidence"] = "low"
        return state

    prompt = get_agent_prompt("act_summary", language, user_type, context)
    answer, usage = _call_llm(prompt, question, language)

    state["answer"] = answer
    state["token_usage"] = usage
    state["confidence"] = "high" if len(chunks) >= 10 else "medium"
    return state


# ── Agent: Amendment Comparison ────────────────────────────────────────

def amendment_compare_agent(state: AgentState) -> AgentState:
    """Compare original law vs amendments — what changed."""
    query = state.get("rewritten_question") or state["question"]
    language = state.get("language", "en")
    user_type = state.get("user_type", "general")

    state = _retrieve_and_build_context(state, query, top_k=15)
    context = state.get("context", "")

    if not context.strip():
        state["answer"] = (
            "I could not find amendment information for this query. The amendment may not be in the database yet."
            if language == "en"
            else "এই প্রশ্নের জন্য সংশোধন সংক্রান্ত তথ্য পাওয়া যায়নি। সংশোধনটি ডাটাবেসে নাও থাকতে পারে।"
        )
        state["confidence"] = "low"
        return state

    prompt = get_agent_prompt("amendment_compare", language, user_type, context)
    answer, usage = _call_llm(prompt, state["question"], language)

    state["answer"] = answer
    state["token_usage"] = usage
    state["confidence"] = _score_confidence(state.get("retrieved_chunks", []))
    return state


# ── Agent: Legal QA (default) ──────────────────────────────────────────

def legal_qa_agent(state: AgentState) -> AgentState:
    """General legal question answering (default agent)."""
    query = state.get("rewritten_question") or state["question"]
    language = state.get("language", "en")
    user_type = state.get("user_type", "general")

    state = _retrieve_and_build_context(state, query, top_k=8)
    context = state.get("context", "")

    if not context.strip():
        state["answer"] = (
            "I could not find sufficient legal information in the available Bangladeshi law database for this query.\n\nPlease try rephrasing your question or ask about a different legal topic."
            if language == "en"
            else "আমি উপলব্ধ বাংলাদেশী আইন ডাটাবেসে পর্যাপ্ত আইনি তথ্য খুঁজে পাইনি।\n\nঅনুগ্রহ করে আপনার প্রশ্নটি অন্যভাবে জিজ্ঞাসা করুন বা ভিন্ন কোনো আইনি বিষয়ে জানতে চান।"
        )
        state["confidence"] = "low"
        return state

    prompt = get_agent_prompt("legal_qa", language, user_type, context)
    answer, usage = _call_llm(prompt, state["question"], language)

    state["answer"] = answer
    state["token_usage"] = usage
    state["confidence"] = _score_confidence(state.get("retrieved_chunks", []))
    return state


# ── Node: verify_citations ─────────────────────────────────────────────

def verify_citations(state: AgentState) -> AgentState:
    """Extract and verify citations from retrieved chunk metadata.

    Uses pre-built citations from the ingestion pipeline.
    Never generates citations from scratch.
    """
    builder = _get_citation_builder()
    chunks = state.get("retrieved_chunks", [])
    state["citations"] = builder.build(chunks)
    state["references"] = builder.build_references(chunks)

    logger.info(
        "Citations: %d extracted, %d references",
        len(state["citations"]), len(state["references"]),
    )
    return state


# ── Node: format_response ──────────────────────────────────────────────

def format_response(state: AgentState) -> AgentState:
    """Final response formatting and execution time tracking."""
    state["execution_time_ms"] = 0
    logger.info(
        "Response ready: agent=%s confidence=%s citations=%d",
        state.get("routed_agent", "unknown"),
        state.get("confidence", "unknown"),
        len(state.get("citations", [])),
    )
    return state


# ═══════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════

def _route_to_agent(state: AgentState) -> str:
    """Conditional edge: route to the selected specialized agent."""
    return state.get("routed_agent", "legal_qa_agent")


def create_workflow() -> StateGraph:
    graph = StateGraph(AgentState)

    # ── Core nodes ──
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("query_router", query_router)

    # ── Specialized agents ──
    graph.add_node("law_search_agent", law_search_agent)
    graph.add_node("legal_analysis_agent", legal_analysis_agent)
    graph.add_node("act_summary_agent", act_summary_agent)
    graph.add_node("amendment_compare_agent", amendment_compare_agent)
    graph.add_node("legal_qa_agent", legal_qa_agent)

    # ── Post-agent nodes ──
    graph.add_node("verify_citations", verify_citations)
    graph.add_node("format_response", format_response)

    # ── Edges: core pipeline ──
    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "rewrite_query")
    graph.add_edge("rewrite_query", "query_router")

    # ── Edges: router → agents (conditional) ──
    graph.add_conditional_edges(
        "query_router",
        _route_to_agent,
        {
            "law_search_agent": "law_search_agent",
            "legal_analysis_agent": "legal_analysis_agent",
            "act_summary_agent": "act_summary_agent",
            "amendment_compare_agent": "amendment_compare_agent",
            "legal_qa_agent": "legal_qa_agent",
        },
    )

    # ── Edges: all agents converge to citation verification → END ──
    for agent_node in [
        "law_search_agent", "legal_analysis_agent",
        "act_summary_agent", "amendment_compare_agent", "legal_qa_agent",
    ]:
        graph.add_edge(agent_node, "verify_citations")

    graph.add_edge("verify_citations", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()
