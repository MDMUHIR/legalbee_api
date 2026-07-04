"""FastAPI routes for the Legal Bee RAG API."""

from __future__ import annotations

import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.agents.legal_agent import LegalBeeAgent
from app.models.schemas import (
    ChatRequest,
    SearchRequest,
    AnalyzeRequest,
    SummaryRequest,
    AgentResponse,
    HealthResponse,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_agent: Optional[LegalBeeAgent] = None


def get_agent() -> LegalBeeAgent:
    global _agent
    if _agent is None:
        _agent = LegalBeeAgent()
    return _agent


@router.post("/chat", response_model=AgentResponse, tags=["Agent"])
async def chat(request: ChatRequest):
    """Ask a legal question. Returns a grounded answer with citations."""
    try:
        if not request.question.strip() or len(request.question.strip()) < 3:
            raise HTTPException(status_code=400, detail="Question must be at least 3 characters")

        agent = get_agent()
        return agent.chat(
            question=request.question.strip(),
            language=request.language or "",
            user_type=request.user_type,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/search", response_model=AgentResponse, tags=["Agent"])
async def search(request: SearchRequest):
    """Search Bangladeshi laws with optional metadata filters."""
    try:
        agent = get_agent()
        return agent.search(
            query=request.query.strip(),
            language=request.language or "",
            filters=request.filters,
            top_k=request.top_k,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Search endpoint error")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@router.post("/analyze", response_model=AgentResponse, tags=["Agent"])
async def analyze(request: AnalyzeRequest):
    """Analyze a factual scenario according to Bangladeshi law."""
    try:
        agent = get_agent()
        return agent.analyze(
            facts=request.facts.strip(),
            language=request.language or "",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analyze endpoint error")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.post("/summary", response_model=AgentResponse, tags=["Agent"])
async def summary(request: SummaryRequest):
    """Summarize a Bangladeshi legal act."""
    try:
        agent = get_agent()
        return agent.summarize(
            act_name=request.act_name.strip(),
            language=request.language or "",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Summary endpoint error")
        raise HTTPException(status_code=500, detail=f"Summary error: {str(e)}")


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Service health check."""
    return HealthResponse(timestamp=datetime.now().isoformat())


@router.get("/languages", tags=["Info"])
async def languages():
    return {
        "supported_languages": [
            {"code": "en", "name": "English"},
            {"code": "bn", "name": "Bengali (বাংলা)"},
        ],
        "note": "Language is auto-detected if not specified",
    }
