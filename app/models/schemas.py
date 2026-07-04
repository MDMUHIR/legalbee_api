"""Pydantic models for the Legal Bee API."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class ChatRequest(BaseModel):
    """A chat message to the legal agent."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the punishment for theft under Bangladeshi law?",
                "language": "en",
                "user_type": "general",
                "conversation_id": None,
            }
        }
    )

    question: str = Field(..., min_length=1, max_length=4000, description="Legal question")
    language: Optional[str] = Field(None, description="'en' or 'bn' (auto-detected if none)")
    user_type: str = Field(default="general", description="'lawyer' or 'general'")
    conversation_id: Optional[str] = Field(None, description="For multi-turn conversations")


class SearchRequest(BaseModel):
    """A law search request."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Article 90E",
                "filters": {"act_name": "Representation of the People Order, 1972"},
                "language": "en",
                "top_k": 8,
            }
        }
    )

    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    filters: Optional[dict] = Field(None, description="Metadata filters")
    language: Optional[str] = Field(None)
    top_k: int = Field(default=8, ge=1, le=50)


class AnalyzeRequest(BaseModel):
    """A fact analysis request."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "facts": "My landlord has increased rent without notice.",
                "language": "en",
            }
        }
    )

    facts: str = Field(..., min_length=5, max_length=5000, description="Factual scenario")
    language: Optional[str] = Field(None)


class SummaryRequest(BaseModel):
    """An act summary request."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "act_name": "Representation of the People (Amendment) Act, 2026",
                "language": "en",
            }
        }
    )

    act_name: str = Field(..., min_length=1, max_length=500, description="Act name to summarize")
    language: Optional[str] = Field(None)


class RetrievedChunk(BaseModel):
    """A retrieved legal chunk with metadata."""

    text: str = ""
    chunk_id: str = ""
    score: float = 0.0
    chunk_type: str = ""
    citation: str = ""
    act_name: str = ""
    year: int = 0
    hierarchy: dict = Field(default_factory=dict)
    references: list[dict] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Response from the legal agent."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the punishment for theft?",
                "answer": "## Answer\n\nAccording to Section 379...",
                "answer_markdown": "## Answer\n\n...",
                "language_detected": "en",
                "query_type": "legal_question",
                "confidence": "high",
                "citations": ["Penal Code, 1860, Section 379"],
                "retrieved_chunks": 8,
                "references": [{"type": "act", "target": "Penal Code, 1860"}],
                "execution_time_ms": 1234,
                "token_usage": {"prompt_tokens": 500, "completion_tokens": 200},
                "timestamp": "2026-01-01T00:00:00",
            }
        }
    )

    question: str
    answer: str
    answer_markdown: str = ""
    language_detected: str = "en"
    query_type: str = ""
    confidence: str = "medium"
    citations: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    references: list[dict] = Field(default_factory=list)
    execution_time_ms: float = 0
    token_usage: dict = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "Legal Bee API"
    version: str = "1.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
