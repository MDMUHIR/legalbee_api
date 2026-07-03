"""
Legal Bee FastAPI Application
REST API for Bangladeshi Legal RAG System
"""

import os
import logging
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
from agent import ask_law_question, detect_language

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Legal Bee API",
    description="Bangladeshi Legal RAG System - Ask legal questions in Bengali or English",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Request/Response Models ============


class LegalQuery(BaseModel):
    """Legal question request model"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the punishment for theft under Bangladeshi law?",
                "language": "en",
                "user_type": "general",
            }
        }
    )

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Legal question in Bengali or English",
    )
    language: Optional[str] = Field(
        None,
        description="Language hint: 'en' for English, 'bn' for Bengali (auto-detected if not provided)",
    )
    user_type: Optional[str] = Field(
        default="general",
        description="User type: 'lawyer' for detailed legal analysis, 'general' for simple explanation",
    )

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Legal question in Bengali or English",
    )
    language: Optional[str] = Field(
        None,
        description="Language hint: 'en' for English, 'bn' for Bengali (auto-detected if not provided)",
    )


class LegalAnswer(BaseModel):
    """Legal answer response model"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the punishment for theft?",
                "answer": "According to the Bangladeshi Penal Code...",
                "language_detected": "en",
                "timestamp": "2024-01-15T10:30:45.123456",
                "source": "qdrant_cloud_database",
            }
        }
    )

    question: str = Field(..., description="The original legal question")
    answer: str = Field(..., description="Legal answer from database")
    language_detected: str = Field(..., description="Detected language: 'en' or 'bn'")
    timestamp: str = Field(..., description="Response timestamp")
    source: str = Field(default="qdrant_cloud_database", description="Data source")


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(default="healthy", description="Service status")
    service: str = Field(default="Legal Bee API", description="Service name")
    version: str = Field(default="1.0.0", description="API version")
    timestamp: str = Field(..., description="Health check timestamp")


# ============ API Endpoints ============


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - redirects to API documentation"""
    return {
        "message": "Welcome to Legal Bee API",
        "docs": "/api/docs",
        "redoc": "/api/redoc",
        "endpoints": {"ask": "/api/ask", "health": "/api/health"},
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and connectivity to Qdrant"""
    try:
        # Verify Qdrant connection by checking credentials
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise HTTPException(
                status_code=503, detail="Qdrant Cloud credentials not configured"
            )

        return HealthResponse(timestamp=datetime.now().isoformat())
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/api/ask", response_model=LegalAnswer, tags=["Legal Query"])
async def ask_legal_question(query: LegalQuery):
    """
    Ask a legal question about Bangladeshi law

    The system will:
    1. Auto-detect language (Bengali or English)
    2. Search the legal database
    3. Return answers strictly from the database (no hallucination)
    4. Provide citations to relevant Acts and Sections

    Supported languages:
    - English: "What is the punishment for theft?"
    - Bengali: "চুরির শাস্তি কী?"
    """
    try:
        logger.info(f"Received legal query: {query.question[:50]}...")

        # Validate query
        if not query.question or len(query.question.strip()) < 3:
            raise HTTPException(
                status_code=400, detail="Question must be at least 3 characters long"
            )

        # Detect language
        lang = (
            query.language
            if query.language in ["en", "bn"]
            else detect_language(query.question)
        )
        logger.info(f"Detected language: {lang}")

        # Get answer from RAG system
        user_type = (
            query.user_type if query.user_type in ["lawyer", "general"] else "general"
        )
        answer = ask_law_question(query.question, user_type=user_type)

        if not answer or answer.strip() == "":
            raise HTTPException(
                status_code=500, detail="Failed to generate answer from database"
            )

        logger.info(f"Successfully answered query")

        return LegalAnswer(
            question=query.question,
            answer=answer,
            language_detected=lang,
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing legal query: {str(e)}"
        )


@app.post("/api/ask-batch", tags=["Legal Query"])
async def ask_multiple_questions(questions: list[LegalQuery]):
    """
    Ask multiple legal questions in a single batch request

    Returns an array of answers for multiple questions
    """
    try:
        results = []
        for query in questions:
            try:
                answer = ask_law_question(query.question)
                lang = (
                    query.language
                    if query.language in ["en", "bn"]
                    else detect_language(query.question)
                )
                results.append(
                    LegalAnswer(
                        question=query.question,
                        answer=answer,
                        language_detected=lang,
                        timestamp=datetime.now().isoformat(),
                    )
                )
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                results.append(
                    {
                        "question": query.question,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return {"count": len(results), "results": results}

    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Batch processing failed: {str(e)}"
        )


@app.get("/api/languages", tags=["Info"])
async def supported_languages():
    """Get list of supported languages"""
    return {
        "supported_languages": [
            {
                "code": "en",
                "name": "English",
                "example": "What is the punishment for theft under Bangladeshi law?",
            },
            {"code": "bn", "name": "Bengali (বাংলা)", "example": "বাংলাদেশে চুরির শাস্তি কী?"},
        ],
        "note": "Language is auto-detected if not specified",
    }


@app.get("/api/about", tags=["Info"])
async def about():
    """Get information about Legal Bee"""
    return {
        "name": "Legal Bee",
        "description": "Bangladeshi Legal RAG System - Agentic Retrieval Augmented Generation",
        "version": "1.0.0",
        "features": [
            "Bengali & English support",
            "Legal database retrieval",
            "Anti-hallucination mode (database-only answers)",
            "Citation of Acts and Sections",
            "Real-time legal queries",
        ],
        "database": "Qdrant Cloud Vector Database",
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "llm_model": "Groq Llama 3.1 8B Instant",
    }


# ============ Error Handlers ============


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat(),
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat(),
    }


# ============ Application Entry Point ============

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,  # Set to True for development
    )
