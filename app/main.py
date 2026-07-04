"""
Legal Bee API — FastAPI application entry point.

Run:
    python -m app.main
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Legal Bee API",
    description="Bangladeshi Legal RAG System — Ask legal questions in Bengali or English",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Welcome to Legal Bee API",
        "docs": "/api/docs",
        "redoc": "/api/redoc",
        "endpoints": {
            "chat": "POST /api/chat",
            "search": "POST /api/search",
            "analyze": "POST /api/analyze",
            "summary": "POST /api/summary",
            "health": "GET /api/health",
        },
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat(),
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", str(exc))
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
