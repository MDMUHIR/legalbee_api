"""
Legal Bee API — FastAPI application entry point.

Run:
    python -m app.main
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

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


_FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend.html"


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_frontend():
    if _FRONTEND_PATH.exists():
        return _FRONTEND_PATH.read_text(encoding="utf-8")
    return """
    <html><body>
    <h1>Legal Bee API</h1>
    <p>Frontend not found. Visit <a href="/api/docs">/api/docs</a> for API docs.</p>
    </body></html>
    """


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
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
