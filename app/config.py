"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "bangladesh_laws")
    )
    dense_vector_name: str = field(
        default_factory=lambda: os.getenv("DENSE_VECTOR_NAME", "law_dense_vector")
    )

    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    )

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0"))
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2048"))
    )

    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))

    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "8")))
    score_threshold: float = field(
        default_factory=lambda: float(os.getenv("SCORE_THRESHOLD", "0.3"))
    )
    use_reranker: bool = field(
        default_factory=lambda: os.getenv("USE_RERANKER", "false").lower() == "true"
    )
    use_hybrid_search: bool = field(
        default_factory=lambda: os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"
    )

    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "2")))
    request_timeout: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "60")))

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.qdrant_url:
            errors.append("QDRANT_URL is required")
        if not self.qdrant_api_key:
            errors.append("QDRANT_API_KEY is required")
        if self.llm_provider == "groq" and not self.groq_api_key:
            errors.append("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        if self.llm_provider == "gemini" and not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")
        return errors


config = Config()
