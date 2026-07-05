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

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-2.5-flash"))
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0"))
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2048"))
    )

    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))

    cerebras_api_key: str = field(default_factory=lambda: os.getenv("CEREBRAS_API_KEY", ""))
    cerebras_model: str = field(default_factory=lambda: os.getenv("CEREBRAS_MODEL", "gpt-oss-120b"))

    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_model: str = field(default_factory=lambda: os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free"))

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
        if self.llm_provider == "gemini" and not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")
        if not self.cerebras_api_key:
            errors.append("CEREBRAS_API_KEY is required (first fallback provider)")
        if not self.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY is required (second fallback provider)")
        return errors


config = Config()
