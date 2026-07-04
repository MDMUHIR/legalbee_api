"""LLM service — provider-agnostic abstraction over Groq and Gemini."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from langchain_core.language_models import BaseChatModel

from app.config import config

logger = logging.getLogger(__name__)


class LLMService:
    """Provides a LangChain-compatible chat model configured from env."""

    def __init__(self):
        self.provider = config.llm_provider
        self.model_name = config.llm_model
        self.temperature = config.llm_temperature
        self.max_tokens = config.llm_max_tokens
        self._llm: Optional[BaseChatModel] = None

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self) -> BaseChatModel:
        if self.provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=config.groq_api_key,
            )
        elif self.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                google_api_key=config.google_api_key,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def get_model_name(self) -> str:
        return self.model_name


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()
