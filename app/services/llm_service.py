"""LLM service — multi-provider with automatic fallback (Gemini → Cerebras → OpenRouter)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.config import config

logger = logging.getLogger(__name__)


class FallbackChatModel:
    """Wraps multiple chat models and falls through to the next on failure."""

    def __init__(self, models: list[BaseChatModel], names: list[str]):
        self._models = models
        self._names = names

    def __getattr__(self, name):
        return getattr(self._models[0], name)

    def invoke(self, input: list[BaseMessage], **kwargs):
        last_error = None
        for i, model in enumerate(self._models):
            try:
                logger.debug("Trying provider %s", self._names[i])
                return model.invoke(input, **kwargs)
            except Exception as e:
                logger.warning("Provider '%s' failed: %s", self._names[i], e)
                last_error = e
        raise last_error or RuntimeError("All LLM providers failed")

    async def ainvoke(self, input: list[BaseMessage], **kwargs):
        last_error = None
        for i, model in enumerate(self._models):
            try:
                logger.debug("Trying provider %s", self._names[i])
                return await model.ainvoke(input, **kwargs)
            except Exception as e:
                logger.warning("Provider '%s' failed: %s", self._names[i], e)
                last_error = e
        raise last_error or RuntimeError("All LLM providers failed")


class LLMService:
    """Provides a LangChain-compatible chat model with fallback chain."""

    def __init__(self):
        self.provider = config.llm_provider
        self.model_name = config.llm_model
        self.temperature = config.llm_temperature
        self.max_tokens = config.llm_max_tokens
        self._llm: Optional[FallbackChatModel] = None

    @property
    def llm(self) -> FallbackChatModel:
        if self._llm is None:
            self._llm = self._build_fallback_chain()
        return self._llm

    def _build_fallback_chain(self) -> FallbackChatModel:
        models: list[BaseChatModel] = []
        names: list[str] = []

        primary = self._create_gemini()
        if primary is not None:
            models.append(primary)
            names.append("gemini")

        cerebras = self._create_cerebras()
        if cerebras is not None:
            models.append(cerebras)
            names.append("cerebras")

        openrouter = self._create_openrouter()
        if openrouter is not None:
            models.append(openrouter)
            names.append("openrouter")

        if not models:
            raise RuntimeError(
                "No LLM providers configured — set GOOGLE_API_KEY, CEREBRAS_API_KEY, or OPENROUTER_API_KEY"
            )

        logger.info("LLM fallback chain: %s", " → ".join(names))
        return FallbackChatModel(models, names)

    def _create_gemini(self) -> Optional[BaseChatModel]:
        if not config.google_api_key:
            return None
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                google_api_key=config.google_api_key,
            )
        finally:
            if gemini_key is not None:
                os.environ["GEMINI_API_KEY"] = gemini_key

    def _create_cerebras(self) -> Optional[BaseChatModel]:
        if not config.cerebras_api_key:
            return None
        from langchain_cerebras import ChatCerebras
        return ChatCerebras(
            model=config.cerebras_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=config.cerebras_api_key,
        )

    def _create_openrouter(self) -> Optional[BaseChatModel]:
        if not config.openrouter_api_key:
            return None
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.openrouter_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=config.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def get_model_name(self) -> str:
        return self.model_name


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()
