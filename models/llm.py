"""
LLM factory.

LLM_PROVIDER selects the primary Researcher model:
- ollama: local default
- nvidia: NVIDIA build.nvidia.com (OpenAI-compatible, tool calling)
- groq:   Groq (OpenAI-compatible)

LLM_FALLBACK_PROVIDERS (comma separated) are tried in order when the
primary call raises, e.g. LLM_PROVIDER=nvidia, LLM_FALLBACK_PROVIDERS=groq.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from config.settings import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    LLM_FALLBACK_PROVIDERS,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_RESEARCHER_MODEL,
    OLLAMA_MODEL,
)


def build_chat_model(provider: str) -> BaseChatModel:
    """Create one chat model for a named provider."""

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=OLLAMA_MODEL, temperature=LLM_TEMPERATURE)

    if provider in {"nvidia", "groq"}:
        from langchain_openai import ChatOpenAI

        if provider == "nvidia":
            key, base_url, model = (
                NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_RESEARCHER_MODEL
            )
        else:
            key, base_url, model = GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

        if not key:
            raise ValueError(f"{provider.upper()}_API_KEY is not set.")

        return ChatOpenAI(
            model=model,
            api_key=key,
            base_url=base_url,
            temperature=LLM_TEMPERATURE,
            max_retries=1,
            timeout=60,
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")


def _configured_providers() -> list[str]:
    """Primary provider first, then fallbacks that are usable."""

    providers = [LLM_PROVIDER]
    for name in LLM_FALLBACK_PROVIDERS:
        if name not in providers:
            providers.append(name)
    return providers


def _with_fallbacks(make: Any) -> Runnable:
    """Apply `make(model)` per provider and chain with fallbacks."""

    runnables: list[Runnable] = []
    for name in _configured_providers():
        try:
            runnables.append(make(build_chat_model(name)))
        except ValueError as error:
            # A missing key on a fallback must not break startup;
            # a missing key on the primary provider should.
            if name == LLM_PROVIDER:
                raise
            print(f">>> skipping fallback provider {name}: {error}",
                  flush=True)

    primary, *rest = runnables
    return primary.with_fallbacks(rest) if rest else primary


def get_llm() -> Runnable:
    """Free-text model with provider fallbacks."""

    return _with_fallbacks(lambda model: model)


def get_structured_llm(schema: type) -> Runnable:
    """Structured-output model with provider fallbacks."""

    return _with_fallbacks(lambda model: model.with_structured_output(schema))


llm = get_llm()
