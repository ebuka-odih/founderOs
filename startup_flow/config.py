"""Configuration helpers for FounderOs backend."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Mapping

from dotenv import load_dotenv

OTHER_PROVIDER_PREFIX = "FOUNDEROS_LLM_"
OTHER_PROVIDER_SUFFIX = "_API_KEY"

load_dotenv(override=False)


@dataclass(frozen=True)
class LLMSettings:
    """Settings container for LLM provider API keys.

    OpenAI is considered the primary provider; if its key is missing the
    configuration falls back to other vendors in priority order.
    """

    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    anthropic_api_key: str | None = None
    # Additional providers discovered from environment variables.
    additional_api_keys: Dict[str, str] = field(default_factory=dict)

    @property
    def primary_provider(self) -> str | None:
        """Return the preferred provider based on available credentials."""

        if self.openai_api_key:
            return "openai"
        if self.gemini_api_key:
            return "gemini"
        if self.anthropic_api_key:
            return "anthropic"
        for provider, api_key in self.additional_api_keys.items():
            if api_key:
                return provider
        return None

    def get_api_key(self, provider: str | None = None) -> str | None:
        """Return the API key for the requested provider.

        When *provider* is omitted the primary provider's key is returned.
        """

        resolved_provider = provider or self.primary_provider
        if resolved_provider == "openai":
            return self.openai_api_key
        if resolved_provider == "gemini":
            return self.gemini_api_key
        if resolved_provider == "anthropic":
            return self.anthropic_api_key
        if resolved_provider is None:
            return None
        return self.additional_api_keys.get(resolved_provider)

    @property
    def has_any_keys(self) -> bool:
        """True when at least one provider API key is configured."""

        if self.primary_provider:
            return True
        return any(self.additional_api_keys.values())


def _extract_additional_api_keys(environ: Mapping[str, str]) -> Dict[str, str]:
    """Collect provider keys having the ``FOUNDEROS_LLM_*_API_KEY`` pattern."""

    discovered: Dict[str, str] = {}
    for env_key, value in environ.items():
        if not env_key.startswith(OTHER_PROVIDER_PREFIX) or not env_key.endswith(OTHER_PROVIDER_SUFFIX):
            continue

        provider = env_key[len(OTHER_PROVIDER_PREFIX) : -len(OTHER_PROVIDER_SUFFIX)].lower()
        if provider in {"openai", "gemini", "anthropic"}:
            # Skip duplicates for providers already handled explicitly.
            continue
        if value:
            discovered[provider] = value
    return discovered


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    """Read environment variables and return cached LLM settings."""

    environ = os.environ
    return LLMSettings(
        openai_api_key=environ.get("OPENAI_API_KEY"),
        gemini_api_key=environ.get("GEMINI_API_KEY"),
        anthropic_api_key=environ.get("ANTHROPIC_API_KEY"),
        additional_api_keys=_extract_additional_api_keys(environ),
    )
