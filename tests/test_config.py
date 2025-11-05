import pytest

from startup_flow.config import get_llm_settings


@pytest.fixture(autouse=True)
def clear_llm_cache() -> None:
    """Ensure cached settings do not leak between tests."""

    get_llm_settings.cache_clear()


def test_primary_provider_prefers_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")

    settings = get_llm_settings()

    assert settings.primary_provider == "openai"
    assert settings.get_api_key() == "test-openai"


def test_primary_provider_falls_back_to_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("FOUNDEROS_LLM_GROQ_API_KEY", "groq-key")

    settings = get_llm_settings()

    assert settings.primary_provider == "groq"
    assert settings.get_api_key() == "groq-key"
