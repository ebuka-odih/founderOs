"""Application factory for the FounderOs FastAPI backend."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_llm_settings
from .routers import stages


DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]


def _resolve_allowed_origins() -> list[str]:
    """Return allowed origins, optionally sourced from an env override."""

    raw = os.getenv("FOUNDEROS_ALLOWED_ORIGINS")
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return DEFAULT_ALLOWED_ORIGINS


def create_app() -> FastAPI:
    """Create and configure a FastAPI application instance."""
    app = FastAPI(
        title="FounderOs Backend",
        version="0.1.0",
        description="AI-powered startup validation backend for FounderOs.",
    )
    allowed_origins = _resolve_allowed_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_origin_regex=r"http://localhost:\d+$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.llm_settings = get_llm_settings()
    app.include_router(stages.router)
    return app


app = create_app()
