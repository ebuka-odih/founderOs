"""Journey stage endpoints for the FounderOs FastAPI backend."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..journey import generate_stage, list_personas, list_stage_definitions
from ..memory import session_memory
from ..schemas import (
    BusinessStage,
    PersonaDefinition,
    SessionResponse,
    StageDefinition,
    StageRequest,
    StageResponse,
)


router = APIRouter(prefix="/journey", tags=["journey"])


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    """Simple health check endpoint."""

    return {"status": "ok"}


@router.get("/stages", response_model=list[StageDefinition])
async def list_stages() -> list[StageDefinition]:
    """Expose stage metadata to the UI."""

    return list_stage_definitions()


@router.get("/personas", response_model=list[PersonaDefinition])
async def list_personas_endpoint() -> list[PersonaDefinition]:
    """Expose founder clone profiles to the UI."""

    return list_personas()


@router.post("/{stage}", response_model=StageResponse)
async def run_stage(stage: BusinessStage, payload: StageRequest) -> StageResponse:
    """Generate the requested stage deliverable."""

    response = generate_stage(stage, payload)
    if payload.session_id:
        session_memory.upsert_stage(payload.session_id, stage.value, response)
    return response


@router.get("/session/{session_id}", response_model=SessionResponse)
async def fetch_session(session_id: str) -> SessionResponse:
    """Return all stored stage results for the given session."""

    stages = session_memory.get_session(session_id)
    if not stages:
        raise HTTPException(status_code=404, detail=f"No journey data found for session '{session_id}'.")

    combined = session_memory.combined_markdown(session_id) or ""

    # Rehydrate StageResponse models for response payload.
    stage_models = {
        slug: StageResponse.model_validate(data)
        for slug, data in stages.items()
    }

    return SessionResponse(session_id=session_id, stages=stage_models, combined_markdown=combined)
