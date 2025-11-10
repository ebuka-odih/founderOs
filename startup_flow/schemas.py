"""Pydantic models and enums for the FounderOs startup journey API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BusinessStage(str, Enum):
    """Enumerate the supported journey stages."""

    IDEATION = "ideation"
    VALIDATION = "validation"
    PLANNING = "planning"
    STRATEGY = "strategy"
    LAUNCH = "launch"
    SCALE = "scale"

    @property
    def order(self) -> int:
        """Return a human-friendly order index for the stage."""
        stage_order = {
            BusinessStage.IDEATION: 1,
            BusinessStage.VALIDATION: 2,
            BusinessStage.PLANNING: 3,
            BusinessStage.STRATEGY: 4,
            BusinessStage.LAUNCH: 5,
            BusinessStage.SCALE: 6,
        }
        return stage_order[self]


class FounderClone(str, Enum):
    """Enumerate the available AI personas."""

    ELON = "elon"
    JEFF = "jeff"
    NANCY = "nancy"


class StageRequest(BaseModel):
    """Payload for generating a stage deliverable."""

    prompt: str = Field(
        ...,
        min_length=10,
        description="Primary startup prompt or idea supplied by the user.",
    )
    clone: Optional[FounderClone] = Field(
        default=None,
        description="Optional founder clone persona that guides the response tone.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier used to stitch together multi-stage flows.",
    )
    context_markdown: Optional[str] = Field(
        default=None,
        description="Optional markdown context carried forward from a prior stage.",
    )
    context_summary: Optional[str] = Field(
        default=None,
        description="Optional short summary derived from a previous stage.",
    )
    context_structured: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured payload forwarded from a previous stage.",
    )


class StageResponse(BaseModel):
    """Structured response for a single stage."""

    stage: BusinessStage
    stage_label: str
    clone: FounderClone
    summary: str
    markdown: str
    structured: Dict[str, Any]


class StageDefinition(BaseModel):
    """Expose metadata that describes a stage to the UI."""

    id: BusinessStage
    label: str
    description: str


class PersonaDefinition(BaseModel):
    """Expose persona metadata to the UI."""

    id: FounderClone
    label: str
    tagline: str
    tone: str


class SessionResponse(BaseModel):
    """Aggregate the stage results for a session."""

    session_id: str
    stages: Dict[str, StageResponse]
    combined_markdown: str


class JourneyRunRequest(BaseModel):
    """Request payload for running the full journey pipeline."""

    prompt: str = Field(..., min_length=10)
    clone: Optional[FounderClone] = None
    session_id: Optional[str] = None


class JourneyRunResponse(SessionResponse):
    """Reuse the session response schema for immediate journey runs."""

    pass
