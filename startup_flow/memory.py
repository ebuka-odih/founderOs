"""Simple in-memory store for FounderOs journey stage responses."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict

from .schemas import BusinessStage, StageResponse


class SessionMemory:
    """Persist per-session stage outputs so the UI can rebuild context."""

    def __init__(self) -> None:
        self._store: DefaultDict[str, Dict[str, dict]] = defaultdict(dict)

    def upsert_stage(self, session_id: str, stage: str, response: StageResponse) -> None:
        """Persist a stage response for the given session."""

        self._store[session_id][stage] = response.model_dump()

    def get_session(self, session_id: str) -> Dict[str, dict]:
        """Return a shallow copy of the stored stage responses."""

        return dict(self._store.get(session_id, {}))

    def combined_markdown(self, session_id: str) -> str | None:
        """Concatenate stage markdown in stage order for export."""

        stage_data = self._store.get(session_id)
        if not stage_data:
            return None

        ordered_markdown = []
        for stage, data in sorted(
            stage_data.items(),
            key=lambda item: BusinessStage(item[0]).order if item[0] in BusinessStage._value2member_map_ else item[0],
        ):
            markdown = data.get("markdown")
            if markdown:
                ordered_markdown.append(markdown)
        if not ordered_markdown:
            return None

        return "\n\n---\n\n".join(ordered_markdown)


session_memory = SessionMemory()
