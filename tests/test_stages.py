from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from startup_flow.app import create_app
from startup_flow.schemas import BusinessStage, FounderClone


client = TestClient(create_app())


def _sample_payload() -> dict[str, object]:
    return {
        "prompt": "A platform that helps remote teams align around focused weekly outcomes using AI coaches.",
        "clone": FounderClone.NANCY.value,
        "session_id": "demo-session",
    }


def test_healthcheck() -> None:
    response = client.get("/journey/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_stages_exposes_all() -> None:
    response = client.get("/journey/stages")
    assert response.status_code == 200
    stages = response.json()
    assert len(stages) == len(BusinessStage)
    ids = {item["id"] for item in stages}
    assert ids == {stage.value for stage in BusinessStage}


def test_list_personas_exposes_founder_clones() -> None:
    response = client.get("/journey/personas")
    assert response.status_code == 200
    personas = response.json()
    assert {item["id"] for item in personas} == {clone.value for clone in FounderClone}


@pytest.mark.parametrize("stage", [stage.value for stage in BusinessStage])
def test_generate_stage_returns_markdown(stage: str) -> None:
    payload = _sample_payload()
    response = client.post(f"/journey/{stage}", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["stage"] == stage
    assert isinstance(data["markdown"], str) and data["markdown"].startswith("##")
    assert isinstance(data["structured"], dict)


def test_session_endpoint_returns_combined_markdown() -> None:
    payload = _sample_payload()
    client.post(f"/journey/{BusinessStage.IDEATION.value}", json=payload)
    client.post(f"/journey/{BusinessStage.VALIDATION.value}", json=payload)

    response = client.get("/journey/session/demo-session")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "demo-session"
    assert set(data["stages"]) >= {BusinessStage.IDEATION.value, BusinessStage.VALIDATION.value}
    assert data["combined_markdown"].count("##") >= 2
