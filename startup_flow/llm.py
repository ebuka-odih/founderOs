"""OpenAI-powered generators that mirror the notebook agent flow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Dict

from openai import APIError, OpenAI

from .config import get_llm_settings
from .schemas import BusinessStage, StageRequest


@dataclass(frozen=True)
class PersonaSnapshot:
    """Serializable subset of persona metadata used by prompt builders."""

    id: str
    label: str
    tagline: str
    tone: str
    strategic_bias: str


@dataclass(frozen=True)
class PromptSpec:
    """Container describing how to call the LLM for a stage."""

    system_prompt: str
    user_prompt: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 900


ClientCache = tuple[str, OpenAI]
_client_cache: ClientCache | None = None


def _get_client() -> OpenAI | None:
    """Return a cached OpenAI client when an API key is configured."""

    global _client_cache
    settings = get_llm_settings()
    api_key = settings.openai_api_key
    if not api_key:
        return None
    if _client_cache and _client_cache[0] == api_key:
        return _client_cache[1]
    client = OpenAI(api_key=api_key)
    _client_cache = (api_key, client)
    return client


def _parse_structured_response(raw_text: str) -> Dict[str, Any] | None:
    """Attempt to coerce the model output into JSON."""

    text = raw_text.strip()
    if text.startswith("```"):
        lines = [line.rstrip() for line in text.splitlines()]
        if len(lines) >= 2:
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _persona_hint(persona: PersonaSnapshot, enabled: bool) -> str:
    if not enabled:
        return ""
    return (
        f"\nAdopt the mindset of {persona.label} — {persona.tagline}. "
        f"Bias toward {persona.strategic_bias} and keep the tone {persona.tone}."
    )


def _pretty_json(data: Any, fallback: str) -> str:
    if not data:
        return fallback
    try:
        return json.dumps(data, indent=2)
    except TypeError:
        return json.dumps(json.loads(json.dumps(data, default=str)), indent=2)


def _invoke(client: OpenAI, spec: PromptSpec) -> Dict[str, Any] | None:
    try:
        response = client.chat.completions.create(
            model=spec.model,
            messages=[
                {"role": "system", "content": spec.system_prompt.strip()},
                {"role": "user", "content": spec.user_prompt.strip()},
            ],
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )
    except APIError:
        return None

    message = response.choices[0].message.content if response.choices else None
    if not message:
        return None
    return _parse_structured_response(message)


def _run_ideation(client: OpenAI, persona_hint: str, prompt_text: str) -> Dict[str, Any] | None:
    user_prompt = dedent(
        f"""
        You are an expert product ideation agent.{persona_hint}

        The user has entered an idea: "{prompt_text}"

        Expand this idea into a structured concept with deep insight.
        Output in JSON format with the following keys:
        {{
          "summary": string,
          "problem_statement": string,
          "target_audience": [string],
          "unique_value_proposition": string,
          "key_features": [string],
          "potential_challenges": [string],
          "opportunity_analysis": string
        }}

        Make it thoughtful, well-structured, and realistic.
        """
    )
    spec = PromptSpec(
        system_prompt="You are an expert in startup ideation and product strategy.",
        user_prompt=user_prompt,
        temperature=0.8,
        max_tokens=800,
    )
    return _invoke(client, spec)


def _run_validation(
    client: OpenAI,
    persona_hint: str,
    idea_data: Dict[str, Any],
) -> Dict[str, Any] | None:
    user_prompt = dedent(
        f"""
        You are a startup validation expert.{persona_hint}
        Based on the following idea details, perform market validation:

        {_pretty_json(idea_data, "No idea data provided.")}

        Produce output in JSON with:
        {{
          "market_size_estimate": string,
          "competitor_analysis": [
              {{"name": "string", "strengths": "string", "weaknesses": "string"}}
          ],
          "target_market_insight": string,
          "viability_rating": "1–10 with short reasoning",
          "pivot_or_focus_recommendation": string,
          "growth_opportunity": string,
          "risks_and_mitigations": [string]
        }}
        """
    )
    spec = PromptSpec(
        system_prompt="You are an expert in market research and product validation.",
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=1000,
    )
    return _invoke(client, spec)


def _run_planning(
    client: OpenAI,
    persona_hint: str,
    validation_data: Dict[str, Any],
) -> Dict[str, Any] | None:
    user_prompt = dedent(
        f"""
        You are a startup execution strategist.{persona_hint}
        Based on the validated idea below, produce an actionable delivery plan:

        {_pretty_json(validation_data, "No validation data provided.")}

        Structure the response as JSON with:
        {{
          "mvp_scope": [string],
          "product_roadmap": [
            {{"phase": string, "objectives": [string], "expected_duration_weeks": integer}}
          ],
          "team_requirements": [
            {{"role": string, "responsibility": string}}
          ],
          "resource_estimates": {{
            "estimated_budget_usd": string,
            "tools_and_stack": [string]
          }},
          "documentation_to_prepare": [string],
          "key_risks_and_dependencies": [string]
        }}

        Keep the plan realistic for a lean founding team and call out the biggest execution risks.
        """
    )
    spec = PromptSpec(
        system_prompt="You are an expert startup operator focused on delivery planning.",
        user_prompt=user_prompt,
        temperature=0.6,
        max_tokens=900,
    )
    return _invoke(client, spec)


def _run_business_plan(
    client: OpenAI,
    persona_hint: str,
    planning_data: Dict[str, Any],
) -> Dict[str, Any] | None:
    user_prompt = dedent(
        f"""
        You are a professional startup strategist.{persona_hint}
        Use the following planning output to draft a comprehensive business plan:

        {_pretty_json(planning_data, "No planning data provided.")}

        Return JSON with this structure:
        {{
          "business_plan": {{
            "executive_summary": string,
            "problem_and_solution": string,
            "market_analysis": string,
            "product_and_services": string,
            "operations_plan": string,
            "team_and_roles": string,
            "marketing_strategy": string,
            "growth_opportunity": string
          }}
        }}
        """
    )
    spec = PromptSpec(
        system_prompt="You are an expert business planner.",
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=1200,
    )
    return _invoke(client, spec)


def _run_financial_model(
    client: OpenAI,
    persona_hint: str,
    plan_data: Dict[str, Any],
) -> Dict[str, Any] | None:
    user_prompt = dedent(
        f"""
        Using the business plan below, craft a lightweight 3-year financial model.{persona_hint}

        {_pretty_json(plan_data, "No business plan data provided.")}

        Return JSON structured as:
        {{
          "financial_model": {{
            "revenue_streams": [string],
            "pricing_strategy": string,
            "cost_structure": [string],
            "revenue_projection_usd": {{
              "year_1": number,
              "year_2": number,
              "year_3": number
            }},
            "profitability_forecast": string,
            "funding_needed_usd": string
          }}
        }}
        """
    )
    spec = PromptSpec(
        system_prompt="You are a startup financial analyst and CFO advisor.",
        user_prompt=user_prompt,
        temperature=0.6,
        max_tokens=1000,
    )
    return _invoke(client, spec)


def _run_investor_readiness(
    client: OpenAI,
    persona_hint: str,
    financial_data: Dict[str, Any],
) -> Dict[str, Any] | None:
    user_prompt = dedent(
        f"""
        Using the financial model below, prepare an investor readiness profile.{persona_hint}

        {_pretty_json(financial_data, "No financial data provided.")}

        Return JSON with:
        {{
          "investor_readiness": {{
            "funding_round_type": string,
            "ideal_investor_profile": string,
            "funding_use_plan": [string],
            "pitch_focus_points": [string],
            "risk_analysis": [string],
            "overall_readiness_score": string
          }}
        }}
        """
    )
    spec = PromptSpec(
        system_prompt="You are an investor relations and fundraising strategy expert.",
        user_prompt=user_prompt,
        temperature=0.65,
        max_tokens=1000,
    )
    return _invoke(client, spec)


def _run_launch(
    client: OpenAI,
    persona_hint: str,
    strategy_data: Dict[str, Any],
) -> Dict[str, Any] | None:
    user_prompt = dedent(
        f"""
        You are a go-to-market launch architect.{persona_hint}
        Using the strategic blueprint below, design a launch plan that proves the story in-market.

        {_pretty_json(strategy_data, "No strategy data provided.")}

        Return JSON with:
        {{
          "launch_objective": string,
          "launch_milestones": [{{"title": string, "owner": string, "timeline": string}}],
          "marketing_channels": [string],
          "enablement_plan": [string],
          "success_metrics": [string],
          "post_launch_motion": string
        }}
        """
    )
    spec = PromptSpec(
        system_prompt="You are an expert in orchestrating product launches.",
        user_prompt=user_prompt,
        temperature=0.65,
        max_tokens=900,
    )
    return _invoke(client, spec)


def _run_scale(
    client: OpenAI,
    persona_hint: str,
    launch_data: Dict[str, Any],
) -> Dict[str, Any] | None:
    user_prompt = dedent(
        f"""
        You are a scale-stage operator who specialises in post-launch growth.{persona_hint}
        With the launch learnings below, outline how to scale responsibly.

        {_pretty_json(launch_data, "No launch data provided.")}

        Provide JSON:
        {{
          "scale_pillars": [string],
          "organizational_design": string,
          "automation_targets": [string],
          "expansion_pathways": string,
          "funding_strategy": string,
          "risk_watchlist": [string]
        }}
        """
    )
    spec = PromptSpec(
        system_prompt="You are an expert in designing post-launch growth loops.",
        user_prompt=user_prompt,
        temperature=0.6,
        max_tokens=900,
    )
    return _invoke(client, spec)


def generate_stage_structured(
    stage: BusinessStage,
    persona: PersonaSnapshot,
    payload: StageRequest,
    *,
    persona_selected: bool,
) -> Dict[str, Any] | None:
    """Call the LLM for the requested stage and return structured data."""

    client = _get_client()
    if client is None:
        return None

    persona_hint = _persona_hint(persona, persona_selected)
    prompt_text = payload.prompt.strip()

    if stage is BusinessStage.IDEATION:
        return _run_ideation(client, persona_hint, prompt_text)

    if stage is BusinessStage.VALIDATION:
        idea_data = payload.context_structured or _run_ideation(client, persona_hint, prompt_text)
        if not idea_data:
            return None
        return _run_validation(client, persona_hint, idea_data)

    if stage is BusinessStage.PLANNING:
        validation_data = payload.context_structured
        if not validation_data:
            idea_data = _run_ideation(client, persona_hint, prompt_text)
            if not idea_data:
                return None
            validation_data = _run_validation(client, persona_hint, idea_data)
        if not validation_data:
            return None
        return _run_planning(client, persona_hint, validation_data)

    if stage is BusinessStage.STRATEGY:
        planning_data = payload.context_structured
        if not planning_data:
            idea_data = _run_ideation(client, persona_hint, prompt_text)
            if not idea_data:
                return None
            validation_data = _run_validation(client, persona_hint, idea_data)
            if not validation_data:
                return None
            planning_data = _run_planning(client, persona_hint, validation_data)
        if not planning_data:
            return None

        business_plan_payload = _run_business_plan(client, persona_hint, planning_data)
        business_plan = business_plan_payload.get("business_plan") if business_plan_payload else None

        financial_payload = _run_financial_model(client, persona_hint, business_plan or planning_data)
        financial_model = financial_payload.get("financial_model") if financial_payload else None

        investor_payload = _run_investor_readiness(client, persona_hint, financial_model or {})
        investor_profile = investor_payload.get("investor_readiness") if investor_payload else None

        return {
            "business_plan": business_plan or {},
            "financial_model": financial_model or {},
            "investor_readiness": investor_profile or {},
        }

    if stage is BusinessStage.LAUNCH:
        strategy_data = payload.context_structured or {}
        return _run_launch(client, persona_hint, strategy_data)

    if stage is BusinessStage.SCALE:
        launch_data = payload.context_structured or {}
        return _run_scale(client, persona_hint, launch_data)

    return None
