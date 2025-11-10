"""Heuristic stage generators that mimic the notebook pipeline for the API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import math
import re
from collections import Counter

from .llm import PersonaSnapshot, generate_stage_structured
from .memory import session_memory
from .schemas import (
    BusinessStage,
    FounderClone,
    PersonaDefinition,
    StageDefinition,
    StageRequest,
    StageResponse,
)


# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonaProfile:
    """Describe an available founder clone persona."""

    id: FounderClone
    label: str
    tagline: str
    tone: str
    strategic_bias: str


PERSONAS: Dict[FounderClone, PersonaProfile] = {
    FounderClone.ELON: PersonaProfile(
        id=FounderClone.ELON,
        label="Elon",
        tagline="Moonshot visionary focused on bold leaps.",
        tone="audacious and future-casting",
        strategic_bias="push for ambitious bets and technical differentiation",
    ),
    FounderClone.JEFF: PersonaProfile(
        id=FounderClone.JEFF,
        label="Jeff",
        tagline="Execution obsessed operator with a systems mindset.",
        tone="operationally precise and customer-obsessed",
        strategic_bias="optimize for scalable execution and resource efficiency",
    ),
    FounderClone.NANCY: PersonaProfile(
        id=FounderClone.NANCY,
        label="Nancy",
        tagline="Strategic growth architect who blends market insight with culture.",
        tone="analytical yet empathetic",
        strategic_bias="balance strategic positioning with sustainable growth",
    ),
}

DEFAULT_CLONE = FounderClone.NANCY


def _persona_snapshot(persona: PersonaProfile) -> PersonaSnapshot:
    """Adapt the persona for the LLM module."""

    return PersonaSnapshot(
        id=persona.id.value,
        label=persona.label,
        tagline=persona.tagline,
        tone=persona.tone,
        strategic_bias=persona.strategic_bias,
    )


# ---------------------------------------------------------------------------
# Stage infrastructure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageContext:
    """Bundle together request data that every generator needs."""

    prompt: str
    persona: PersonaProfile
    keywords: List[str]
    context_markdown: str | None
    context_summary: str | None
    context_structured: Dict[str, object] | None


@dataclass(frozen=True)
class StageArtifacts:
    """Container for structured data and a short summary."""

    structured: Dict[str, object]
    summary: str


GeneratorFn = Callable[[StageContext], StageArtifacts]
FormatterFn = Callable[[Dict[str, object]], str]


@dataclass(frozen=True)
class StageInfo:
    """Runtime definition used by the registry below."""

    slug: BusinessStage
    label: str
    description: str
    generator: GeneratorFn
    formatter: FormatterFn


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "and",
    "the",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "their",
    "about",
    "using",
    "entrepreneur",
    "startup",
    "business",
    "solution",
    "platform",
    "create",
    "building",
    "make",
    "service",
    "help",
    "users",
    "customer",
    "customers",
}


def _structured_to_text(data: Dict[str, object] | None) -> str:
    """Serialize structured context into a keyword-friendly text blob."""

    if not data:
        return ""
    try:
        return json.dumps(data, ensure_ascii=False)
    except TypeError:
        return str(data)


def _extract_keywords(*texts: str, max_terms: int = 5) -> List[str]:
    """Extract the top keywords from the provided text fragments."""

    joined = " ".join(part for part in texts if part)
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", joined.lower())
    counts: Counter[str] = Counter(word for word in words if word not in STOP_WORDS and len(word) > 2)
    most_common = [word for word, _ in counts.most_common(max_terms)]

    if not most_common:
        seed = re.findall(r"[a-zA-Z]+", joined.lower())
        if seed:
            most_common = seed[:max_terms]
    if not most_common:
        most_common = ["innovation", "growth", "launch"]
    return most_common


def _titleize(word: str) -> str:
    """Return a simple title-case transformation."""

    return word.replace("-", " ").title()


def _persona_prefix(persona: PersonaProfile) -> str:
    """Create a sentence prefix aligned with the persona tone."""

    if persona.id is FounderClone.ELON:
        return "Visionary take:"
    if persona.id is FounderClone.JEFF:
        return "Execution lens:"
    return "Strategic lens:"


def _scale_currency(value: float) -> str:
    """Format a numeric value into a friendly funding range string."""

    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${int(value)}"


def _project_duration(index: int) -> int:
    """Return an increasing project duration expressed in weeks."""

    return max(2, 4 + index * 2)


def _first_sentence(text: str) -> str:
    """Extract the first sentence-like chunk from text for summary use."""

    snippet = text.strip().split("\n", 1)[0]
    parts = re.split(r"[.!?]", snippet)
    return f"{parts[0].strip()}." if parts and parts[0].strip() else snippet.strip()


# ---------------------------------------------------------------------------
# Stage generators
# ---------------------------------------------------------------------------


def _generate_ideation(context: StageContext) -> StageArtifacts:
    persona = context.persona
    keywords = context.keywords

    summary = (
        f"{persona.tone.title()} concept that centers on {', '.join(_titleize(k) for k in keywords[:2])} "
        f"while leaning on {persona.strategic_bias}."
    )

    target_audience = [
        f"{_titleize(keyword)} early adopters" for keyword in keywords[:3]
    ]
    if len(target_audience) < 3:
        target_audience.append("Adjacent communities seeking practical innovation")

    key_features = [
        f"{_titleize(keyword)}-driven offering that advances the core experience."
        for keyword in keywords[:3]
    ]
    key_features.append(f"{persona.label}'s signature: {persona.tagline.lower()}")

    potential_challenges = [
        f"Securing credible evidence that {keyword} demand exists beyond pilot groups."
        for keyword in keywords[:2]
    ]
    potential_challenges.append(
        f"Maintaining focus as {persona.label.lower()} pushes for {persona.tone} improvements."
    )

    structured = {
        "summary": f"{_persona_prefix(persona)} {summary}",
        "problem_statement": (
            f"Builders navigating {_titleize(keywords[0])} lack integrated support tailored to {persona.strategic_bias}."
        ),
        "target_audience": target_audience,
        "unique_value_proposition": (
            f"Positions the product as the {persona.tone} choice that blends {_titleize(keywords[0])} "
            f"with {_titleize(keywords[1]) if len(keywords) > 1 else 'focused execution'}."
        ),
        "key_features": key_features,
        "potential_challenges": potential_challenges,
        "opportunity_analysis": (
            f"{persona.label} highlights a momentum window as {_titleize(keywords[0])} gains mainstream acceptance. "
            f"Capturing the moment requires aligning culture, product velocity, and differentiated storytelling."
        ),
    }
    return StageArtifacts(structured=structured, summary=structured["summary"])


def _generate_validation(context: StageContext) -> StageArtifacts:
    persona = context.persona
    keywords = context.keywords
    first_keyword = keywords[0]
    second_keyword = keywords[1] if len(keywords) > 1 else "differentiation"

    word_count = len(re.findall(r"[a-zA-Z]+", context.prompt))
    base_tam = 120_000_000 + word_count * 6_500_000
    base_sam = base_tam * 0.25
    base_som = base_tam * 0.08
    viability_score = min(9, max(5, int(math.ceil(word_count / 12)) + 5))

    competitor_analysis = []
    for index, keyword in enumerate(keywords[:3]):
        competitor_analysis.append(
            {
                "name": f"{_titleize(keyword)} Labs",
                "strengths": f"Established footprint in {_titleize(keyword)} workflows with reliable delivery.",
                "weaknesses": f"Limited focus on {persona.strategic_bias.split()[0]} and customer intimacy.",
            }
        )
    if not competitor_analysis:
        competitor_analysis.append(
            {
                "name": "Status Quo Provider",
                "strengths": "Incumbent trust and distribution.",
                "weaknesses": "Slow innovation cadence and rigid pricing.",
            }
        )

    structured = {
        "market_size_estimate": (
            f"TAM for {_titleize(first_keyword)}-centric solutions approaches {_scale_currency(base_tam)} "
            f"with SAM near {_scale_currency(base_sam)} and SOM around {_scale_currency(base_som)} in early beachheads."
        ),
        "competitor_analysis": competitor_analysis,
        "target_market_insight": (
            f"High-intent segments reveal a pain gap around {_titleize(second_keyword)}. "
            f"{persona.label} recommends leading with trust signals and measurable wins."
        ),
        "viability_rating": f"{viability_score} / 10 — {persona.tone} outlook given emerging demand.",
        "pivot_or_focus_recommendation": (
            f"Double down on {_titleize(second_keyword)} messaging while prototyping a premium layer tied to "
            f"{persona.strategic_bias}."
        ),
        "growth_opportunity": (
            f"Adjacent verticals (e.g. {_titleize(keywords[-1]) if keywords else 'enterprise workflows'}) "
            f"show willingness to adopt if onboarding remains low friction."
        ),
        "risks_and_mitigations": [
            f"Adoption drag if {_titleize(first_keyword)} energy fizzles — mitigate with lighthouse customers and public ROI proofs.",
            f"Pricing pressure from incumbents — anchor value on {persona.strategic_bias} outcomes rather than feature parity.",
        ],
    }

    summary = (
        f"{persona.label} pegs {_titleize(first_keyword)} demand as sizable with viability at {structured['viability_rating']}."
    )
    return StageArtifacts(structured=structured, summary=summary)


def _generate_planning(context: StageContext) -> StageArtifacts:
    persona = context.persona
    keywords = context.keywords

    mvp_scope = [
        f"Pilot the {_titleize(keywords[0])} experience with a tightly scoped cohort."
    ]
    if len(keywords) > 1:
        mvp_scope.append(f"Instrument {_titleize(keywords[1])} signals to showcase early proof.")

    product_roadmap = []
    for index, keyword in enumerate(keywords[:3]):
        product_roadmap.append(
            {
                "phase": f"Phase {index + 1}: {_titleize(keyword)} focus",
                "objectives": [
                    f"Ship a lovable slice that demonstrates {_titleize(keyword)} impact.",
                    f"Capture feedback loops that inform {persona.strategic_bias}.",
                ],
                "expected_duration_weeks": _project_duration(index),
            }
        )

    team_requirements = [
        {"role": "Product Lead", "responsibility": f"Translate {persona.strategic_bias} into learnable experiments."},
        {"role": "Builder/Engineer", "responsibility": f"Deliver velocity on {_titleize(keywords[0])} experiences."},
        {"role": "Customer Strategist", "responsibility": "Own user discovery and convert insights into roadmaps."},
    ]

    resource_estimates = {
        "estimated_budget_usd": f"{_scale_currency(65_000 + len(keywords) * 12_000)} over first two quarters.",
        "tools_and_stack": [
            "Collaboration suite (Notion or Linear) for backlog clarity.",
            "Experiment tracking (Airtable, GSheets) to visualise learning cadence.",
            "Analytics instrumentation (Amplitude, PostHog) for behaviour signals.",
        ],
    }

    structured = {
        "mvp_scope": mvp_scope,
        "product_roadmap": product_roadmap,
        "team_requirements": team_requirements,
        "resource_estimates": resource_estimates,
        "documentation_to_prepare": [
            "Discovery log that captures hypotheses, experiments, and decisions.",
            "Customer storyboards for onboarding and first-win experience.",
            "Risk register aligned with the most sensitive assumptions.",
        ],
        "key_risks_and_dependencies": [
            f"Dependency on {_titleize(keywords[0])} data quality — mitigate with manual assurance during pilot.",
            "Capacity risk if execution velocity stalls — reserve buffer sprint for hardening work.",
        ],
    }

    summary = (
        f"{persona.label} frames a staged delivery plan anchored in {_titleize(keywords[0])} with clear owners and guardrails."
    )
    return StageArtifacts(structured=structured, summary=summary)


def _generate_strategy(context: StageContext) -> StageArtifacts:
    persona = context.persona
    keywords = context.keywords

    focus_keyword = _titleize(keywords[0])
    aux_keyword = _titleize(keywords[1]) if len(keywords) > 1 else "differentiation"

    business_plan = {
        "executive_summary": (
            f"{persona.label} frames {focus_keyword} as the wedge to build a durable company that compounds learning "
            f"from early adopters into a category-defining platform."
        ),
        "problem_and_solution": (
            f"Founders struggle to activate {focus_keyword} workflows with the right mix of tooling and judgement. "
            f"The solution blends {persona.strategic_bias} with hands-on enablement."
        ),
        "market_analysis": (
            f"Signals show {focus_keyword} demand growing across adjacent industries. {_titleize(aux_keyword)} "
            "remains under-served, creating a window for a focused entrant."
        ),
        "product_and_services": (
            f"Core platform with playbooks, automation, and services that embed best practices for {focus_keyword}."
        ),
        "operations_plan": (
            "Operate as a lean pod: product, go-to-market, and customer strategy collaborate around a shared scorecard."
        ),
        "team_and_roles": (
            "Founding trio covering product, technical execution, and customer development with fractional specialists."
        ),
        "marketing_strategy": (
            f"Anchor storytelling around {focus_keyword} wins, leaning on persona-led narratives and partner validation."
        ),
        "growth_opportunity": (
            f"Expand into neighbouring segments once {focus_keyword} category leadership is established."
        ),
    }

    financial_model = {
        "revenue_streams": [
            "Subscription platform access",
            "Advisory sprints for complex deployments",
            "Performance-based success fees with pilot customers",
        ],
        "pricing_strategy": "Tiered model with proof-of-value pilot leading into annual contracts.",
        "cost_structure": [
            "Product and engineering",
            "GTM and enablement",
            "Customer success and community programs",
        ],
        "revenue_projection_usd": {
            "year_1": 500_000,
            "year_2": 1_400_000,
            "year_3": 3_000_000,
        },
        "profitability_forecast": "Breakeven near end of Year 3 with disciplined CAC payback targets.",
        "funding_needed_usd": "$1.2M seed to fund 18 months of runway.",
    }

    investor_readiness = {
        "funding_round_type": "Pre-seed / Seed",
        "ideal_investor_profile": (
            "Operator-angel or early-stage fund with deep conviction in future-of-work and workflow tooling."
        ),
        "funding_use_plan": [
            "Accelerate product roadmap tied to {focus_keyword} differentiation".format(focus_keyword=focus_keyword),
            "Build customer experience pods for lighthouse accounts",
            "Invest in measurement instrumentation for ROI storytelling",
        ],
        "pitch_focus_points": [
            f"Category insight into {focus_keyword}",
            f"Unique persona-driven approach championed by {persona.label}",
            "Evidence of rapid learning loops and engaged design partners",
        ],
        "risk_analysis": [
            "Time-to-value risk if onboarding is complex",
            "Talent risk if specialist roles are hard to hire",
        ],
        "overall_readiness_score": "7 / 10 — compelling wedge with clear plan to validate capital efficiency.",
    }

    structured = {
        "business_plan": business_plan,
        "financial_model": financial_model,
        "investor_readiness": investor_readiness,
    }

    summary = business_plan["executive_summary"]
    return StageArtifacts(structured=structured, summary=summary)


def _generate_launch(context: StageContext) -> StageArtifacts:
    persona = context.persona
    keywords = context.keywords

    launch_milestones = [
        {"title": "Private beta", "owner": "Product + Customer", "timeline": "Weeks 1-4"},
        {"title": "Public story", "owner": "Marketing", "timeline": "Week 5"},
        {"title": "Full launch", "owner": "Go-To-Market", "timeline": "Week 8"},
    ]

    structured = {
        "launch_objective": (
            f"Orchestrate a launch that proves {_titleize(keywords[0])} delivers unmistakable value within 60 days."
        ),
        "launch_milestones": launch_milestones,
        "marketing_channels": [
            f"Hero case study featuring {_titleize(keywords[0])} results.",
            "Live demo roadshow coupled with hands-on onboarding clinics.",
            "Founder-led narrative across owned channels to humanise the mission.",
        ],
        "enablement_plan": [
            "Sales brief with objection handling anchored in data.",
            "Customer success playbooks for onboarding and retention nudges.",
            "Support macros anticipating top 10 launch questions.",
        ],
        "success_metrics": [
            "Qualified waitlist to customer conversion above 35%.",
            "Launch cohort expansion rate > 25% within first 90 days.",
            f"Press or influencer mentions highlighting {_titleize(keywords[1] if len(keywords) > 1 else keywords[0])}.",
        ],
        "post_launch_motion": (
            f"Institute a weekly launch retrospective to fold learning into roadmap and to defend {persona.strategic_bias}."
        ),
    }

    summary = f"{persona.label} choreographs a disciplined launch emphasising {_titleize(keywords[0])} proof points."
    return StageArtifacts(structured=structured, summary=summary)


def _generate_scale(context: StageContext) -> StageArtifacts:
    persona = context.persona
    keywords = context.keywords

    structured = {
        "scale_pillars": [
            f"Instrumented growth loops around {_titleize(keywords[0])} usage.",
            "Operational excellence through automation and shared dashboards.",
            f"Culture of experimentation led by {persona.tone} storytelling.",
        ],
        "organizational_design": (
            "Stand up a growth pod (product, data, lifecycle) that owns experimentation backlog. "
            "Pair with customer advocacy squad to keep proximity to signal."
        ),
        "automation_targets": [
            "Lifecycle messaging triggered by behavioural cohorts.",
            "Revenue forecasting pipeline tied to product usage signals.",
            "Self-serve enablement library with dynamic updates.",
        ],
        "expansion_pathways": (
            f"Sequence expansion: deepen {persona.strategic_bias.split()[0]} in core vertical, "
            f"then expand toward {_titleize(keywords[-1]) if keywords else 'adjacent markets'} with localized narratives."
        ),
        "funding_strategy": (
            f"Qualify capital partners that resonate with {persona.tagline.lower()} — focus on smart capital "
            f"that accelerates {_titleize(keywords[0])} outcomes."
        ),
        "risk_watchlist": [
            "Team burnout risk as scope widens — institutionalise recharge rituals.",
            "Signal decay if experimentation cadence slows — enforce guardrail metrics.",
        ],
    }

    summary = f"{persona.label} frames scale as disciplined growth loops anchored in {_titleize(keywords[0])}."
    return StageArtifacts(structured=structured, summary=summary)


# ---------------------------------------------------------------------------
# Markdown formatters
# ---------------------------------------------------------------------------


def _bullet_list(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items if item)


def _format_ideation_markdown(data: Dict[str, object]) -> str:
    return "\n\n".join(
        section
        for section in [
            f"## Summary\n\n{data.get('summary', '')}",
            f"## Problem Statement\n\n{data.get('problem_statement', '')}" if data.get("problem_statement") else "",
            f"## Target Audience\n\n{_bullet_list(data.get('target_audience', []))}" if data.get("target_audience") else "",
            f"## Unique Value Proposition\n\n{data.get('unique_value_proposition', '')}"
            if data.get("unique_value_proposition")
            else "",
            f"## Key Features\n\n{_bullet_list(data.get('key_features', []))}" if data.get("key_features") else "",
            f"## Potential Challenges\n\n{_bullet_list(data.get('potential_challenges', []))}"
            if data.get("potential_challenges")
            else "",
            f"## Opportunity Analysis\n\n{data.get('opportunity_analysis', '')}"
            if data.get("opportunity_analysis")
            else "",
        ]
        if section
    )


def _format_validation_markdown(data: Dict[str, object]) -> str:
    competitor_lines = []
    for entry in data.get("competitor_analysis", []):
        name = entry.get("name", "Competitor")
        strengths = entry.get("strengths", "")
        weaknesses = entry.get("weaknesses", "")
        competitor_lines.append(f"- **{name}** — strengths: {strengths}; weaknesses: {weaknesses}")

    return "\n\n".join(
        section
        for section in [
            f"## Market Size Estimate\n\n{data.get('market_size_estimate', '')}"
            if data.get("market_size_estimate")
            else "",
            f"## Competitor Analysis\n\n{_bullet_list(competitor_lines)}" if competitor_lines else "",
            f"## Target Market Insight\n\n{data.get('target_market_insight', '')}"
            if data.get("target_market_insight")
            else "",
            f"## Viability Rating\n\n{data.get('viability_rating', '')}" if data.get("viability_rating") else "",
            f"## Focus Recommendation\n\n{data.get('pivot_or_focus_recommendation', '')}"
            if data.get("pivot_or_focus_recommendation")
            else "",
            f"## Growth Opportunity\n\n{data.get('growth_opportunity', '')}" if data.get("growth_opportunity") else "",
            f"## Risks & Mitigations\n\n{_bullet_list(data.get('risks_and_mitigations', []))}"
            if data.get("risks_and_mitigations")
            else "",
        ]
        if section
    )


def _format_planning_markdown(data: Dict[str, object]) -> str:
    roadmap_sections = []
    for phase in data.get("product_roadmap", []):
        lines = [f"**{phase.get('phase', 'Phase')}**"]
        if phase.get("expected_duration_weeks"):
            lines.append(f"*Duration:* {phase['expected_duration_weeks']} weeks")
        if phase.get("objectives"):
            lines.append(f"*Objectives:*\n{_bullet_list(phase['objectives'])}")
        roadmap_sections.append("\n".join(lines))

    team_lines = [
        f"- **{member.get('role', 'Role')}** — {member.get('responsibility', 'Responsibility')}"
        for member in data.get("team_requirements", [])
    ]

    resources = data.get("resource_estimates", {})
    tools_block = _bullet_list(resources.get("tools_and_stack", [])) if resources.get("tools_and_stack") else ""

    return "\n\n".join(
        section
        for section in [
            f"## MVP Scope\n\n{_bullet_list(data.get('mvp_scope', []))}" if data.get("mvp_scope") else "",
            f"## Product Roadmap\n\n{'\n\n'.join(roadmap_sections)}" if roadmap_sections else "",
            f"## Team Requirements\n\n{_bullet_list(team_lines)}" if team_lines else "",
            f"## Resource Estimates\n\n{resources.get('estimated_budget_usd', '')}" if resources.get("estimated_budget_usd") else "",
            f"### Tools & Stack\n\n{tools_block}" if tools_block else "",
            f"## Documentation To Prepare\n\n{_bullet_list(data.get('documentation_to_prepare', []))}"
            if data.get("documentation_to_prepare")
            else "",
            f"## Key Risks & Dependencies\n\n{_bullet_list(data.get('key_risks_and_dependencies', []))}"
            if data.get("key_risks_and_dependencies")
            else "",
        ]
        if section
    )


def _format_business_plan_markdown(plan: Dict[str, object]) -> str:
    if not isinstance(plan, dict):
        return ""
    return "\n\n".join(
        section
        for section in [
            f"## Executive Summary\n\n{plan.get('executive_summary', '')}" if plan.get("executive_summary") else "",
            f"## Problem & Solution\n\n{plan.get('problem_and_solution', '')}" if plan.get("problem_and_solution") else "",
            f"## Market Analysis\n\n{plan.get('market_analysis', '')}" if plan.get("market_analysis") else "",
            f"## Product & Services\n\n{plan.get('product_and_services', '')}" if plan.get("product_and_services") else "",
            f"## Operations Plan\n\n{plan.get('operations_plan', '')}" if plan.get("operations_plan") else "",
            f"## Team & Roles\n\n{plan.get('team_and_roles', '')}" if plan.get("team_and_roles") else "",
            f"## Marketing Strategy\n\n{plan.get('marketing_strategy', '')}" if plan.get("marketing_strategy") else "",
            f"## Growth Opportunity\n\n{plan.get('growth_opportunity', '')}" if plan.get("growth_opportunity") else "",
        ]
        if section
    )


def _format_financial_model_markdown(model: Dict[str, object]) -> str:
    if not isinstance(model, dict):
        return ""

    projection = model.get("revenue_projection_usd", {})
    projection_lines = [
        f"- Year {key.split('_')[-1]}: {value}"
        for key, value in projection.items()
    ]

    return "\n\n".join(
        section
        for section in [
            f"## Revenue Streams\n\n{_bullet_list(model.get('revenue_streams', []))}" if model.get("revenue_streams") else "",
            f"## Pricing Strategy\n\n{model.get('pricing_strategy', '')}" if model.get("pricing_strategy") else "",
            f"## Cost Structure\n\n{_bullet_list(model.get('cost_structure', []))}" if model.get("cost_structure") else "",
            f"## Revenue Projection (USD)\n\n{_bullet_list(projection_lines)}" if projection_lines else "",
            f"## Profitability Forecast\n\n{model.get('profitability_forecast', '')}" if model.get("profitability_forecast") else "",
            f"## Funding Needed (USD)\n\n{model.get('funding_needed_usd', '')}" if model.get("funding_needed_usd") else "",
        ]
        if section
    )


def _format_investor_readiness_markdown(readiness: Dict[str, object]) -> str:
    if not isinstance(readiness, dict):
        return ""
    return "\n\n".join(
        section
        for section in [
            f"## Funding Round Type\n\n{readiness.get('funding_round_type', '')}" if readiness.get("funding_round_type") else "",
            f"## Ideal Investor Profile\n\n{readiness.get('ideal_investor_profile', '')}" if readiness.get("ideal_investor_profile") else "",
            f"## Funding Use Plan\n\n{_bullet_list(readiness.get('funding_use_plan', []))}" if readiness.get("funding_use_plan") else "",
            f"## Pitch Focus Points\n\n{_bullet_list(readiness.get('pitch_focus_points', []))}" if readiness.get("pitch_focus_points") else "",
            f"## Risk Analysis\n\n{_bullet_list(readiness.get('risk_analysis', []))}" if readiness.get("risk_analysis") else "",
            f"## Overall Readiness Score\n\n{readiness.get('overall_readiness_score', '')}" if readiness.get("overall_readiness_score") else "",
        ]
        if section
    )


def _format_strategy_markdown(data: Dict[str, object]) -> str:
    return "\n\n".join(
        section
        for section in [
            _format_business_plan_markdown(data.get("business_plan", {})),
            _format_financial_model_markdown(data.get("financial_model", {})),
            _format_investor_readiness_markdown(data.get("investor_readiness", {})),
        ]
        if section
    )


def _format_launch_markdown(data: Dict[str, object]) -> str:
    milestone_lines = [
        f"- **{milestone.get('title', 'Milestone')}** — owner: {milestone.get('owner', 'TBD')}; timeline: {milestone.get('timeline', 'TBD')}"
        for milestone in data.get("launch_milestones", [])
    ]

    return "\n\n".join(
        section
        for section in [
            f"## Launch Objective\n\n{data.get('launch_objective', '')}" if data.get("launch_objective") else "",
            f"## Launch Milestones\n\n{_bullet_list(milestone_lines)}" if milestone_lines else "",
            f"## Marketing Channels\n\n{_bullet_list(data.get('marketing_channels', []))}"
            if data.get("marketing_channels")
            else "",
            f"## Enablement Plan\n\n{_bullet_list(data.get('enablement_plan', []))}" if data.get("enablement_plan") else "",
            f"## Success Metrics\n\n{_bullet_list(data.get('success_metrics', []))}"
            if data.get("success_metrics")
            else "",
            f"## Post-Launch Motion\n\n{data.get('post_launch_motion', '')}" if data.get("post_launch_motion") else "",
        ]
        if section
    )


def _format_scale_markdown(data: Dict[str, object]) -> str:
    return "\n\n".join(
        section
        for section in [
            f"## Scale Pillars\n\n{_bullet_list(data.get('scale_pillars', []))}"
            if data.get("scale_pillars")
            else "",
            f"## Organizational Design\n\n{data.get('organizational_design', '')}"
            if data.get("organizational_design")
            else "",
            f"## Automation Targets\n\n{_bullet_list(data.get('automation_targets', []))}"
            if data.get("automation_targets")
            else "",
            f"## Expansion Pathways\n\n{data.get('expansion_pathways', '')}" if data.get("expansion_pathways") else "",
            f"## Funding Strategy\n\n{data.get('funding_strategy', '')}" if data.get("funding_strategy") else "",
            f"## Risk Watchlist\n\n{_bullet_list(data.get('risk_watchlist', []))}"
            if data.get("risk_watchlist")
            else "",
        ]
        if section
    )


# ---------------------------------------------------------------------------
# Stage registry and orchestration
# ---------------------------------------------------------------------------


STAGE_REGISTRY: Dict[BusinessStage, StageInfo] = {
    BusinessStage.IDEATION: StageInfo(
        slug=BusinessStage.IDEATION,
        label="Ideation",
        description="Explore the core concept, value proposition, and audience.",
        generator=_generate_ideation,
        formatter=_format_ideation_markdown,
    ),
    BusinessStage.VALIDATION: StageInfo(
        slug=BusinessStage.VALIDATION,
        label="Validation",
        description="Assess market size, competitive dynamics, and viability.",
        generator=_generate_validation,
        formatter=_format_validation_markdown,
    ),
    BusinessStage.PLANNING: StageInfo(
        slug=BusinessStage.PLANNING,
        label="Planning",
        description="Translate insights into a phased delivery and resource plan.",
        generator=_generate_planning,
        formatter=_format_planning_markdown,
    ),
    BusinessStage.STRATEGY: StageInfo(
        slug=BusinessStage.STRATEGY,
        label="Strategy",
        description="Frame strategic pillars, positioning, and metrics.",
        generator=_generate_strategy,
        formatter=_format_strategy_markdown,
    ),
    BusinessStage.LAUNCH: StageInfo(
        slug=BusinessStage.LAUNCH,
        label="Launch",
        description="Coordinate launch motions, channels, and success tracking.",
        generator=_generate_launch,
        formatter=_format_launch_markdown,
    ),
    BusinessStage.SCALE: StageInfo(
        slug=BusinessStage.SCALE,
        label="Scale",
        description="Design post-launch growth loops and readiness plans.",
        generator=_generate_scale,
        formatter=_format_scale_markdown,
    ),
}


def list_stage_definitions() -> List[StageDefinition]:
    """Return UI-friendly descriptors for all stages."""

    return [
        StageDefinition(id=info.slug, label=info.label, description=info.description)
        for info in sorted(STAGE_REGISTRY.values(), key=lambda item: item.slug.order)
    ]


def list_personas() -> List[PersonaDefinition]:
    """Return persona definitions suitable for the UI."""

    return [
        PersonaDefinition(
            id=profile.id,
            label=profile.label,
            tagline=profile.tagline,
            tone=profile.tone,
        )
        for profile in PERSONAS.values()
    ]


def run_full_journey(prompt: str, clone: FounderClone | None, session_id: str | None = None) -> Tuple[Dict[str, StageResponse], str]:
    """Execute every stage in order using the provided seed prompt and clone."""

    ordered_infos = sorted(STAGE_REGISTRY.values(), key=lambda info: info.slug.order)
    context_markdown: str | None = None
    context_summary: str | None = None
    context_structured: Dict[str, object] | None = None
    responses: Dict[str, StageResponse] = {}

    resolved_clone = clone or DEFAULT_CLONE

    for info in ordered_infos:
        stage_request = StageRequest(
            prompt=prompt,
            clone=resolved_clone,
            session_id=session_id,
            context_markdown=context_markdown,
            context_summary=context_summary,
            context_structured=context_structured,
        )
        stage_response = generate_stage(info.slug, stage_request)
        responses[info.slug.value] = stage_response

        if session_id:
            session_memory.upsert_stage(session_id, info.slug.value, stage_response)

        context_markdown = stage_response.markdown
        context_summary = stage_response.summary
        context_structured = stage_response.structured

    combined = "\n\n---\n\n".join(
        response.markdown for response in responses.values() if response.markdown
    )
    return responses, combined


def generate_stage(stage: BusinessStage, payload: StageRequest) -> StageResponse:
    """Execute the configured generator and formatter for the requested stage."""

    clone_key = payload.clone or DEFAULT_CLONE
    persona = PERSONAS[clone_key]
    stage_info = STAGE_REGISTRY[stage]
    keywords = _extract_keywords(
        payload.prompt,
        payload.context_markdown or "",
        payload.context_summary or "",
        _structured_to_text(payload.context_structured),
    )
    context = StageContext(
        prompt=payload.prompt,
        persona=persona,
        keywords=keywords,
        context_markdown=payload.context_markdown,
        context_summary=payload.context_summary,
        context_structured=payload.context_structured,
    )

    persona_selected = payload.clone is not None
    llm_structured = generate_stage_structured(
        stage_info.slug,
        _persona_snapshot(persona),
        payload,
        persona_selected=persona_selected,
    )
    if llm_structured:
        artifacts = StageArtifacts(structured=llm_structured, summary=str(llm_structured.get("summary", "")))
    else:
        artifacts = stage_info.generator(context)
    markdown = stage_info.formatter(artifacts.structured)

    if not artifacts.summary:
        artifacts_summary = _first_sentence(markdown)
    else:
        artifacts_summary = _first_sentence(artifacts.summary)

    return StageResponse(
        stage=stage_info.slug,
        stage_label=stage_info.label,
        clone=persona.id,
        summary=artifacts_summary,
        markdown=markdown.strip(),
        structured=artifacts.structured,
    )
