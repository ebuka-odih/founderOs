# ====================================================
# Shared Setup & Utilities
# ====================================================

from pathlib import Path
from typing import Iterable

import json
import re
import os

from dotenv import load_dotenv
try:
    from IPython.display import Markdown, display
except ImportError:
    def Markdown(text: str) -> str:
        return text
    def display(value: str) -> None:
        print(value)
from openai import OpenAI

ENV_PATH = Path('.env')
load_dotenv(dotenv_path=ENV_PATH, override=True)

PLACEHOLDER_KEY = 'YOUR_OPENAI_API_KEY_HERE'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or PLACEHOLDER_KEY

def create_client(api_key: str) -> OpenAI | None:
    """Instantiate an OpenAI client if a real API key is present."""
    return OpenAI(api_key=api_key) if api_key != PLACEHOLDER_KEY else None

client = create_client(OPENAI_API_KEY)

def require_client() -> OpenAI:
    """Return an OpenAI client or raise if no API key is configured."""
    if client is None:
        raise RuntimeError('Set OPENAI_API_KEY in your .env file before running an agent.')
    return client

def call_structured_agent(*, system_prompt: str, user_prompt: str, model: str = 'gpt-4o-mini', temperature: float = 0.8, max_tokens: int = 800) -> dict:
    """Send a chat completion request and parse the structured JSON response."""
    response = require_client().chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw_text = response.choices[0].message.content.strip()
    return parse_structured_response(raw_text)

def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if len(lines) >= 2:
            # drop opening fence (e.g., ```json)
            lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            return '\n'.join(lines).strip()
    return text

def parse_structured_response(text: str) -> dict:
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except Exception:
        return {'raw_text': text}

def write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2))

def read_json(path: Path) -> dict:
    return json.loads(path.read_text())

def bullet_list(items: Iterable[str]) -> str:
    return '\n'.join(f'- {item}' for item in items)

def join_sections(sections: Iterable[str]) -> str:
    return '\n\n'.join(section for section in sections if section)

def display_markdown(markdown: str) -> None:
    display(Markdown(markdown))

def format_viability_rating(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if '/' in text or 'out of' in lowered:
        return text
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return text
    rating = match.group(1)
    tail = text[match.end():].strip(" -:–—")
    rating_str = rating.rstrip("0").rstrip(".") if "." in rating else str(int(float(rating)))
    detail = f" — {tail}" if tail else ""
    return f"{rating_str} / 10 (1 = low confidence, 10 = high confidence){detail}"

IDEA_OUTPUT_PATH = Path('idea_output.json')
VALIDATION_OUTPUT_PATH = Path('validation_output.json')
EXECUTION_OUTPUT_PATH = Path('execution_output.json')
PLANNING_OUTPUT_PATH = Path('planning_output.json')
BUSINESS_STRATEGY_OUTPUT_PATH = Path('business_strategy_output.json')

def format_idea_markdown(data: dict) -> str:
    if 'summary' not in data:
        return f"```json\n{json.dumps(data, indent=2)}\n```"
    sections = [
        f"## Summary\n\n{data.get('summary', '')}",
        f"## Problem Statement\n\n{data.get('problem_statement', '')}" if data.get('problem_statement') else '',
        f"## Target Audience\n\n{bullet_list(data.get('target_audience', []))}" if data.get('target_audience') else '',
        f"## Unique Value Proposition\n\n{data.get('unique_value_proposition', '')}" if data.get('unique_value_proposition') else '',
        f"## Key Features\n\n{bullet_list(data.get('key_features', []))}" if data.get('key_features') else '',
        f"## Potential Challenges\n\n{bullet_list(data.get('potential_challenges', []))}" if data.get('potential_challenges') else '',
        f"## Opportunity Analysis\n\n{data.get('opportunity_analysis', '')}" if data.get('opportunity_analysis') else '',
    ]
    return join_sections(sections)

def format_validation_markdown(data: dict) -> str:
    if 'market_size_estimate' not in data:
        return f"```json\n{json.dumps(data, indent=2)}\n```"
    competitors = data.get('competitor_analysis', [])
    competitor_lines = [
        f"- **{entry.get('name', 'Unknown')}** — strengths: {entry.get('strengths', 'n/a')}; weaknesses: {entry.get('weaknesses', 'n/a')}"
        for entry in competitors
    ]
    risks = bullet_list(data.get('risks_and_mitigations', [])) if data.get('risks_and_mitigations') else ''
    viability_text = format_viability_rating(data.get('viability_rating', ''))
    sections = [
        f"## Market Size Estimate\n\n{data.get('market_size_estimate', '')}",
        f"## Competitor Analysis\n\n{'\n'.join(competitor_lines)}" if competitor_lines else '',
        f"## Target Market Insight\n\n{data.get('target_market_insight', '')}" if data.get('target_market_insight') else '',
        f"## Viability Rating\n\n{viability_text}" if viability_text else '',
        f"## Pivot or Focus Recommendation\n\n{data.get('pivot_or_focus_recommendation', '')}" if data.get('pivot_or_focus_recommendation') else '',
        f"## Growth Opportunity\n\n{data.get('growth_opportunity', '')}" if data.get('growth_opportunity') else '',
        f"## Risks & Mitigations\n\n{risks}" if risks else '',
    ]
    return join_sections(sections)


def format_execution_markdown(data: dict) -> str:
    if 'mvp_scope' not in data:
        return f"```json\n{json.dumps(data, indent=2)}\n```"
    scope = bullet_list(data.get('mvp_scope', [])) if data.get('mvp_scope') else ''
    roadmap_sections = []
    for phase in data.get('product_roadmap', []):
        phase_lines = [f"**{phase.get('phase', 'Phase')}**"]
        if phase.get('expected_duration_weeks'):
            phase_lines.append(f"*Duration:* {phase['expected_duration_weeks']} weeks")
        if phase.get('objectives'):
            phase_lines.append(f"*Objectives:*\n{bullet_list(phase['objectives'])}")
        roadmap_sections.append('\n'.join(phase_lines))
    team_lines = [
        f"- **{member.get('role', 'Role')}** — {member.get('responsibility', 'Responsibility')}"
        for member in data.get('team_requirements', [])
    ]
    resources = data.get('resource_estimates', {})
    budget_line = resources.get('estimated_budget_usd')
    tools = resources.get('tools_and_stack', [])
    tools_block = bullet_list(tools) if tools else ''
    documentation = bullet_list(data.get('documentation_to_prepare', [])) if data.get('documentation_to_prepare') else ''
    risks = bullet_list(data.get('key_risks_and_dependencies', [])) if data.get('key_risks_and_dependencies') else ''
    sections = [
        f"## MVP Scope\n\n{scope}" if scope else '',
        f"## Product Roadmap\n\n{'\n\n'.join(roadmap_sections)}" if roadmap_sections else '',
        f"## Team Requirements\n\n{'\n'.join(team_lines)}" if team_lines else '',
        f"## Resource Estimates\n\n{budget_line}" if budget_line else '',
        f"### Tools & Stack\n\n{tools_block}" if tools_block else '',
        f"## Documentation To Prepare\n\n{documentation}" if documentation else '',
        f"## Key Risks & Dependencies\n\n{risks}" if risks else '',
    ]
    return join_sections(sections)



def format_business_plan_markdown(data: dict) -> str:
    plan = data.get('business_plan', data) if isinstance(data, dict) else data
    if not isinstance(plan, dict) or not plan:
        return f"```json\n{json.dumps(data, indent=2)}\n```"
    sections = [
        f"## Executive Summary\n\n{plan.get('executive_summary', '')}" if plan.get('executive_summary') else '',
        f"## Problem & Solution\n\n{plan.get('problem_and_solution', '')}" if plan.get('problem_and_solution') else '',
        f"## Market Analysis\n\n{plan.get('market_analysis', '')}" if plan.get('market_analysis') else '',
        f"## Product & Services\n\n{plan.get('product_and_services', '')}" if plan.get('product_and_services') else '',
        f"## Operations Plan\n\n{plan.get('operations_plan', '')}" if plan.get('operations_plan') else '',
        f"## Team & Roles\n\n{plan.get('team_and_roles', '')}" if plan.get('team_and_roles') else '',
        f"## Marketing Strategy\n\n{plan.get('marketing_strategy', '')}" if plan.get('marketing_strategy') else '',
        f"## Growth Opportunity\n\n{plan.get('growth_opportunity', '')}" if plan.get('growth_opportunity') else '',
    ]
    return join_sections(sections)


def format_financial_model_markdown(data: dict) -> str:
    model = data.get('financial_model', data) if isinstance(data, dict) else data
    if not isinstance(model, dict) or not model:
        return f"```json\n{json.dumps(data, indent=2)}\n```"
    revenue_projection = model.get('revenue_projection_usd', {})
    projection_lines = [
        f"- Year {key.split('_')[-1]}: {value}"
        for key, value in revenue_projection.items()
    ]
    sections = [
        f"## Revenue Streams\n\n{bullet_list(model.get('revenue_streams', []))}" if model.get('revenue_streams') else '',
        f"## Pricing Strategy\n\n{model.get('pricing_strategy', '')}" if model.get('pricing_strategy') else '',
        f"## Cost Structure\n\n{bullet_list(model.get('cost_structure', []))}" if model.get('cost_structure') else '',
        f"## Revenue Projection (USD)\n\n{bullet_list(projection_lines)}" if projection_lines else '',
        f"## Profitability Forecast\n\n{model.get('profitability_forecast', '')}" if model.get('profitability_forecast') else '',
        f"## Funding Needed (USD)\n\n{model.get('funding_needed_usd', '')}" if model.get('funding_needed_usd') else '',
    ]
    return join_sections(sections)


def format_investor_readiness_markdown(data: dict) -> str:
    readiness = data.get('investor_readiness', data) if isinstance(data, dict) else data
    if not isinstance(readiness, dict) or not readiness:
        return f"```json\n{json.dumps(data, indent=2)}\n```"
    sections = [
        f"## Funding Round Type\n\n{readiness.get('funding_round_type', '')}" if readiness.get('funding_round_type') else '',
        f"## Ideal Investor Profile\n\n{readiness.get('ideal_investor_profile', '')}" if readiness.get('ideal_investor_profile') else '',
        f"## Funding Use Plan\n\n{bullet_list(readiness.get('funding_use_plan', []))}" if readiness.get('funding_use_plan') else '',
        f"## Pitch Focus Points\n\n{bullet_list(readiness.get('pitch_focus_points', []))}" if readiness.get('pitch_focus_points') else '',
        f"## Risk Analysis\n\n{bullet_list(readiness.get('risk_analysis', []))}" if readiness.get('risk_analysis') else '',
        f"## Overall Readiness Score\n\n{readiness.get('overall_readiness_score', '')}" if readiness.get('overall_readiness_score') else '',
    ]
    return join_sections(sections)


def format_business_strategy_markdown(data: dict) -> str:
    sections = [
        format_business_plan_markdown(data.get('business_plan', {})),
        format_financial_model_markdown(data.get('financial_model', {})),
        format_investor_readiness_markdown(data.get('investor_readiness', {})),
    ]
    return join_sections(section for section in sections if section)

def show_warning(message: str) -> None:
    display_markdown(f"⚠️ {message}")

# Lightweight smoke test for formatting helpers so refactors fail fast
_smoke_example = {
    'summary': 'Demo summary',
    'problem_statement': 'Demo problem',
    'target_audience': ['Audience A', 'Audience B'],
}
assert 'Audience A' in format_idea_markdown(_smoke_example)
assert '## Market Size Estimate' in format_validation_markdown({'market_size_estimate': 'large', 'viability_rating': '6'})
assert '## MVP Scope' in format_execution_markdown({'mvp_scope': ['item']})
assert '## Executive Summary' in format_business_plan_markdown({'business_plan': {'executive_summary': 'summary'}})
assert '## Revenue Streams' in format_financial_model_markdown({'financial_model': {'revenue_streams': ['stream']}})
assert '## Funding Round Type' in format_investor_readiness_markdown({'investor_readiness': {'funding_round_type': 'Seed'}})
assert format_business_strategy_markdown({
    'business_plan': {'executive_summary': 'summary'},
    'financial_model': {'revenue_streams': ['stream']},
    'investor_readiness': {'funding_round_type': 'Seed'}
})
del _smoke_example



# ====================================================
# Stage 1 – Idea Agent
# ====================================================

def build_idea_prompt(user_input: str) -> str:
    return f"""
    You are an expert product ideation agent.

    The user has entered an idea: "{user_input}"

    Expand this idea into a structured concept with deep insight.
    Output in JSON format with the following keys:
    {{
      "summary": string,
      "problem_statement": string,
      "target_audience": list of strings,
      "unique_value_proposition": string,
      "key_features": list of strings,
      "potential_challenges": list of strings,
      "opportunity_analysis": string
    }}

    Make it thoughtful, well-structured, and realistic.
    """

def idea_agent(user_input: str, *, model: str = 'gpt-4o-mini', temperature: float = 0.8, max_tokens: int = 800) -> dict:
    return call_structured_agent(
        system_prompt='You are an expert in startup ideation and product strategy.',
        user_prompt=build_idea_prompt(user_input),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

def run_idea_stage(user_input: str) -> dict:
    result = idea_agent(user_input)
    write_json(result, IDEA_OUTPUT_PATH)
    display_markdown(format_idea_markdown(result))
    return result

if client is None:
    show_warning('OpenAI API key missing. Set `OPENAI_API_KEY` in your `.env` file and rerun this cell.')
else:
    IDEA_USER_INPUT = 'fasting app for bodybuilders'  # Adjust this seed idea as needed.
    idea_result = run_idea_stage(IDEA_USER_INPUT)



# ====================================================
# Stage 2 – Validation Agent
# ====================================================

def build_validation_prompt(idea_data: dict) -> str:
    return f"""
    You are a startup validation expert.
    Based on the following idea details, perform market validation:

    {json.dumps(idea_data, indent=2)}

    Produce output in JSON with:
    {{
      "market_size_estimate": "string (describe TAM/SAM/SOM or size)",
      "competitor_analysis": [
          {{
            "name": "string",
            "strengths": "string",
            "weaknesses": "string"
          }}
      ],
      "target_market_insight": "string",
      "viability_rating": "1–10 with short reasoning",
      "pivot_or_focus_recommendation": "string",
      "growth_opportunity": "string",
      "risks_and_mitigations": [
          "risk1: mitigation1",
          "risk2: mitigation2"
      ]
    }}
    """

def validation_agent(idea_data: dict, *, model: str = 'gpt-4o-mini', temperature: float = 0.7, max_tokens: int = 1000) -> dict:
    return call_structured_agent(
        system_prompt='You are an expert in market research and product validation.',
        user_prompt=build_validation_prompt(idea_data),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

def acquire_idea_input() -> dict | None:
    if 'idea_result' in globals():
        return idea_result
    if IDEA_OUTPUT_PATH.exists():
        return read_json(IDEA_OUTPUT_PATH)
    show_warning('Idea stage output not found. Run Stage 1 first to generate `idea_result`.')
    return None

if client is None:
    show_warning('OpenAI API key missing. Set `OPENAI_API_KEY` in your `.env` file and rerun this cell.')
else:
    _idea_input = acquire_idea_input()
    if _idea_input is not None:
        validation_result = validation_agent(_idea_input)
        write_json(validation_result, VALIDATION_OUTPUT_PATH)
        display_markdown(format_validation_markdown(validation_result))
        del _idea_input



# ====================================================
# Stage 3 – Execution Planning Agent
# ====================================================

def build_execution_prompt(validation_data: dict) -> str:
    return f"""
    You are a startup execution strategist.
    Based on the validated idea below, produce an actionable delivery plan:

    {json.dumps(validation_data, indent=2)}

    Structure the response as JSON with the following keys:
    {{
      "mvp_scope": [string],
      "product_roadmap": [
        {{
          "phase": string,
          "objectives": [string],
          "expected_duration_weeks": integer
        }}
      ],
      "team_requirements": [
        {{
          "role": string,
          "responsibility": string
        }}
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

def execution_agent(validation_data: dict, *, model: str = 'gpt-4o-mini', temperature: float = 0.6, max_tokens: int = 900) -> dict:
    return call_structured_agent(
        system_prompt='You are an expert startup operator focused on delivery planning.',
        user_prompt=build_execution_prompt(validation_data),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

def acquire_validation_output() -> dict | None:
    if 'validation_result' in globals():
        return validation_result
    if VALIDATION_OUTPUT_PATH.exists():
        return read_json(VALIDATION_OUTPUT_PATH)
    show_warning('Validation stage output not found. Run Stage 2 to generate `validation_result`.')
    return None

if client is None:
    show_warning('OpenAI API key missing. Set `OPENAI_API_KEY` in your `.env` file and rerun this cell.')
else:
    _validation_input = acquire_validation_output()
    if _validation_input is not None:
        execution_plan = execution_agent(_validation_input)
        write_json(execution_plan, EXECUTION_OUTPUT_PATH)
        display_markdown(format_execution_markdown(execution_plan))
        del _validation_input


# ====================================================
# Stage 4 – Business Strategy Agents
# ====================================================



def build_business_plan_prompt(planning_data: dict) -> str:
    return f"""
    You are a professional startup strategist.
    Use the following planning output to draft a comprehensive business plan:

    {json.dumps(planning_data, indent=2)}

    Return JSON with this structure:
    {{
      \"business_plan\": {{
        \"executive_summary\": string,
        \"problem_and_solution\": string,
        \"market_analysis\": string,
        \"product_and_services\": string,
        \"operations_plan\": string,
        \"team_and_roles\": string,
        \"marketing_strategy\": string,
        \"growth_opportunity\": string
      }}
    }}
    """


def business_plan_agent(planning_data: dict, *, model: str = 'gpt-4o-mini', temperature: float = 0.7, max_tokens: int = 1200) -> dict:
    return call_structured_agent(
        system_prompt='You are an expert business planner.',
        user_prompt=build_business_plan_prompt(planning_data),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_financial_model_prompt(plan_data: dict) -> str:
    return f"""
    Based on the business plan below, craft a lightweight 3-year financial model:

    {json.dumps(plan_data, indent=2)}

    Return JSON structured as:
    {{
      \"financial_model\": {{
        \"revenue_streams\": [string],
        \"pricing_strategy\": string,
        \"cost_structure\": [string],
        \"revenue_projection_usd\": {{
          \"year_1\": number,
          \"year_2\": number,
          \"year_3\": number
        }},
        \"profitability_forecast\": string,
        \"funding_needed_usd\": string
      }}
    }}
    """


def financial_model_agent(plan_data: dict, *, model: str = 'gpt-4o-mini', temperature: float = 0.6, max_tokens: int = 1000) -> dict:
    return call_structured_agent(
        system_prompt='You are a startup financial analyst and CFO advisor.',
        user_prompt=build_financial_model_prompt(plan_data),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_investor_readiness_prompt(financial_data: dict) -> str:
    return f"""
    Using the financial model below, prepare an investor readiness profile:

    {json.dumps(financial_data, indent=2)}

    Return JSON with:
    {{
      \"investor_readiness\": {{
        \"funding_round_type\": string,
        \"ideal_investor_profile\": string,
        \"funding_use_plan\": [string],
        \"pitch_focus_points\": [string],
        \"risk_analysis\": [string],
        \"overall_readiness_score\": string
      }}
    }}
    """


def investor_readiness_agent(financial_data: dict, *, model: str = 'gpt-4o-mini', temperature: float = 0.65, max_tokens: int = 1000) -> dict:
    return call_structured_agent(
        system_prompt='You are an investor relations and fundraising strategy expert.',
        user_prompt=build_investor_readiness_prompt(financial_data),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def acquire_planning_output() -> tuple[dict | None, str | None]:
    if 'planning_result' in globals():
        return planning_result, 'notebook memory'
    if PLANNING_OUTPUT_PATH.exists():
        return read_json(PLANNING_OUTPUT_PATH), 'planning_output.json'
    if EXECUTION_OUTPUT_PATH.exists():
        return read_json(EXECUTION_OUTPUT_PATH), 'execution_output.json'
    show_warning('Planning or execution output not found. Run the planning workflow to generate `planning_result`.')
    return None, None

if client is None:
    show_warning('OpenAI API key missing. Set `OPENAI_API_KEY` in your `.env` file and rerun this cell.')
else:
    _planning_input, _planning_source = acquire_planning_output()
    if _planning_input is not None:
        if _planning_source:
            display_markdown(f"*Using planning data from `{_planning_source}`.*")
        business_plan = business_plan_agent(_planning_input)
        display_markdown(format_business_plan_markdown(business_plan))

        financial_model = financial_model_agent(business_plan)
        display_markdown(format_financial_model_markdown(financial_model))

        investor_readiness = investor_readiness_agent(financial_model)
        display_markdown(format_investor_readiness_markdown(investor_readiness))

        business_strategy = {
            'business_plan': business_plan.get('business_plan', {}),
            'financial_model': financial_model.get('financial_model', {}),
            'investor_readiness': investor_readiness.get('investor_readiness', {}),
        }
        write_json(business_strategy, BUSINESS_STRATEGY_OUTPUT_PATH)
        display_markdown(format_business_strategy_markdown(business_strategy))

        del _planning_input, _planning_source


# ====================================================
# Helper Sanity Tests (offline)
# ====================================================

sample_idea = {
    'summary': 'Test summary',
    'problem_statement': 'Test problem',
    'target_audience': ['A', 'B'],
    'unique_value_proposition': 'UVP',
    'key_features': ['Feat1'],
    'potential_challenges': ['Challenge'],
    'opportunity_analysis': 'Opportunity',
}
assert '## Summary' in format_idea_markdown(sample_idea)

sample_validation = {
    'viability_rating': '8',
    'market_size_estimate': 'Large market',
    'competitor_analysis': [{'name': 'Comp A', 'strengths': 'Strong', 'weaknesses': 'Weak'}],
    'risks_and_mitigations': ['Risk: Mitigation'],
}
assert 'Comp A' in format_validation_markdown(sample_validation)
assert ' / 10' in format_validation_markdown(sample_validation)


sample_execution = {
    'mvp_scope': ['Scope item'],
    'product_roadmap': [{
        'phase': 'Phase 1',
        'objectives': ['Do thing'],
        'expected_duration_weeks': 4,
    }],
    'team_requirements': [{
        'role': 'Engineer',
        'responsibility': 'Build system',
    }],
    'resource_estimates': {
        'estimated_budget_usd': '$10k',
        'tools_and_stack': ['Tool'],
    },
    'documentation_to_prepare': ['Doc'],
    'key_risks_and_dependencies': ['Risk'],
}
assert '## MVP Scope' in format_execution_markdown(sample_execution)
raw = 'not json'
parsed = parse_structured_response(raw)
assert parsed['raw_text'] == raw
wrapped = '```json\n{"key": "value"}\n```'
parsed_wrapped = parse_structured_response(wrapped)
assert parsed_wrapped.get('key') == 'value'
del wrapped, parsed_wrapped

print('Helper tests passed (formatters and parser).')

del sample_idea, sample_validation, sample_execution, raw, parsed
