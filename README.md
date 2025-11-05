# 🧠 Project: FounderOs – AI Startup Validation Platform

## 📘 Overview

**FounderOs** is an AI-driven platform that helps founders validate startup ideas from concept to funding readiness.  
The system breaks down validation into **structured stages**, where each stage processes user input and generates **persona-aware Markdown playbooks** that can be previewed or exported.

Unlike a chat interface, FounderOs uses a **guided, step-based UI** that ensures clarity, structure, and actionable feedback.

Each stage is powered by **AI agents** working behind the scenes — analyzing, researching, evaluating, and synthesizing outputs.

---

## 🎯 Core Objective

To provide an **interactive AI workspace** for startup validation by:
1. Allowing users to input startup ideas and choose a founder clone persona.
2. Running stage-specific heuristics inspired by the original notebook flow.
3. Returning structured Markdown for six stages: Ideation → Validation → Planning → Strategy → Launch → Scale.
4. Persisting results per session so founders can iterate, export, or pass context to the next stage.

---

## ⚙️ System Architecture

### 🧩 Tech Stack

- **Frontend:** Next.js + Tailwind CSS + ShadCN UI
- **Backend:** FastAPI (Python)
- **AI Core:** OpenAI GPT-4/5 or Anthropic Claude via LangChain or custom orchestrator
- **Database:** PostgreSQL (via Prisma or SQLAlchemy)
- **Hosting:** Dockerized microservices on DigitalOcean, AWS, or Vercel

---

## 🧭 Backend Overview (FastAPI)

The backend serves as the **central orchestrator** for the journey flow.  
It manages persona-aware heuristics, session memory, and delivers Markdown payloads the UI can render or forward.

### Responsibilities
- Handle the `/journey/{stage}` API for ideation, validation, planning, strategy, launch, and scale.
- Incorporate founder clone tone to keep outputs on-brand.
- Persist session context so subsequent stages can inherit insights.
- Return Markdown plus structured JSON for easy editing or export.

### Key Endpoints
- `GET /journey/health` – lightweight health check.
- `GET /journey/stages` – exposes stage metadata (labels, descriptions).
- `GET /journey/personas` – lists the available founder clones.
- `POST /journey/{stage}` – generates the requested stage output.
- `GET /journey/session/{session_id}` – returns the stored journey results and combined Markdown export.

---

## 🔐 LLM Credentials

FounderOs reads LLM provider credentials from environment variables so they can be shared across the backend:

- `OPENAI_API_KEY` (primary provider)
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`
- `FOUNDEROS_LLM_<PROVIDER>_API_KEY` for any additional vendors (e.g. `FOUNDEROS_LLM_GROQ_API_KEY`)

The FastAPI app caches these values at startup via `app.state.llm_settings`, making them available to routers, stage generators, or future agent integrations.

---

## 🤖 Stage Generator System (Backend Logic)

Each journey stage is powered by a **deterministic generator** that mirrors the structure of the original notebook.  
Instead of external API calls, the backend uses prompt heuristics, keyword extraction, and persona tone rules to craft:

- Structured JSON (for downstream use)
- Markdown sections (for the UI preview/raw toggle)
- A short summary (for stage cards and context passing)

This keeps responses snappy and makes it easy to plug in real LLM calls later.
