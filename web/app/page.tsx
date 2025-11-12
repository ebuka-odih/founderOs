"use client";

import { useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

type StageSlug = "ideation" | "validation" | "planning" | "strategy" | "launch" | "scale";

type StageConfig = {
  id: StageSlug;
  order: number;
  label: string;
  description: string;
  icon: string;
};

type FounderClone = {
  id: "none" | "elon" | "jeff" | "nancy";
  label: string;
  focus: string;
  description: string;
  emoji: string;
};

type StageResult = {
  stage: StageSlug;
  stage_label: string;
  markdown: string;
  summary: string;
  structured: Record<string, unknown>;
  clone: string;
};

type JourneyRunResponse = {
  session_id: string;
  stages: Record<string, StageResult>;
  combined_markdown: string;
};

type ViewMode = "preview" | "raw";
type PromptHistoryEntry = {
  id: string;
  prompt: string;
  cloneLabel: string;
  timestamp: number;
};

const STAGES: StageConfig[] = [
  {
    id: "ideation",
    order: 1,
    label: "Ideation",
    description: "Shape the concept and capture the value proposition.",
    icon: "💡"
  },
  {
    id: "validation",
    order: 2,
    label: "Validation",
    description: "Assess market size, risks, and viability signals.",
    icon: "🔍"
  },
  {
    id: "planning",
    order: 3,
    label: "Planning",
    description: "Translate insights into a delivery plan and resource map.",
    icon: "🗺️"
  },
  {
    id: "strategy",
    order: 4,
    label: "Strategy",
    description: "Clarify positioning, pillars, and success metrics.",
    icon: "🎯"
  },
  {
    id: "launch",
    order: 5,
    label: "Launch",
    description: "Coordinate channels, enablement, and launch milestones.",
    icon: "🚀"
  },
  {
    id: "scale",
    order: 6,
    label: "Scale",
    description: "Design post-launch growth loops and funding strategy.",
    icon: "📈"
  }
];

const FOUNDER_CLONES: FounderClone[] = [
  {
    id: "none",
    label: "None",
    focus: "Default",
    description: "No specific founder clone selected.",
    emoji: "🤖"
  },
  {
    id: "elon",
    label: "Elon",
    focus: "Vision",
    description: "Moonshot thinker who pushes bold differentiation.",
    emoji: "🚀"
  },
  {
    id: "jeff",
    label: "Jeff",
    focus: "Execution",
    description: "Operational strategist with relentless customer focus.",
    emoji: "🧮"
  },
  {
    id: "nancy",
    label: "Nancy",
    focus: "Strategy",
    description: "Growth architect blending market insight and culture.",
    emoji: "🧭"
  }
];

const DEFAULT_CLONE = FOUNDER_CLONES.find(clone => clone.id === "none")!;

function clsx(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}

function escapeHtml(input: string): string {
  return input.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function applyInlineFormatting(text: string): string {
  // Apply basic markdown emphasis after escaping to guard against HTML injection.
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>");
}

function simpleMarkdownToHtml(markdown: string): string {
  const lines = markdown.split("\n");
  let html = "";
  let inList = false;

  const closeList = () => {
    if (inList) {
      html += "</ul>";
      inList = false;
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    if (!line.trim()) {
      closeList();
      html += "<div class=\"mt-3\"></div>";
      continue;
    }

    if (line.startsWith("### ")) {
      closeList();
      html += `<h3 class="mt-4 text-lg font-semibold text-textPrimary">${applyInlineFormatting(
        line.slice(4).trim()
      )}</h3>`;
      continue;
    }

    if (line.startsWith("## ")) {
      closeList();
      html += `<h2 class="mt-6 text-xl font-semibold text-textPrimary">${applyInlineFormatting(
        line.slice(3).trim()
      )}</h2>`;
      continue;
    }

    if (line.startsWith("- ")) {
      if (!inList) {
        html += '<ul class="mt-3 space-y-2 list-disc pl-6 text-sm text-textMuted">';
        inList = true;
      }
      html += `<li>${applyInlineFormatting(line.slice(2).trim())}</li>`;
      continue;
    }

    if (line.startsWith("*Objectives:*")) {
      closeList();
      html += `<p class="mt-3 text-sm font-semibold text-textPrimary">${applyInlineFormatting(
        line
      )}</p>`;
      continue;
    }

    closeList();
    html += `<p class="mt-3 text-sm leading-relaxed text-textMuted">${applyInlineFormatting(line)}</p>`;
  }

  closeList();
  return html;
}

export default function JourneyWorkspace(): JSX.Element {
  const [sessionId] = useState(() => {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      return crypto.randomUUID();
    }
    return `session-${Date.now()}`;
  });
  const [prompt, setPrompt] = useState("");
  const [selectedClone, setSelectedClone] = useState<FounderClone>(DEFAULT_CLONE);
  const [activeStage, setActiveStage] = useState<StageConfig>(STAGES[0]);
  const [stageOutputs, setStageOutputs] = useState<Record<StageSlug, StageResult>>(() => ({} as Record<StageSlug, StageResult>));
  const [viewMode, setViewMode] = useState<ViewMode>("preview");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasCopied, setHasCopied] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [promptHistory, setPromptHistory] = useState<PromptHistoryEntry[]>([]);

  const canGenerate = prompt.trim().length >= 10 && !loading;
  const generateDisabledReason = useMemo(() => {
    if (loading || canGenerate) {
      return null;
    }
    if (prompt.trim().length < 10) {
      return "Add a bit more detail (10+ characters) to your idea to begin.";
    }
    return null;
  }, [prompt, canGenerate, loading]);
  const currentResult = stageOutputs[activeStage.id];
  const previousStage = useMemo(() => {
    const currentIndex = STAGES.findIndex((stage) => stage.id === activeStage.id);
    if (currentIndex <= 0) {
      return null;
    }
    const prevStage = STAGES[currentIndex - 1];
    const prevResult = stageOutputs[prevStage.id];
    if (!prevResult) {
      return null;
    }
    return { stage: prevStage, result: prevResult };
  }, [activeStage, stageOutputs]);
  const inheritedContext = previousStage
    ? {
        markdown: previousStage.result.markdown,
        summary: previousStage.result.summary,
        originStage: previousStage.stage.id
      }
    : null;
  const originStageLabel = previousStage?.stage.label ?? null;
  const previousStructured = previousStage?.result.structured;
const previewHtml = useMemo(() => {
  if (!currentResult) {
    return "";
  }
  return simpleMarkdownToHtml(currentResult.markdown);
}, [currentResult]);
  const historyItems = promptHistory.slice(0, 10);

  async function handleRunJourney() {
    if (!canGenerate) {
      setError("Provide a more detailed prompt to start the journey.");
      return;
    }
    setLoading(true);
    setError(null);
    setHasCopied(false);

    try {
      const payload: Record<string, unknown> = {
        prompt: prompt.trim(),
        session_id: sessionId
      };
      if (selectedClone.id !== "none") {
        payload.clone = selectedClone.id;
      }

      const response = await fetch(`${API_BASE}/journey/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || "Unable to generate the full journey.");
      }

      const data = (await response.json()) as JourneyRunResponse;
      const normalized = Object.entries(data.stages ?? {}).reduce((acc, [key, value]) => {
        acc[key as StageSlug] = value as StageResult;
        return acc;
      }, {} as Record<StageSlug, StageResult>);

      setStageOutputs(normalized);
      setActiveStage(STAGES[0]);
      setViewMode("preview");
      setPromptHistory((prev) => [
        {
          id: typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `prompt-${Date.now()}`,
          prompt: prompt.trim(),
          cloneLabel: selectedClone.label,
          timestamp: Date.now()
        },
        ...prev
      ]);
    } catch (fetchError) {
      const message =
        fetchError instanceof Error ? fetchError.message : "Something went wrong while generating the journey.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  async function handleRegenerateStage() {
    if (prompt.trim().length < 10) {
      setError("Add more detail to your prompt before regenerating.");
      return;
    }
    setLoading(true);
    setError(null);
    setHasCopied(false);

    try {
      const regenPayload: Record<string, unknown> = {
        prompt: prompt.trim(),
        session_id: sessionId
      };
      const resolvedClone = selectedClone.id !== "none" ? selectedClone.id : currentResult?.clone ?? null;
      if (resolvedClone) {
        regenPayload.clone = resolvedClone;
      }
      if (inheritedContext?.markdown) {
        regenPayload.context_markdown = inheritedContext.markdown;
      }
      if (inheritedContext?.summary) {
        regenPayload.context_summary = inheritedContext.summary;
      }
      if (previousStructured) {
        regenPayload.context_structured = previousStructured;
      }

      const response = await fetch(`${API_BASE}/journey/${activeStage.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(regenPayload)
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `Unable to regenerate the ${activeStage.label} stage.`);
      }

      const data = (await response.json()) as StageResult;
      setStageOutputs((prev) => ({
        ...prev,
        [data.stage as StageSlug]: data
      }));
    } catch (fetchError) {
      const message =
        fetchError instanceof Error ? fetchError.message : "Something went wrong while regenerating this stage.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  async function handleCopyMarkdown() {
    if (!currentResult) {
      return;
    }
    try {
      await navigator.clipboard.writeText(currentResult.markdown);
      setHasCopied(true);
      setTimeout(() => setHasCopied(false), 2000);
    } catch {
      setError("Clipboard copy failed. Try again.");
    }
  }

  async function handleDownloadSession() {
    if (!Object.keys(stageOutputs).length) {
      return;
    }
    setDownloading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/journey/session/${sessionId}`);
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || "Unable to download the session report.");
      }

      const payload = await response.json();
      const combinedMarkdown = (payload?.combined_markdown as string) || Object.values(stageOutputs)
        .map((result) => result.markdown)
        .join("\n\n---\n\n");
      const blob = new Blob([combinedMarkdown], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `founderos-journey-${sessionId}.md`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (downloadError) {
      const message =
        downloadError instanceof Error ? downloadError.message : "Session export failed. Please retry.";
      setError(message);
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="min-h-screen bg-muted">
      <div className="mx-auto max-w-6xl px-4 py-2 lg:px-6 lg:py-4">
        <div className="rounded-2xl border border-borderLight bg-surface p-4 lg:p-6 shadow-sm mb-2 lg:mb-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold uppercase tracking-wide text-textMuted">FounderOs</p>
              <h1 className="mt-1 text-2xl font-semibold text-textPrimary">Startup Journey Lab</h1>
            </div>
            <span className="text-3xl" role="img" aria-label="sparkles">
              ✨
            </span>
          </div>
          <p className="mt-4 text-sm leading-relaxed text-textMuted">
            Work through each stage of the journey with guided outputs. Pass insights forward, compare personas, and
            export everything as Markdown.
          </p>
        </div>

        <section className="rounded-2xl border border-borderLight bg-surface p-4 lg:p-6 shadow-sm mb-2 lg:mb-4">
          <div>
            <p className="text-xs uppercase tracking-wide text-textMuted">Prompt Input</p>
            <h2 className="text-xl font-semibold text-textPrimary">
              Describe your business idea
            </h2>
            <p className="text-sm text-textMuted mt-1">
              Choose a founder clone and share your idea to get AI-powered startup guidance through the full journey.
            </p>
          </div>
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Outline the idea, goals, or constraints you want the founder clone to process."
            className="mt-4 h-32 w-full resize-none rounded-xl border border-borderLight bg-muted px-4 py-3 text-sm text-textPrimary placeholder:text-textMuted focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/30"
          />
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label htmlFor="clone-select" className="block text-sm font-medium text-textPrimary mb-2">
                Select Founder Clone
              </label>
              <select
                id="clone-select"
                value={selectedClone.id}
                onChange={(e) => {
                  const clone = FOUNDER_CLONES.find(c => c.id === e.target.value);
                  if (clone) {
                    setSelectedClone(clone);
                    setError(null);
                  }
                }}
                className="w-full h-9 rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-1 focus:ring-ring"
              >
                {FOUNDER_CLONES.map((clone) => (
                  <option key={clone.id} value={clone.id}>
                    {clone.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="file-upload" className="block text-sm font-medium text-textPrimary mb-2">
                Upload File (Optional)
              </label>
              <input
                id="file-upload"
                type="file"
                accept=".pdf,.doc,.docx,.txt,.md"
                className="w-full text-sm text-textPrimary file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-accent file:text-white hover:file:bg-accent/90"
              />
            </div>
          </div>
          {error ? (
            <div className="mt-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
              {error}
            </div>
          ) : null}
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={handleRunJourney}
              disabled={!canGenerate}
              className={clsx(
                "flex items-center gap-2 rounded-xl bg-accent px-5 py-2 text-sm font-semibold text-white shadow-sm transition",
                !canGenerate && "cursor-not-allowed opacity-60"
              )}
            >
              ⚡ Generate Journey
              {loading ? <span className="text-xs text-accentSoft">working…</span> : null}
            </button>
            <button
              type="button"
              onClick={handleDownloadSession}
              disabled={!Object.keys(stageOutputs).length || downloading}
              className={clsx(
                "rounded-xl border border-borderLight px-4 py-2 text-sm text-textPrimary transition hover:border-accent hover:text-accent",
                (!Object.keys(stageOutputs).length || downloading) && "cursor-not-allowed opacity-60"
              )}
            >
              ⬇️ Export Markdown
              {downloading ? <span className="ml-2 text-xs text-textMuted">preparing…</span> : null}
            </button>
          </div>
          {!canGenerate && generateDisabledReason ? (
            <p className="mt-2 text-xs text-textMuted" aria-live="polite">
              {generateDisabledReason}
            </p>
          ) : null}
        </section>
      </div>

      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-6 lg:gap-10 lg:px-6 lg:py-10 lg:flex-row">
        <aside className="w-full space-y-2 lg:w-72">
          <div className="hidden lg:block rounded-2xl border border-borderLight bg-surface p-6 shadow-sm">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-textMuted">Stage Flow</h2>
            <ul className="mt-4 space-y-4">
              {STAGES.map((stage) => {
                const completed = Boolean(stageOutputs[stage.id]);
                const summary = stageOutputs[stage.id]?.summary;
                return (
                  <li key={stage.id} className="flex items-start gap-3">
                    <span
                      className={clsx(
                        "flex h-9 w-9 items-center justify-center rounded-full border text-sm font-semibold",
                        completed ? "border-accent bg-accent text-white" : "border-borderLight bg-muted text-textMuted"
                      )}
                    >
                      {stage.order}
                    </span>
                    <div>
                      <p className="text-sm font-semibold text-textPrimary">{stage.label}</p>
                      <p className="text-xs text-textMuted">{summary ?? stage.description}</p>
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>

          <div className="hidden lg:block rounded-2xl border border-borderLight bg-surface p-6 shadow-sm">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-textMuted">Idea History</h2>
            {historyItems.length === 0 ? (
              <p className="mt-4 text-xs text-textMuted">Your prompts will appear here after you run the journey.</p>
            ) : (
              <ul className="mt-4 space-y-3">
                {historyItems.map((entry) => (
                  <li key={entry.id} className="rounded-xl border border-borderLight px-3 py-3">
                    <p className="text-xs uppercase tracking-wide text-textMuted">
                      {new Date(entry.timestamp).toLocaleString(undefined, {
                        hour: "2-digit",
                        minute: "2-digit"
                      })}
                      {" · "}
                      {entry.cloneLabel}
                    </p>
                    <p className="mt-1 text-sm text-textPrimary" style={{ display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
                      {entry.prompt}
                    </p>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>

        <main className="flex-1 space-y-2">
          <section className="rounded-2xl border border-borderLight bg-surface p-6 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2">
                {STAGES.map((stage) => {
                  const active = stage.id === activeStage.id;
                  return (
                    <button
                      key={stage.id}
                      type="button"
                      onClick={() => {
                        setActiveStage(stage);
                        setViewMode("preview");
                        setError(null);
                      }}
                      className={clsx(
                        "rounded-full px-4 py-2 text-sm font-medium transition",
                        active ? "bg-accent text-white shadow-sm" : "bg-muted text-textMuted hover:text-textPrimary"
                      )}
                    >
                      {stage.icon} {stage.label}
                    </button>
                  );
                })}
              </div>

              <div className="flex items-center gap-2 rounded-full border border-borderLight bg-muted p-1">
                <button
                  type="button"
                  onClick={() => setViewMode("preview")}
                  className={clsx(
                    "rounded-full px-3 py-1 text-xs font-semibold transition",
                    viewMode === "preview" ? "bg-surface text-textPrimary shadow-sm" : "text-textMuted"
                  )}
                >
                  Preview
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode("raw")}
                  className={clsx(
                    "rounded-full px-3 py-1 text-xs font-semibold transition",
                    viewMode === "raw" ? "bg-surface text-textPrimary shadow-sm" : "text-textMuted"
                  )}
                >
                  Raw Markdown
                </button>
              </div>
            </div>

            <div className="mt-5 rounded-2xl border border-borderLight bg-muted p-5">
              <header className="flex flex-wrap items-center justify-between gap-3 border-b border-borderLight pb-4">
                <div>
                  <p className="text-xs uppercase tracking-wide text-textMuted">Stage</p>
                  <h3 className="text-lg font-semibold text-textPrimary">{activeStage.label}</h3>
                  <p className="text-xs text-textMuted">{activeStage.description}</p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={handleCopyMarkdown}
                    disabled={!currentResult}
                    className={clsx(
                      "rounded-lg border border-borderLight px-3 py-1.5 text-xs font-semibold text-textMuted transition hover:text-textPrimary",
                      !currentResult && "cursor-not-allowed opacity-50"
                    )}
                  >
                    {hasCopied ? "Copied!" : "Copy Markdown"}
                  </button>
                  <button
                    type="button"
                    onClick={handleRegenerateStage}
                    disabled={loading}
                    className={clsx(
                      "rounded-lg border border-borderLight px-3 py-1.5 text-xs font-semibold text-textMuted transition hover:text-textPrimary",
                      loading && "cursor-not-allowed opacity-50"
                    )}
                  >
                    {loading ? "Working…" : "Regenerate Stage"}
                  </button>
                </div>
              </header>

              {inheritedContext ? (
                <div className="mt-4 rounded-xl border border-borderLight bg-surface px-4 py-3 text-xs text-textMuted shadow-sm">
                  Context forwarded from{" "}
                  <span className="font-semibold text-textPrimary">{originStageLabel}</span>: {inheritedContext.summary}
                </div>
              ) : null}

              <div className="mt-4 rounded-xl bg-surface p-4 shadow-inner">
                {currentResult ? (
                  viewMode === "raw" ? (
                    <pre className="whitespace-pre-wrap break-words text-sm leading-relaxed text-textMuted">
                      {currentResult.markdown}
                    </pre>
                  ) : (
                    <div className="space-y-3 text-sm leading-relaxed text-textMuted" dangerouslySetInnerHTML={{ __html: previewHtml }} />
                  )
                ) : (
                  <p className="text-sm text-textMuted">
                    Generate content for {activeStage.label} to see the structured Markdown preview.
                  </p>
                )}
              </div>
            </div>
          </section>

          <section className="lg:hidden rounded-2xl border border-borderLight bg-surface p-4 shadow-sm">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-textMuted">Idea History</h2>
            {historyItems.length === 0 ? (
              <p className="mt-4 text-xs text-textMuted">Your prompts will appear here after you run the journey.</p>
            ) : (
              <ul className="mt-4 space-y-3">
                {historyItems.map((entry) => (
                  <li key={entry.id} className="rounded-xl border border-borderLight px-3 py-3">
                    <p className="text-xs uppercase tracking-wide text-textMuted">
                      {new Date(entry.timestamp).toLocaleString(undefined, {
                        hour: "2-digit",
                        minute: "2-digit"
                      })}
                      {" · "}
                      {entry.cloneLabel}
                    </p>
                    <p className="mt-1 text-sm text-textPrimary" style={{ display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
                      {entry.prompt}
                    </p>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}
