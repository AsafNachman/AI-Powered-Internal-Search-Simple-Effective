import type {
  CleanupReport,
  FilingProposal,
  Health,
  Job,
  RootSummary,
  SearchPlan,
  Source,
  TreeResponse,
  WatchStatus,
} from "./types";

/**
 * All requests go to a relative /api path. next.config.ts rewrites that to the
 * FastAPI process, so the browser sees a single origin and CORS never applies.
 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // Non-JSON error body; the status line is the best we have.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/api/health"),

  roots: () => request<{ roots: RootSummary[] }>("/api/roots"),

  forgetRoot: (path: string) =>
    request<{ detail: string }>(`/api/roots?path=${encodeURIComponent(path)}`, {
      method: "DELETE",
    }),

  startIndex: (path: string, force = false) =>
    request<Job>("/api/index", {
      method: "POST",
      body: JSON.stringify({ path, force }),
    }),

  job: (id: string) => request<Job>(`/api/jobs/${id}`),

  cancelJob: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),

  tree: (path: string, maxDepth = 4) =>
    request<TreeResponse>(
      `/api/tree?path=${encodeURIComponent(path)}&max_depth=${maxDepth}`,
    ),

  cleanup: (path: string) =>
    request<CleanupReport>(`/api/cleanup?path=${encodeURIComponent(path)}`),

  suggestFiling: (path: string, file: string) =>
    request<FilingProposal>("/api/filing/suggest", {
      method: "POST",
      body: JSON.stringify({ path, file, apply: false }),
    }),

  applyFiling: (path: string, file: string, destination: string, filename?: string) =>
    request<{ applied: boolean; finalPath: string }>("/api/filing/apply", {
      method: "POST",
      body: JSON.stringify({ path, file, destination, filename, apply: true }),
    }),

  setWatch: (path: string, enabled: boolean) =>
    request<WatchStatus>("/api/watch", {
      method: "POST",
      body: JSON.stringify({ path, enabled }),
    }),

  watchStatus: (path: string) =>
    request<WatchStatus>(`/api/watch?path=${encodeURIComponent(path)}`),
};

/** Poll a job until it reaches a terminal state. */
export async function pollJob(
  id: string,
  onUpdate: (job: Job) => void,
  intervalMs = 700,
): Promise<Job> {
  for (;;) {
    const job = await api.job(id);
    onUpdate(job);
    if (job.status === "succeeded" || job.status === "failed" || job.status === "cancelled") {
      return job;
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

export interface ChatHandlers {
  onPlan?: (plan: SearchPlan) => void;
  onSources?: (sources: Source[]) => void;
  onToken?: (token: string) => void;
  onError?: (detail: string) => void;
  onDone?: () => void;
}

/**
 * Consume the /api/chat SSE stream.
 *
 * EventSource cannot be used because it only issues GET requests and cannot
 * send a JSON body, so we read the response stream by hand. The parser buffers
 * bytes until it sees the SSE frame delimiter (a blank line); a chunk boundary
 * can land anywhere, including mid-token, so partial frames must be carried
 * over to the next read.
 */
export async function streamChat(
  body: { path: string; question: string; top_k?: number },
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  // The connection itself can fail before any response exists (backend
  // restarting, Ollama dying mid-request, a dropped dev-proxy socket). That
  // throws out of fetch() rather than resolving to a non-ok response, so it
  // must be caught here too -- otherwise it becomes an unhandled rejection
  // that never calls onDone, leaving the UI stuck in a "busy" state forever.
  let response: Response;
  try {
    response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, stream: true }),
      signal,
    });
  } catch (error) {
    if ((error as Error)?.name !== "AbortError") {
      handlers.onError?.(
        `Could not reach the backend: ${(error as Error)?.message || "network error"}`,
      );
    }
    handlers.onDone?.();
    return;
  }

  if (!response.ok || !response.body) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const parsed = await response.json();
      if (parsed?.detail) detail = parsed.detail;
    } catch {
      /* keep the status line */
    }
    handlers.onError?.(detail);
    handlers.onDone?.();
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;

  // The server sends an explicit "done" frame and the stream also ends; this
  // guarantees the caller's onDone runs exactly once either way.
  const finish = () => {
    if (finished) return;
    finished = true;
    handlers.onDone?.();
  };

  const dispatch = (frame: string) => {
    let event = "message";
    const dataLines: string[] = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (dataLines.length === 0) return;

    let payload: Record<string, unknown>;
    try {
      payload = JSON.parse(dataLines.join("\n"));
    } catch {
      return;
    }

    switch (event) {
      case "plan":
        handlers.onPlan?.(payload as unknown as SearchPlan);
        break;
      case "sources":
        handlers.onSources?.((payload.sources ?? []) as Source[]);
        break;
      case "token":
        handlers.onToken?.(String(payload.t ?? ""));
        break;
      case "error":
        handlers.onError?.(String(payload.detail ?? "Unknown error"));
        break;
      case "done":
        finish();
        break;
    }
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      // stream: true keeps multi-byte UTF-8 characters intact across chunks.
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        dispatch(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf("\n\n");
      }
    }
    if (buffer.trim()) dispatch(buffer);
  } catch (error) {
    if ((error as Error)?.name !== "AbortError") {
      handlers.onError?.((error as Error).message);
    }
  } finally {
    finish();
  }
}

export function formatBytes(bytes: number): string {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / Math.pow(1024, exponent);
  return `${exponent === 0 ? value : value.toFixed(1)} ${units[exponent]}`;
}

export function formatDate(epochSeconds: number | null | undefined): string {
  if (!epochSeconds) return "-";
  return new Date(epochSeconds * 1000).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}
