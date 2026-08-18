"use client";

import { useEffect, useRef, useState } from "react";
import { api, formatBytes, pollJob } from "@/lib/api";
import type { Health, Job, RootSummary, WatchStatus } from "@/lib/types";
import { Badge, Button, Spinner } from "./ui";

export function IndexBar({
  root,
  onRootChange,
  onIndexed,
}: {
  root: string;
  onRootChange: (value: string) => void;
  onIndexed: () => void;
}) {
  const [health, setHealth] = useState<Health | null>(null);
  const [roots, setRoots] = useState<RootSummary[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");
  const [watch, setWatch] = useState<WatchStatus | null>(null);
  const lastIndexedAtRef = useRef<number | null>(null);

  const refreshRoots = async () => {
    try {
      setRoots((await api.roots()).roots);
    } catch {
      /* the roots list is informational only */
    }
  };

  // Re-poll health so the badge recovers on its own when the backend or
  // Ollama is started after the browser tab was already open.
  useEffect(() => {
    const check = () => void api.health().then(setHealth).catch(() => setHealth(null));
    check();
    void refreshRoots();
    const timer = setInterval(check, 10_000);
    return () => clearInterval(timer);
  }, []);

  // Live watching: once a folder is being watched, poll its status so a
  // background re-index (triggered by a file the user just added or edited,
  // not by clicking "Index") still refreshes the tree/chat as soon as it
  // lands, exactly as if the user had clicked "Index" themselves.
  useEffect(() => {
    const target = root.trim();
    if (!target) {
      setWatch(null);
      return;
    }
    lastIndexedAtRef.current = null;
    let cancelled = false;

    const poll = async () => {
      try {
        const status = await api.watchStatus(target);
        if (cancelled) return;
        setWatch(status);
        if (
          status.watching &&
          status.lastIndexedAt &&
          status.lastIndexedAt !== lastIndexedAtRef.current
        ) {
          const isFirstRead = lastIndexedAtRef.current === null;
          lastIndexedAtRef.current = status.lastIndexedAt;
          if (!isFirstRead) {
            onIndexed();
            void refreshRoots();
          }
        }
      } catch {
        if (!cancelled) setWatch(null);
      }
    };

    void poll();
    const timer = setInterval(poll, 2500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [root]);

  const startIndex = async (force: boolean) => {
    if (!root.trim()) return;
    setError("");
    try {
      const created = await api.startIndex(root.trim(), force);
      setJob(created);
      const finished = await pollJob(created.id, setJob);
      if (finished.status === "succeeded") {
        onIndexed();
        void refreshRoots();
        // Start listening for further changes the moment a folder has a
        // usable index, so the user never has to remember to re-index.
        try {
          setWatch(await api.setWatch(root.trim(), true));
        } catch {
          /* watching is a convenience layer; indexing already succeeded */
        }
      } else if (finished.status === "failed") {
        setError(finished.error);
      }
    } catch (caught) {
      setError((caught as Error).message);
    }
  };

  const toggleWatch = async () => {
    const target = root.trim();
    if (!target) return;
    try {
      setWatch(await api.setWatch(target, !watch?.watching));
    } catch (caught) {
      setError((caught as Error).message);
    }
  };

  const running = job?.status === "running" || job?.status === "pending";
  const ollamaOk = health?.ollama.reachable && health.ollama.chatModelReady;

  return (
    <header className="shrink-0 border-b border-[var(--color-line)] bg-[var(--color-surface)]">
      <div className="flex flex-wrap items-center gap-3 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-lg">{"\u{1F50E}"}</span>
          <h1 className="text-sm font-semibold tracking-tight">AI-Powered Internal Search</h1>
        </div>

        <div className="flex min-w-[320px] flex-1 items-center gap-2">
          <input
            value={root}
            onChange={(event) => onRootChange(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && void startIndex(false)}
            list="known-roots"
            placeholder="C:\Users\me\Documents  or  /Users/me/Documents"
            disabled={running}
            className="min-w-0 flex-1 rounded-lg border border-[var(--color-line)] bg-[var(--color-canvas)] px-3 py-2 font-mono text-[12px] outline-none placeholder:text-[var(--color-muted)] focus:border-[var(--color-accent)] disabled:opacity-50"
          />
          <datalist id="known-roots">
            {roots.map((entry) => (
              <option key={entry.path} value={entry.path} />
            ))}
          </datalist>

          <Button onClick={() => void startIndex(false)} disabled={running || !root.trim()}>
            {running ? <Spinner /> : "Index"}
          </Button>
          <Button
            variant="ghost"
            onClick={() => void startIndex(true)}
            disabled={running || !root.trim()}
            title="Discard the cache and re-embed everything"
          >
            Rebuild
          </Button>
        </div>

        <div className="flex items-center gap-2">
          {root.trim() && (
            <button
              onClick={() => void toggleWatch()}
              title={
                watch?.watching
                  ? "Watching this folder -- new and edited files are indexed automatically. Click to stop."
                  : "Not watching. Click to auto-index changes to this folder as they happen."
              }
              className="cursor-pointer"
            >
              <Badge tone={watch?.watching ? "good" : "neutral"}>
                {watch?.watching ? (watch.reindexing ? "reindexing..." : "live") : "not watching"}
              </Badge>
            </button>
          )}
          {health ? (
            <Badge tone={ollamaOk ? "good" : "bad"}>
              {ollamaOk ? `${health.chatModel} ready` : "ollama offline"}
            </Badge>
          ) : (
            <Badge tone="bad">backend offline</Badge>
          )}
        </div>
      </div>

      {(running || error || job?.status === "succeeded" || watch?.lastError) && (
        <div className="px-4 pb-3">
          {watch?.lastError && !running && (
            <p className="mb-1 text-[11px] text-[var(--color-warn)]">
              Live watch error: {watch.lastError}
            </p>
          )}
          {running && job && (
            <div className="space-y-1">
              <div className="h-1 overflow-hidden rounded-full bg-[var(--color-surface-2)]">
                <div
                  className="h-full rounded-full bg-[var(--color-accent)] transition-[width] duration-300"
                  style={{ width: `${Math.max(2, job.progress * 100)}%` }}
                />
              </div>
              <div className="flex items-center justify-between text-[11px] text-[var(--color-muted)]">
                <span className="truncate">{job.message}</span>
                <button
                  onClick={() => void api.cancelJob(job.id).catch(() => undefined)}
                  className="shrink-0 hover:text-[var(--color-bad)]"
                >
                  cancel
                </button>
              </div>
            </div>
          )}

          {!running && job?.status === "succeeded" && job.result && (
            <div className="flex flex-wrap gap-3 text-[11px] text-[var(--color-muted)]">
              <span>
                indexed <strong className="text-[var(--color-ink)]">{job.result.indexedFiles}</strong>
              </span>
              <span>
                unchanged{" "}
                <strong className="text-[var(--color-ink)]">{job.result.skippedUnchanged}</strong>
              </span>
              <span>
                vectors{" "}
                <strong className="text-[var(--color-ink)]">{job.result.chunksWritten}</strong>
              </span>
              <span>
                summaries{" "}
                <strong className="text-[var(--color-ink)]">
                  {job.result.foldersSummarized}
                </strong>
              </span>
              <span>
                {formatBytes(job.result.totalSizeBytes)} in {job.result.durationSeconds}s
              </span>
              {job.result.warnings.map((warning) => (
                <span key={warning} className="text-[var(--color-warn)]">
                  {warning}
                </span>
              ))}
            </div>
          )}

          {error && <p className="text-[11px] text-[var(--color-bad)]">{error}</p>}
        </div>
      )}
    </header>
  );
}
