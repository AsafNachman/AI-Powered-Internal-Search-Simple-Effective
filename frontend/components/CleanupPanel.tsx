"use client";

import { useState, type ReactNode } from "react";
import { api, formatBytes, formatDate } from "@/lib/api";
import type { CleanupReport, Finding } from "@/lib/types";
import { Badge, Button, Empty, Spinner } from "./ui";

function FindingRows({ items, tone }: { items: Finding[]; tone: "warn" | "neutral" }) {
  return (
    <ul className="space-y-1">
      {items.map((item) => (
        <li
          key={item.rel_path}
          className="flex items-center gap-2 rounded-md bg-[var(--color-surface-2)] px-2 py-1.5 text-[12px]"
        >
          <span className="min-w-0 flex-1 truncate" title={item.rel_path}>
            {item.rel_path}
          </span>
          <Badge tone={tone}>{formatBytes(item.size_bytes)}</Badge>
          <span className="shrink-0 text-[10px] text-[var(--color-muted)]">
            {formatDate(item.modified_at)}
          </span>
        </li>
      ))}
    </ul>
  );
}

function Section({
  title,
  count,
  children,
}: {
  title: string;
  count: number;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  if (count === 0) return null;
  return (
    <div className="border-t border-[var(--color-line)] pt-3">
      <button
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2 text-left text-[12px] font-semibold uppercase tracking-wider text-[var(--color-muted)] hover:text-[var(--color-ink)]"
      >
        <span>{open ? "\u25be" : "\u25b8"}</span>
        {title}
        <Badge tone="warn">{count}</Badge>
      </button>
      {open && <div className="mt-2">{children}</div>}
    </div>
  );
}

export function CleanupPanel({ root, ready }: { root: string; ready: boolean }) {
  const [report, setReport] = useState<CleanupReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const run = async () => {
    setBusy(true);
    setError("");
    try {
      setReport(await api.cleanup(root));
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (!ready) return <Empty>Index a folder to analyse it for cleanup.</Empty>;

  return (
    <div className="space-y-3 p-4">
      <div className="flex items-center gap-2">
        <Button onClick={() => void run()} disabled={busy}>
          {busy ? <Spinner /> : report ? "Re-analyse" : "Analyse folder"}
        </Button>
        {report && (
          <span className="text-[12px] text-[var(--color-muted)]">
            {report.filesExamined} files - {formatBytes(report.totalSizeBytes)}
          </span>
        )}
      </div>

      {error && (
        <p className="rounded-lg bg-[#2d1216] px-3 py-2 text-[12px] text-[var(--color-bad)]">
          {error}
        </p>
      )}

      {busy && !report && (
        <p className="text-[12px] text-[var(--color-muted)]">
          Hashing candidates and scanning for stale files...
        </p>
      )}

      {report && (
        <>
          <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-surface-2)] p-3">
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-semibold text-[var(--color-good)]">
                {report.reclaimableHuman}
              </span>
              <span className="text-[12px] text-[var(--color-muted)]">reclaimable</span>
            </div>
            {report.narrative && (
              <p className="mt-2 text-[13px] leading-relaxed text-[var(--color-ink)]">
                {report.narrative}
              </p>
            )}
          </div>

          <Section title="Exact duplicates" count={report.duplicateGroups.length}>
            <ul className="space-y-2">
              {report.duplicateGroups.map((group) => (
                <li
                  key={group.digest}
                  className="rounded-md bg-[var(--color-surface-2)] p-2 text-[12px]"
                >
                  <div className="mb-1 flex items-center gap-2">
                    <Badge tone="bad">{group.paths.length} copies</Badge>
                    <span className="text-[var(--color-muted)]">
                      {formatBytes(group.size_bytes)} each - wasting{" "}
                      {formatBytes(group.wasted_bytes)}
                    </span>
                  </div>
                  {group.paths.map((path, position) => (
                    <div
                      key={path}
                      className={`truncate font-mono text-[11px] ${
                        position === 0 ? "text-[var(--color-good)]" : "text-[var(--color-muted)]"
                      }`}
                      title={path}
                    >
                      {position === 0 ? "keep  " : "dupe  "}
                      {path}
                    </div>
                  ))}
                </li>
              ))}
            </ul>
          </Section>

          <Section title="Probable older revisions" count={report.nearDuplicates.length}>
            <FindingRows items={report.nearDuplicates} tone="warn" />
          </Section>

          <Section title="Temp & OS junk" count={report.junkFiles.length}>
            <FindingRows items={report.junkFiles} tone="warn" />
          </Section>

          <Section title="Largest files" count={report.largeFiles.length}>
            <FindingRows items={report.largeFiles} tone="neutral" />
          </Section>

          <Section title="Stale files" count={report.staleFiles.length}>
            <FindingRows items={report.staleFiles} tone="neutral" />
          </Section>

          <Section title="Zero-byte files" count={report.emptyFiles.length}>
            <FindingRows items={report.emptyFiles} tone="neutral" />
          </Section>

          <Section title="Empty folders" count={report.emptyDirs.length}>
            <ul className="space-y-1">
              {report.emptyDirs.map((directory) => (
                <li
                  key={directory}
                  className="truncate rounded-md bg-[var(--color-surface-2)] px-2 py-1 font-mono text-[11px] text-[var(--color-muted)]"
                >
                  {directory}
                </li>
              ))}
            </ul>
          </Section>

          <p className="pt-2 text-[11px] italic text-[var(--color-muted)]">
            Read-only analysis. Nothing is deleted or moved.
          </p>
        </>
      )}
    </div>
  );
}
