"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import type { FilingProposal } from "@/lib/types";
import { Badge, Button, Empty, Spinner } from "./ui";

function confidenceTone(value: number): "good" | "warn" | "bad" {
  if (value >= 0.7) return "good";
  if (value >= 0.45) return "warn";
  return "bad";
}

export function FilingPanel({ root, ready }: { root: string; ready: boolean }) {
  const [file, setFile] = useState("");
  const [proposal, setProposal] = useState<FilingProposal | null>(null);
  const [chosenDir, setChosenDir] = useState("");
  const [filename, setFilename] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [applied, setApplied] = useState("");

  const suggest = async () => {
    if (!file.trim()) return;
    setBusy(true);
    setError("");
    setApplied("");
    setProposal(null);
    try {
      const result = await api.suggestFiling(root, file.trim());
      setProposal(result);
      setChosenDir(result.destinationDir);
      setFilename(result.suggestedFilename);
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const apply = async () => {
    if (!proposal || !chosenDir) return;
    setBusy(true);
    setError("");
    try {
      const result = await api.applyFiling(root, proposal.sourcePath, chosenDir, filename);
      setApplied(result.finalPath);
      setProposal(null);
      setFile("");
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (!ready) return <Empty>Index a folder so the agent has somewhere to file things.</Empty>;

  return (
    <div className="space-y-3 p-4">
      <p className="text-[12px] leading-relaxed text-[var(--color-muted)]">
        Paste the full path of a file. The agent reads it, finds semantically similar
        documents already in the index, and proposes the folder they live in.
      </p>

      <div className="flex gap-2">
        <input
          value={file}
          onChange={(event) => setFile(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && void suggest()}
          placeholder="C:\Users\me\Downloads\invoice-2026-03.pdf"
          className="min-w-0 flex-1 rounded-lg border border-[var(--color-line)] bg-[var(--color-canvas)] px-3 py-2 font-mono text-[12px] outline-none placeholder:text-[var(--color-muted)] focus:border-[var(--color-accent)]"
        />
        <Button onClick={() => void suggest()} disabled={busy || !file.trim()}>
          {busy && !proposal ? <Spinner /> : "Analyse"}
        </Button>
      </div>

      {error && (
        <p className="rounded-lg bg-[#2d1216] px-3 py-2 text-[12px] text-[var(--color-bad)]">
          {error}
        </p>
      )}

      {applied && (
        <p className="rounded-lg bg-[#12301f] px-3 py-2 text-[12px] text-[var(--color-good)]">
          Moved to <span className="font-mono">{applied}</span>
        </p>
      )}

      {proposal && (
        <div className="space-y-3">
          <div className="rounded-xl border border-[var(--color-accent)] bg-[var(--color-accent-soft)] p-3">
            <div className="mb-1 flex items-center gap-2">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--color-accent)]">
                Recommended
              </span>
              <Badge tone={confidenceTone(proposal.confidence)}>
                {(proposal.confidence * 100).toFixed(0)}% confident
              </Badge>
            </div>
            <div className="font-mono text-[13px] break-all">
              {proposal.destinationRel || "(root folder)"}
            </div>
            <p className="mt-1.5 text-[12px] leading-relaxed text-[var(--color-ink)]">
              {proposal.reason}
            </p>
          </div>

          <label className="block">
            <span className="mb-1 block text-[11px] uppercase tracking-wider text-[var(--color-muted)]">
              Destination
            </span>
            <select
              value={chosenDir}
              onChange={(event) => setChosenDir(event.target.value)}
              className="w-full rounded-lg border border-[var(--color-line)] bg-[var(--color-canvas)] px-3 py-2 font-mono text-[12px] outline-none focus:border-[var(--color-accent)]"
            >
              {proposal.candidates.map((candidate) => (
                <option key={candidate.absPath} value={candidate.absPath}>
                  {candidate.relPath || "(root)"} - score {candidate.score.toFixed(2)} (
                  {candidate.matches} matches)
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="mb-1 block text-[11px] uppercase tracking-wider text-[var(--color-muted)]">
              Filename
            </span>
            <input
              value={filename}
              onChange={(event) => setFilename(event.target.value)}
              className="w-full rounded-lg border border-[var(--color-line)] bg-[var(--color-canvas)] px-3 py-2 font-mono text-[12px] outline-none focus:border-[var(--color-accent)]"
            />
          </label>

          <div className="flex gap-2">
            <Button onClick={() => void apply()} disabled={busy}>
              {busy ? <Spinner /> : "Move file here"}
            </Button>
            <Button variant="ghost" onClick={() => setProposal(null)}>
              Cancel
            </Button>
          </div>

          {proposal.candidates.length > 1 && (
            <div className="border-t border-[var(--color-line)] pt-3">
              <h4 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-[var(--color-muted)]">
                Why these folders
              </h4>
              <ul className="space-y-1.5">
                {proposal.candidates.map((candidate) => (
                  <li
                    key={candidate.absPath}
                    className="rounded-md bg-[var(--color-surface-2)] p-2 text-[12px]"
                  >
                    <div className="flex items-center gap-2">
                      <span className="truncate font-mono">
                        {candidate.relPath || "(root)"}
                      </span>
                      <Badge>{candidate.score.toFixed(2)}</Badge>
                    </div>
                    {candidate.summary && (
                      <p className="mt-1 text-[var(--color-muted)]">{candidate.summary}</p>
                    )}
                    {candidate.exampleFiles.length > 0 && (
                      <p className="mt-1 truncate text-[10px] text-[var(--color-muted)]">
                        contains: {candidate.exampleFiles.join(", ")}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
