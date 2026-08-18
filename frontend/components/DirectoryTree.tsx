"use client";

import { useState } from "react";
import { formatBytes } from "@/lib/api";
import type { TreeNode, TreeResponse } from "@/lib/types";
import { Badge, Empty } from "./ui";

const EXT_COLORS: Record<string, string> = {
  ".pdf": "text-[#f2555a]",
  ".docx": "text-[#5b8cff]",
  ".xlsx": "text-[#3ecf8e]",
  ".csv": "text-[#3ecf8e]",
  ".pptx": "text-[#f5a524]",
  ".md": "text-[#9b8cff]",
  ".txt": "text-[var(--color-muted)]",
};

function Node({ node, defaultOpen }: { node: TreeNode; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const hasChildren = node.children.length > 0 || node.files.length > 0;

  return (
    <div className="text-sm">
      <button
        onClick={() => setOpen((value) => !value)}
        className="group flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left hover:bg-[var(--color-surface-2)]"
        style={{ paddingLeft: `${node.depth * 12 + 8}px` }}
      >
        <span className="mt-0.5 w-3 shrink-0 text-[var(--color-muted)]">
          {hasChildren ? (open ? "\u25be" : "\u25b8") : ""}
        </span>
        <span className="shrink-0">{open ? "\u{1F4C2}" : "\u{1F4C1}"}</span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-2">
            <span className="truncate font-medium">{node.name || "/"}</span>
            <Badge>{node.totalFiles}</Badge>
            <span className="shrink-0 text-[11px] text-[var(--color-muted)]">
              {formatBytes(node.totalSize)}
            </span>
          </span>
          {node.summary && (
            <span className="mt-0.5 block pr-2 text-[12px] leading-snug text-[var(--color-muted)]">
              {node.summary}
            </span>
          )}
        </span>
      </button>

      {open && (
        <div>
          {node.children.map((child) => (
            <Node key={child.relPath} node={child} defaultOpen={false} />
          ))}
          {node.files.map((file) => (
            <div
              key={file.relPath}
              className="flex items-center gap-2 rounded-md px-2 py-1 text-[13px] text-[var(--color-muted)] hover:bg-[var(--color-surface-2)]"
              style={{ paddingLeft: `${(node.depth + 1) * 12 + 24}px` }}
              title={file.relPath}
            >
              <span className={EXT_COLORS[file.ext] ?? "text-[var(--color-muted)]"}>
                {"\u25CF"}
              </span>
              <span className="truncate text-[var(--color-ink)]">{file.name}</span>
              <span className="ml-auto shrink-0 text-[11px]">{formatBytes(file.size)}</span>
            </div>
          ))}
          {node.truncated && (
            <div
              className="px-2 py-1 text-[11px] italic text-[var(--color-muted)]"
              style={{ paddingLeft: `${(node.depth + 1) * 12 + 24}px` }}
            >
              deeper folders hidden - raise max depth to expand
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function DirectoryTree({ data }: { data: TreeResponse | null }) {
  if (!data) {
    return <Empty>Index a folder to see its structure and AI-written summaries.</Empty>;
  }

  return (
    <div className="pb-4">
      <div className="sticky top-0 z-10 flex flex-wrap gap-3 border-b border-[var(--color-line)] bg-[var(--color-surface)] px-4 py-2 text-[11px] text-[var(--color-muted)]">
        <span>
          <strong className="text-[var(--color-ink)]">{data.totalFiles}</strong> files
        </span>
        <span>
          <strong className="text-[var(--color-ink)]">{data.directories}</strong> folders
        </span>
        <span>
          <strong className="text-[var(--color-ink)]">{formatBytes(data.totalSize)}</strong>
        </span>
        <span>
          <strong className="text-[var(--color-ink)]">{data.indexedChunks}</strong> vectors
        </span>
      </div>
      <div className="pt-1">
        <Node node={data.tree} defaultOpen />
      </div>
    </div>
  );
}
