"use client";

import { useCallback, useEffect, useState } from "react";
import { ChatPanel } from "@/components/ChatPanel";
import { CleanupPanel } from "@/components/CleanupPanel";
import { DirectoryTree } from "@/components/DirectoryTree";
import { FilingPanel } from "@/components/FilingPanel";
import { IndexBar } from "@/components/IndexBar";
import { Panel } from "@/components/ui";
import { api } from "@/lib/api";
import type { TreeResponse } from "@/lib/types";

type Tab = "cleanup" | "filing";

const ROOT_STORAGE_KEY = "ais:last-root";

export default function Home() {
  const [root, setRoot] = useState("");
  const [tree, setTree] = useState<TreeResponse | null>(null);
  const [tab, setTab] = useState<Tab>("cleanup");
  const [ready, setReady] = useState(false);

  // Restore the last folder on mount. Reading localStorage inside an effect
  // (not during render) keeps the server and client HTML identical, which is
  // what avoids a hydration mismatch.
  useEffect(() => {
    const saved = window.localStorage.getItem(ROOT_STORAGE_KEY);
    if (saved) setRoot(saved);
  }, []);

  const refreshTree = useCallback(async () => {
    const target = root.trim();
    if (!target) return;
    try {
      const data = await api.tree(target, 5);
      setTree(data);
      setReady(data.indexedChunks > 0);
      window.localStorage.setItem(ROOT_STORAGE_KEY, target);
    } catch {
      setTree(null);
      setReady(false);
    }
  }, [root]);

  // Each /api/tree call walks the whole directory, so firing one per keystroke
  // would hammer the disk while the user is still typing a path. Waiting for a
  // pause collapses a burst of edits into a single scan; the cleanup cancels
  // the pending timer whenever `root` changes again.
  useEffect(() => {
    const timer = setTimeout(() => void refreshTree(), 500);
    return () => clearTimeout(timer);
  }, [refreshTree]);

  return (
    <main className="flex h-screen flex-col overflow-hidden">
      <IndexBar root={root} onRootChange={setRoot} onIndexed={() => void refreshTree()} />

      <div className="grid min-h-0 flex-1 gap-3 p-3 lg:grid-cols-[minmax(280px,1fr)_minmax(420px,1.4fr)_minmax(320px,1fr)]">
        <Panel title="Structure">
          <DirectoryTree data={tree} />
        </Panel>

        <Panel title="Chat" bodyClassName="overflow-hidden">
          <ChatPanel root={root.trim()} ready={ready} />
        </Panel>

        <Panel
          title="Tools"
          action={
            <div className="flex gap-1 rounded-lg bg-[var(--color-canvas)] p-0.5">
              {(["cleanup", "filing"] as Tab[]).map((name) => (
                <button
                  key={name}
                  onClick={() => setTab(name)}
                  className={`rounded-md px-2.5 py-1 text-[11px] font-medium capitalize transition-colors ${
                    tab === name
                      ? "bg-[var(--color-accent)] text-white"
                      : "text-[var(--color-muted)] hover:text-[var(--color-ink)]"
                  }`}
                >
                  {name}
                </button>
              ))}
            </div>
          }
        >
          {tab === "cleanup" ? (
            <CleanupPanel root={root.trim()} ready={ready} />
          ) : (
            <FilingPanel root={root.trim()} ready={ready} />
          )}
        </Panel>
      </div>
    </main>
  );
}
