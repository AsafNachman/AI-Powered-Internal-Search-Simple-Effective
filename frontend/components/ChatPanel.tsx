"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { formatDate, streamChat } from "@/lib/api";
import type { ChatMessage, Source } from "@/lib/types";
import { Badge, Button, Empty, Spinner } from "./ui";

const SUGGESTIONS = [
  "What is in this folder?",
  "Find the PDF about pricing",
  "Which folder holds the client contracts?",
  "Summarise anything changed last month",
];

/**
 * Renders an answer with [n] citations turned into clickable chips.
 *
 * Split on the citation pattern with a capturing group so the delimiters are
 * preserved in the output array, letting us map alternate entries to markers.
 */
function AnswerText({ text, sources }: { text: string; sources: Source[] }) {
  const parts = text.split(/(\[\d{1,2}\])/g);
  return (
    <p className="whitespace-pre-wrap text-[14px] leading-relaxed">
      {parts.map((part, index) => {
        const match = /^\[(\d{1,2})\]$/.exec(part);
        if (!match) return <span key={index}>{part}</span>;
        const source = sources.find((s) => s.index === Number(match[1]));
        return (
          <span
            key={index}
            title={source ? `${source.relPath}\n\n${source.excerpt}` : "Unknown source"}
            className="mx-0.5 cursor-help rounded bg-[var(--color-accent-soft)] px-1.5 py-0.5 align-baseline font-mono text-[11px] text-[var(--color-accent)]"
          >
            {match[1]}
          </span>
        );
      })}
    </p>
  );
}

function SourceList({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  if (sources.length === 0) return null;

  return (
    <div className="mt-3 border-t border-[var(--color-line)] pt-2">
      <button
        onClick={() => setOpen((value) => !value)}
        className="text-[11px] font-medium uppercase tracking-wider text-[var(--color-muted)] hover:text-[var(--color-ink)]"
      >
        {open ? "\u25be" : "\u25b8"} {sources.length} sources
      </button>
      {open && (
        <ul className="mt-2 space-y-1.5">
          {sources.map((source) => (
            <li
              key={`${source.index}-${source.relPath}`}
              className="rounded-lg bg-[var(--color-surface-2)] p-2 text-[12px]"
            >
              <div className="flex items-center gap-2">
                <Badge tone="accent">{source.index}</Badge>
                <span className="truncate font-medium">{source.name || source.relPath}</span>
                <span className="ml-auto shrink-0 font-mono text-[10px] text-[var(--color-muted)]">
                  {(source.score * 100).toFixed(0)}%
                </span>
              </div>
              <div className="mt-1 truncate font-mono text-[10px] text-[var(--color-muted)]">
                {source.relPath || "(root)"} - {formatDate(source.modifiedAt)}
              </div>
              <p className="mt-1 line-clamp-3 text-[var(--color-muted)]">{source.excerpt}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function ChatPanel({ root, ready }: { root: string; ready: boolean }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Abort any in-flight stream when the component unmounts or the root
  // changes, so a stale answer can never land in a new conversation.
  useEffect(() => {
    return () => abortRef.current?.abort();
  }, [root]);

  const send = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || busy || !ready) return;

      const answerId = crypto.randomUUID();
      setMessages((previous) => [
        ...previous,
        { id: crypto.randomUUID(), role: "user", content: trimmed },
        { id: answerId, role: "assistant", content: "", streaming: true, sources: [] },
      ]);
      setInput("");
      setBusy(true);

      const patch = (changes: Partial<ChatMessage>) =>
        setMessages((previous) =>
          previous.map((message) =>
            message.id === answerId ? { ...message, ...changes } : message,
          ),
        );

      const controller = new AbortController();
      abortRef.current = controller;

      // streamChat is expected to swallow its own errors and always call
      // onDone, but a try/finally here is a second, independent safety net:
      // whatever goes wrong, `busy` must never get stuck `true`, or every
      // future question silently does nothing with no way to recover short
      // of a page reload.
      try {
        await streamChat(
          { path: root, question: trimmed },
          {
            onPlan: (plan) => patch({ plan }),
            onSources: (sources) => patch({ sources }),
            // Functional update: tokens arrive faster than React re-renders, so
            // appending to a captured value would drop characters.
            onToken: (token) =>
              setMessages((previous) =>
                previous.map((message) =>
                  message.id === answerId
                    ? { ...message, content: message.content + token }
                    : message,
                ),
              ),
            onError: (detail) => patch({ error: detail }),
            onDone: () => {
              patch({ streaming: false });
              setBusy(false);
            },
          },
          controller.signal,
        );
      } catch (error) {
        patch({ streaming: false, error: (error as Error)?.message || "Something went wrong." });
      } finally {
        setBusy(false);
      }
    },
    [busy, ready, root],
  );

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        {messages.length === 0 && (
          <Empty>
            <div className="space-y-4">
              <p>
                {ready
                  ? "Ask anything about the indexed folder."
                  : "Index a folder to start asking questions."}
              </p>
              {ready && (
                <div className="flex flex-wrap justify-center gap-2">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => void send(suggestion)}
                      className="rounded-full border border-[var(--color-line)] px-3 py-1 text-[12px] text-[var(--color-muted)] transition-colors hover:border-[var(--color-accent)] hover:text-[var(--color-ink)]"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </Empty>
        )}

        {messages.map((message) =>
          message.role === "user" ? (
            <div key={message.id} className="flex justify-end">
              <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-[var(--color-accent)] px-4 py-2 text-[14px] text-white">
                {message.content}
              </div>
            </div>
          ) : (
            <div key={message.id} className="flex justify-start">
              <div className="w-full max-w-[92%] rounded-2xl rounded-bl-sm border border-[var(--color-line)] bg-[var(--color-surface-2)] px-4 py-3">
                {message.plan && message.plan.extensions.length + message.plan.daysBack > 0 && (
                  <div className="mb-2 flex flex-wrap gap-1.5">
                    {message.plan.extensions.map((extension) => (
                      <Badge key={extension} tone="accent">
                        {extension}
                      </Badge>
                    ))}
                    {message.plan.daysBack > 0 && (
                      <Badge tone="accent">last {message.plan.daysBack}d</Badge>
                    )}
                  </div>
                )}

                {message.content ? (
                  <AnswerText text={message.content} sources={message.sources ?? []} />
                ) : message.streaming ? (
                  <span className="text-[var(--color-muted)]">
                    <Spinner />
                  </span>
                ) : null}

                {message.error && (
                  <p className="mt-2 rounded-lg bg-[#2d1216] px-3 py-2 text-[12px] text-[var(--color-bad)]">
                    {message.error}
                  </p>
                )}

                {!message.streaming && <SourceList sources={message.sources ?? []} />}
              </div>
            </div>
          ),
        )}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          void send(input);
        }}
        className="flex shrink-0 gap-2 border-t border-[var(--color-line)] p-3"
      >
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder={ready ? "Ask about your files..." : "Index a folder first"}
          disabled={!ready || busy}
          className="flex-1 rounded-lg border border-[var(--color-line)] bg-[var(--color-canvas)] px-3 py-2 text-sm outline-none placeholder:text-[var(--color-muted)] focus:border-[var(--color-accent)] disabled:opacity-50"
        />
        {busy ? (
          <Button variant="ghost" onClick={() => abortRef.current?.abort()}>
            Stop
          </Button>
        ) : (
          <Button type="submit" disabled={!ready || !input.trim()}>
            Ask
          </Button>
        )}
      </form>
    </div>
  );
}
