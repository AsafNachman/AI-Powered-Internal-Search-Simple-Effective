"use client";

import type { ReactNode } from "react";

export function Panel({
  title,
  action,
  children,
  className = "",
  // Panels that host their own scroll container (the chat transcript) pass
  // "overflow-hidden" so the two scrollbars don't nest.
  bodyClassName = "overflow-auto",
}: {
  title?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={`flex min-h-0 flex-col rounded-xl border border-[var(--color-line)] bg-[var(--color-surface)] ${className}`}
    >
      {title && (
        <header className="flex shrink-0 items-center justify-between gap-2 border-b border-[var(--color-line)] px-4 py-3">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-[var(--color-muted)]">
            {title}
          </h2>
          {action}
        </header>
      )}
      <div className={`min-h-0 flex-1 ${bodyClassName}`}>{children}</div>
    </section>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "primary",
  type = "button",
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "ghost" | "danger";
  type?: "button" | "submit";
  title?: string;
}) {
  const styles = {
    primary: "bg-[var(--color-accent)] text-white hover:bg-[#4a7bf0]",
    ghost:
      "border border-[var(--color-line)] bg-transparent text-[var(--color-ink)] hover:bg-[var(--color-surface-2)]",
    danger: "border border-[#4a2126] bg-transparent text-[var(--color-bad)] hover:bg-[#241417]",
  }[variant];

  return (
    <button
      type={type}
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${styles}`}
    >
      {children}
    </button>
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "warn" | "bad" | "accent";
}) {
  const tones = {
    neutral: "bg-[var(--color-surface-2)] text-[var(--color-muted)]",
    good: "bg-[#12301f] text-[var(--color-good)]",
    warn: "bg-[#332508] text-[var(--color-warn)]",
    bad: "bg-[#2d1216] text-[var(--color-bad)]",
    accent: "bg-[var(--color-accent-soft)] text-[var(--color-accent)]",
  }[tone];
  return (
    <span className={`rounded-md px-2 py-0.5 font-mono text-[11px] ${tones}`}>{children}</span>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full items-center justify-center p-8 text-center text-sm text-[var(--color-muted)]">
      {children}
    </div>
  );
}

export function Spinner() {
  return (
    <span className="inline-flex gap-1" aria-label="loading">
      <span className="typing-dot h-1.5 w-1.5 rounded-full bg-current" />
      <span className="typing-dot h-1.5 w-1.5 rounded-full bg-current [animation-delay:0.2s]" />
      <span className="typing-dot h-1.5 w-1.5 rounded-full bg-current [animation-delay:0.4s]" />
    </span>
  );
}
