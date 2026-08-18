#!/usr/bin/env python3
"""Generate a realistic messy folder so the demo has something to index.

Includes the exact conditions each feature is meant to detect: an exact
duplicate pair, a versioned near-duplicate, OS junk, a zero-byte file, a stale
file, and an unfiled document sitting in an Inbox for the filing agent.

    python scripts/make_sample_tree.py ./sample_data
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

FILES: dict[str, str] = {
    "Clients/Acme Corp/contract-acme-2026.md": """# Master Services Agreement - Acme Corp

Effective 2026-01-15. Acme Corp engages us for platform engineering support.

- Term: 24 months, auto-renewing
- Fee: $18,000 per month, invoiced on the first business day
- Payment terms: net 30
- Termination: 90 days written notice by either party
- Liability cap: 12 months of fees
- Primary contact: Dana Whitfield, VP Engineering
""",
    "Clients/Acme Corp/meeting-notes-2026-02-11.md": """# Acme Corp - Quarterly Review, 11 Feb 2026

Attendees: Dana Whitfield (Acme), Priya Raman (us).

- Latency on the ingest endpoint is the top complaint; p99 sits at 1.8s.
- Dana asked about expanding the contract to cover on-call rotation.
- Renewal conversation scheduled for April.
- Action: send a revised pricing sheet covering the on-call add-on.
""",
    "Clients/Northwind/contract-northwind-2025.md": """# Master Services Agreement - Northwind Trading

Effective 2025-06-01. Data warehouse migration and ongoing support.

- Term: 12 months
- Fee: $9,500 per month
- Payment terms: net 45
- Scope: Snowflake migration, dbt model authoring, analyst training
- Primary contact: Marcus Bell, Head of Data
""",
    "Clients/Northwind/migration-postmortem.md": """# Northwind Migration Post-mortem

The cutover ran four hours past the window.

Root cause: an undocumented nightly ETL job held locks on the source tables.
We discovered it only when the copy stalled at 60%.

Fixes applied:
1. Inventory every scheduled job before a cutover, not just the documented ones.
2. Run the final delta copy with the source in read-only mode.
3. Budget a two-hour buffer inside the maintenance window.
""",
    "Finance/invoices/invoice-2026-01-acme.md": """Invoice INV-2026-001
Bill to: Acme Corp
Date: 2026-01-31
Line item: Platform engineering retainer, January 2026 - $18,000.00
Total due: $18,000.00
Terms: net 30
""",
    "Finance/invoices/invoice-2026-02-acme.md": """Invoice INV-2026-014
Bill to: Acme Corp
Date: 2026-02-28
Line item: Platform engineering retainer, February 2026 - $18,000.00
Line item: Additional on-call coverage - $2,400.00
Total due: $20,400.00
Terms: net 30
""",
    "Finance/invoices/invoice-2026-02-northwind.md": """Invoice INV-2026-015
Bill to: Northwind Trading
Date: 2026-02-28
Line item: Data warehouse support retainer, February 2026 - $9,500.00
Total due: $9,500.00
Terms: net 45
""",
    "Finance/budget-2026.csv": """category,q1,q2,q3,q4,notes
Salaries,420000,430000,445000,445000,Two hires planned in Q2
Cloud infrastructure,58000,61000,64000,67000,Growth tracks customer count
Software licences,12000,12000,13500,13500,Renewal in Q3
Marketing,35000,48000,42000,55000,Conference spend concentrated in Q2 and Q4
Travel,9000,14000,9000,12000,
Contingency,25000,25000,25000,25000,
""",
    "Finance/pricing-model.csv": """tier,monthly_usd,seats_included,overage_per_seat,support_sla
Starter,299,10,29,next business day
Team,899,40,22,8 business hours
Business,2400,150,17,4 hours
Enterprise,7500,600,12,1 hour with named engineer
""",
    "Product/roadmap-2026.md": """# Product Roadmap 2026

## Q1 - Foundations
- Ship the local-first indexer
- Incremental re-index using the file manifest
- Folder summarisation

## Q2 - Retrieval quality
- Hybrid search: combine BM25 keyword scoring with dense vectors
- Cross-encoder re-ranking of the top 40 candidates
- Per-user relevance feedback

## Q3 - Automation
- Auto-filing agent with a human approval queue
- Scheduled background re-index via a filesystem watcher

## Q4 - Scale
- Multi-root workspaces
- Team deployment with shared indexes
""",
    "Product/specs/search-ranking-spec.md": """# Spec: Ranking

Retrieval over-fetches 40 candidates and re-ranks before answering.

Signals:
- Dense cosine similarity from the embedding model (primary)
- Filename lexical match, worth +0.25 when the query names a file
- Recency, applied as a hard metadata filter rather than a score nudge
- Per-file diversity cap of 3 chunks, so one long document cannot dominate

Open question: whether a cross-encoder is worth the latency on CPU-only
hardware. Benchmarks suggest +12% MRR at a cost of roughly 400ms per query.
""",
    "Product/specs/filing-agent-spec.md": """# Spec: Auto-filing Agent

Goal: given a new file, choose the folder a careful archivist would pick.

Approach: embed the incoming file, retrieve its nearest neighbours in the
existing index, and aggregate their parent folders into scored candidates.
The LLM chooses among those candidates by index; it never emits a raw path,
which makes a hallucinated destination structurally impossible.

Confidence below 0.6 always routes to a proposal rather than an automatic move.
""",
    "Engineering/runbooks/incident-response.md": """# Incident Response Runbook

1. Declare the incident in #ops and name a single incident commander.
2. Post the customer-facing status update within 15 minutes.
3. Mitigate before diagnosing. Roll back first, understand later.
4. Capture a timeline as you go; memory degrades fast after the fact.
5. Post-mortem within five business days, blameless, with owned action items.

Escalation: on-call primary -> secondary after 10 minutes -> engineering lead.
""",
    "Engineering/runbooks/backup-restore.md": """# Backup and Restore

Nightly snapshots at 02:00 UTC, retained 30 days. Weekly full backups
retained for one year in cold storage.

Restore drill (run quarterly):
1. Provision an empty staging database.
2. Restore the most recent full backup, then replay WAL segments.
3. Verify row counts against the production read replica.
4. Record the wall-clock restore time; the RTO target is 90 minutes.
""",
    "Engineering/architecture-notes.md": """# Architecture Notes

The indexer is a pipeline: scan, diff, extract, chunk, embed, store.

The diff step compares (size, mtime) against a per-root JSON manifest so a
re-index only touches files that actually changed. On a 5,000-file share this
is the difference between eight minutes and four seconds.

Embeddings live in ChromaDB with cosine distance, one collection per root.
Everything runs locally against Ollama - no document content leaves the host.
""",
    "HR/onboarding-checklist.md": """# Engineering Onboarding

Day 1: laptop, SSO, password manager, repository access
Week 1: ship one small change to production, paired with a buddy
Week 2: shadow an on-call shift
Week 4: own a feature end to end
Day 90: review against the role expectations doc
""",
    "HR/policies/remote-work-policy.md": """# Remote Work Policy

Core collaboration hours are 10:00-15:00 in your local timezone.

Equipment stipend: $1,500 on joining, refreshed every three years.
Co-working allowance: up to $250 per month with manager approval.
Travel: the whole company meets in person twice a year.
""",
    "Archive/2019/legacy-notes.md": """Notes from the 2019 platform rewrite. Kept for historical reference only.
The monolith was split into six services; three were merged back in 2021.
""",
    "Inbox/unsorted-vendor-agreement.md": """# Vendor Agreement - Contoso Logistics

Effective 2026-03-01. Contoso provides fulfilment services.

- Term: 18 months
- Fee: $6,200 per month plus per-shipment charges
- Payment terms: net 30
- Liability cap: 6 months of fees
- Primary contact: Elena Marsh, Account Director

This file is deliberately left unfiled so the auto-filing agent has something
to route. It should land beside the other contracts.
""",
}

# Files that exist to be *found* by the cleanup analyser.
DUPLICATE_OF = "Finance/invoices/invoice-2026-01-acme.md"
DUPLICATE_AT = "Archive/2019/invoice-2026-01-acme (copy).md"
NEAR_DUPLICATE_AT = "Product/roadmap-2026 (1).md"
JUNK = ["Clients/.DS_Store", "Finance/Thumbs.db", "Engineering/notes.txt.tmp"]
EMPTY = "Inbox/placeholder.txt"
STALE = "Archive/2019/legacy-notes.md"


def main() -> int:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_data").resolve()
    target.mkdir(parents=True, exist_ok=True)

    for relative, content in FILES.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    # Byte-identical copy -> exact-duplicate detection.
    duplicate = target / DUPLICATE_AT
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    duplicate.write_text(FILES[DUPLICATE_OF], encoding="utf-8")

    # Same base name, different content -> near-duplicate detection.
    (target / NEAR_DUPLICATE_AT).write_text(
        FILES["Product/roadmap-2026.md"] + "\n(Superseded draft.)\n", encoding="utf-8"
    )

    for relative in JUNK:
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("junk", encoding="utf-8")

    (target / EMPTY).write_text("", encoding="utf-8")
    (target / "Archive/empty-folder").mkdir(parents=True, exist_ok=True)

    # Backdate two years so the stale-file heuristic fires.
    stale_time = time.time() - 730 * 86_400
    os.utime(target / STALE, (stale_time, stale_time))

    file_count = sum(1 for path in target.rglob("*") if path.is_file())
    print(f"Created {file_count} files under {target}")
    print("\nIndex this path in the UI, then try:")
    print('  "What are the payment terms for Acme?"')
    print('  "Which folder holds the client contracts?"')
    print('  "Find the spec about ranking"')
    print(f"\nFor the filing agent, use: {target / 'Inbox' / 'unsorted-vendor-agreement.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
