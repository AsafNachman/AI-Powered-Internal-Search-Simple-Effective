# AI-Powered Internal Search

Local-first semantic search, folder summarisation, cleanup analysis and
auto-filing for local and network folders. Point it at a directory; it maps the
tree, reads what it can, embeds it, and answers questions with citations.

**Nothing leaves the machine.** Inference and embeddings both run on Ollama, and
the vector store is a file on disk.

---

## 1. Architecture

```
┌──────────────────────────────┐        ┌───────────────────────────────────┐
│  Next.js 16 (App Router)     │        │  FastAPI                          │
│                              │        │                                   │
│  IndexBar   job polling      │  HTTP  │  /api/index    202 + job id       │
│  Tree       folder summaries │◄──────►│  /api/jobs/:id progress           │
│  Chat       SSE token stream │  SSE   │  /api/tree     live tree          │
│  Cleanup    findings         │        │  /api/chat     SSE answer         │
│  Filing     proposal + move  │        │  /api/cleanup  findings           │
└──────────────────────────────┘        │  /api/filing/* propose / apply    │
                                        └────────────────┬──────────────────┘
                                                         │
                    ┌────────────────────────────────────┼────────────────────┐
                    │                                    │                    │
            ┌───────▼────────┐                  ┌────────▼────────┐   ┌───────▼───────┐
            │ IndexingService│                  │   RagService    │   │  FilingAgent  │
            │                │                  │                 │   │  (LangGraph)  │
            │ scan → diff →  │                  │ plan → retrieve │   │ inspect →     │
            │ extract →chunk │                  │ → diversify →   │   │ retrieve →    │
            │ → embed →store │                  │ generate        │   │ rank → decide │
            └───────┬────────┘                  └────────┬────────┘   └───────┬───────┘
                    │                                    │                    │
                    └──────────────┬─────────────────────┴────────────────────┘
                                   │
                 ┌─────────────────▼──────────────────┐
                 │  VectorRepository  (ChromaDB)      │
                 │  ManifestStore     (JSON per root) │
                 │  Ollama            (chat + embed)  │
                 └────────────────────────────────────┘
```

**Live watching.** Once a folder has been indexed, the backend can watch it
for changes (`POST /api/watch`) using native OS filesystem notifications. A
debounced handler (`app/core/watcher.py`, `app/services/watch.py`) collapses
a burst of edits into one incremental `index_root` run through the same job
pipeline a manual "Index" click uses, so a file dropped into a watched
folder becomes searchable within a couple of seconds without a manual
re-index. The frontend enables this automatically after a successful index
and polls `GET /api/watch` to refresh the tree the moment a background run
finishes.

**Request flow for a question.** The browser POSTs to `/api/chat`. The RAG
service turns the question into a search plan, embeds the semantic part, pulls
40 candidate chunks from Chroma under any metadata filter the plan produced,
re-ranks and caps them at 3 chunks per file, then streams the model's answer
back as Server-Sent Events. Sources are sent in the first frame so the UI can
render citations before the answer finishes.

### Directory structure

```
ai-internal-search/
├── backend/
│   ├── requirements.txt
│   ├── .env.example
│   └── app/
│       ├── main.py                  # ASGI factory, CORS, lifespan, logging
│       ├── config.py                # pydantic-settings singleton
│       ├── schemas.py               # request/response contracts
│       ├── api/routes.py            # every HTTP endpoint
│       ├── core/
│       │   ├── paths.py             # containment checks, safe renames
│       │   ├── textutils.py         # LLM output parsing helpers
│       │   ├── scanner.py           # os.scandir walk + tree aggregation
│       │   ├── extractors.py        # Strategy registry: pdf/docx/xlsx/pptx/text
│       │   ├── manifest.py          # per-root state for incremental indexing
│       │   ├── watcher.py           # debounced OS filesystem-event handler
│       │   ├── vectorstore.py       # Repository over ChromaDB
│       │   ├── llm.py               # Ollama chat + embedding singletons
│       │   ├── indexer.py           # the ingestion pipeline
│       │   ├── summarizer.py        # bottom-up map-reduce folder summaries
│       │   ├── cleanup.py           # duplicates, stale, junk, empty
│       │   └── filing_agent.py      # LangGraph state machine
│       └── services/
│           ├── container.py         # composition root
│           ├── jobs.py              # async job registry
│           ├── rag.py               # query planning + retrieval + generation
│           └── watch.py             # per-root live watchers + debounce
├── frontend/
│   ├── next.config.ts               # /api rewrite to FastAPI
│   ├── app/{layout,page}.tsx
│   ├── components/{IndexBar,DirectoryTree,ChatPanel,CleanupPanel,FilingPanel,ui}.tsx
│   └── lib/{api,types}.ts           # typed client + SSE parser
└── scripts/make_sample_tree.py      # generates a demo folder
```

---

## 2. Design decisions worth knowing

**Incremental indexing.** A JSON manifest per root records `(size, mtime)` per
file. A second index run diffs the filesystem against it and only extracts and
embeds what changed. Embedding is the wall-clock bottleneck (one HTTP round trip
per batch of 32 chunks), so this turns a re-index from minutes into seconds.
`force=true` rebuilds from scratch.

**Contextualised chunks.** The text sent to the embedding model is prefixed with
the file name and folder; the text stored for display is the raw chunk. A chunk
reading "Q3 was up 14%" is unfindable by "quarterly revenue report" unless the
identifying words from its filename are part of the embedded text.

**Hierarchical summaries.** Folders are summarised in post-order, so a parent
summarises its children's summaries rather than re-reading their files. Cost is
O(directories) LLM calls with a bounded prompt each, instead of one impossible
call over the whole tree.

**Cleanup is deterministic.** No LLM decides whether two files are identical.
Files are grouped by size, singleton groups dropped, survivors split by a 4 KiB
head read, and only the remainder fully hashed. The LLM only narrates findings
that were already computed.

**The filing agent cannot hallucinate a path.** Candidate folders are derived
from the incoming file's nearest neighbours in the index. The LLM picks one *by
index number*; the choice is validated against the candidate list, and anything
out of range falls back to the top-ranked candidate. The move itself is a
separate, explicit API call that re-checks containment and never overwrites.

**Design patterns used.** Strategy + Registry (extractors), Repository (vector
store), Facade (indexer), Singleton via `lru_cache` (settings, LLM clients),
composition-root dependency injection (`container.py`), State machine
(LangGraph filing agent), Producer/consumer with cooperative cancellation
(job manager), Observer (OS filesystem events -> debounced re-index).

---

## 3. Running the demo

### Prerequisites

- Python 3.11+
- Node.js 20+
- [Ollama](https://ollama.com/download)

### Step 1 — models

```bash
ollama serve            # leave running (installers usually start it for you)
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

On a laptop without a GPU, `qwen2.5:3b` answers noticeably faster. Set
`AIS_CHAT_MODEL=qwen2.5:3b` in `backend/.env`.

`nomic-embed-text` is required — it is the embedding model, and swapping it
later invalidates every stored vector (different models produce incompatible
vector spaces), so a re-index with `force=true` is needed after a change.

### Step 2 — backend

```bash
cd backend
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env          # optional; defaults work out of the box

uvicorn app.main:app --reload --port 8000
```

Verify: <http://127.0.0.1:8000/api/health> should report
`"status": "ok"`. If it says `degraded`, the response names the missing piece.
Interactive API docs are at <http://127.0.0.1:8000/docs>.

### Step 3 — frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>. The dev server proxies `/api/*` to port 8000, so
the browser only ever sees one origin and CORS never comes into play.

### Step 4 — sample data

In a third terminal (or reuse the backend one):

```bash
python scripts/make_sample_tree.py ./sample_data
```

This writes a deliberately messy tree: client contracts, invoices, specs,
runbooks, plus an exact duplicate, a versioned near-duplicate, OS junk, a
zero-byte file, a two-year-stale file, and one unfiled document in `Inbox/`.

### Step 5 — try it

1. Paste the **absolute** path of `sample_data` into the top bar and click
   **Index**. Watch the progress bar; the first run also writes folder
   summaries, which is the slow part.
2. Ask in the chat panel:
   - *"What are the payment terms for Acme?"* — cites the contract and invoices
   - *"Which folder holds the client contracts?"* — returns a folder, via the summaries
   - *"Find the spec about ranking"* — filename-driven retrieval
   - *"Show me the PDFs from last month"* — exercises the metadata filter
3. **Tools → Cleanup → Analyse folder.** The duplicate pair, the `(1)` revision
   and the junk files should all appear, with a reclaimable-space total.
4. **Tools → Filing.** Paste the full path to
   `sample_data/Inbox/unsorted-vendor-agreement.md`. The agent should propose
   `Clients/...` and explain why. Click **Move file here** to apply.
5. Click **Index** again — almost everything reports as unchanged.
6. Notice the badge next to the Ollama status now reads **live**: the folder
   is being watched. Drop a new `.txt` file into `sample_data` (or edit an
   existing one) without touching the UI. Within a couple of seconds the
   badge flips to **reindexing...**, then back to **live**, and the tree
   refreshes on its own. Ask about the new file's content in chat — no
   manual re-index needed. Click the badge to turn watching off or on.

---

## 4. Configuration

Every setting is an environment variable prefixed `AIS_`, or a line in
`backend/.env`. See `backend/.env.example` for the annotated list. The ones that
matter most:

| Variable | Default | Effect |
| --- | --- | --- |
| `AIS_CHAT_MODEL` | `llama3.1:8b` | Answer quality vs. speed |
| `AIS_EMBEDDING_MODEL` | `nomic-embed-text` | Changing it invalidates the index |
| `AIS_ALLOWED_ROOTS` | *(empty)* | Semicolon-delimited allow-list of indexable paths |
| `AIS_MAX_FILES_PER_INDEX` | `20000` | Hard cap per run |
| `AIS_SUMMARIZE_FOLDERS` | `true` | Set `false` to make the first index much faster |
| `AIS_RETRIEVAL_TOP_K` | `8` | Sources given to the model |
| `AIS_STALE_DAYS` | `365` | Stale-file threshold |
| `AIS_WATCH_DEBOUNCE_SECONDS` | `2.5` | Quiet period before a live watcher re-indexes |
| `AIS_WATCH_MAX_CONSECUTIVE_ERRORS` | `5` | Auto-disables a watcher that keeps failing |

Index data lives in `backend/.data/` (Chroma database plus one JSON manifest per
root). Deleting that directory resets everything.

---

## 5. Limitations

This is an MVP. Known gaps, in rough priority order:

- **Single-process job registry.** Jobs live in memory, so running uvicorn with
  `--workers > 1` breaks progress polling. A shared store is needed first.
- **No authentication.** Bind to `127.0.0.1` only. `AIS_ALLOWED_ROOTS` limits
  which directories can be indexed but is not a substitute for auth.
- **Change detection uses `(size, mtime)`,** so an edit preserving both is
  missed until a `force=true` rebuild.
- **No OCR.** Scanned PDFs with no text layer index as metadata only.
- **Dense retrieval only.** Hybrid BM25 + vector search and cross-encoder
  re-ranking are the obvious next quality win.
- **Filing handles one file at a time,** with no batch mode and no undo beyond
  the fact that the original is moved rather than copied.
- **Watchers are in-memory, per-process,** like the job registry -- restarting
  the backend forgets which roots were being watched. The frontend re-enables
  watching automatically the next time that root is indexed. A watcher that
  fails `AIS_WATCH_MAX_CONSECUTIVE_ERRORS` times in a row (e.g. the folder
  was deleted) turns itself off rather than retrying forever.
- **A network failure mid-chat now surfaces as a message instead of a stuck
  UI.** Earlier, a dropped connection during `/api/chat` (backend restart,
  Ollama unreachable mid-stream) threw out of an unguarded `fetch()` and left
  the chat panel permanently "busy" with no visible error. Both the fetch
  call and its caller now catch that case and reset the UI.
