"""Central, immutable application configuration.

Loaded once per process and shared through ``get_settings()``. Every tunable
lives here so that no module has to hard-code a magic number.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / ".data"


def _split_list(raw: str | list[str] | None) -> list[str]:
    """Parse a delimited env string into a list.

    Semicolons are the primary separator because Windows paths contain
    drive-letter colons and commas are legal in file names.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [part.strip() for part in raw.split(";") if part.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        env_prefix="AIS_",
        extra="ignore",
    )

    # ---------------------------------------------------------------- server
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "http://localhost:3000;http://127.0.0.1:3000"
    log_level: str = "INFO"

    # ------------------------------------------------------------- ollama/ai
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    llm_temperature: float = 0.1
    llm_num_ctx: int = 8192
    llm_request_timeout_s: float = 180.0

    # -------------------------------------------------------------- storage
    data_dir: Path = _DEFAULT_DATA_DIR

    # --------------------------------------------------------------- safety
    # Semicolon-delimited absolute prefixes. Empty => any local path is
    # allowed, which is acceptable for a single-user local demo but should be
    # populated before this is exposed on a network interface.
    allowed_roots: str = ""

    # -------------------------------------------------------------- scanning
    max_scan_depth: int = 24
    max_files_per_index: int = 20_000
    max_file_size_bytes: int = 25 * 1024 * 1024
    follow_symlinks: bool = False
    ignored_dir_names: str = (
        ".git;.hg;.svn;node_modules;__pycache__;.venv;venv;.mypy_cache;"
        ".pytest_cache;.ruff_cache;.next;dist;build;.idea;.vscode;"
        "$RECYCLE.BIN;System Volume Information;.Trash;.cache;.terraform"
    )
    junk_file_names: str = ".DS_Store;Thumbs.db;desktop.ini;.localized"
    junk_extensions: str = ".tmp;.temp;.bak;.old;.swp;.crdownload;.part;.partial"

    # ---------------------------------------------------------------- watch
    # How long the filesystem must be quiet before a live watcher fires a
    # re-index. Editors write in bursts (temp file, then rename, then a
    # second save a moment later); one debounced run instead of one per
    # event keeps a "watched" folder from re-embedding the same file five
    # times in a row.
    watch_debounce_seconds: float = 2.5
    # A poison-pill root (permissions error, deleted mid-watch) would
    # otherwise retry forever once per debounce window.
    watch_max_consecutive_errors: int = 5

    # ------------------------------------------------------------- indexing
    chunk_size: int = 1200
    chunk_overlap: int = 150
    max_chars_per_file: int = 60_000
    embedding_batch_size: int = 32
    vector_upsert_batch_size: int = 256
    # Files above this size are not content-hashed during duplicate detection
    # unless they collide on size with another file.
    hash_read_block_bytes: int = 1024 * 1024

    # ------------------------------------------------------------- retrieval
    retrieval_top_k: int = 8
    retrieval_fetch_k: int = 40
    max_chunks_per_file: int = 3
    min_relevance_score: float = 0.15
    enable_llm_query_planner: bool = True

    # ------------------------------------------------------------ summaries
    summarize_folders: bool = True
    max_folder_summaries: int = 60
    min_files_for_summary: int = 2
    summary_sample_files: int = 12
    summary_snippet_chars: int = 600

    # -------------------------------------------------------------- cleanup
    stale_days: int = 365
    large_file_top_k: int = 15

    # --------------------------------------------------------------- filing
    filing_candidate_dirs: int = 6
    filing_confidence_threshold: float = 0.6
    filing_content_preview_chars: int = 4000

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand_data_dir(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(os.path.expandvars(value)).expanduser()
        return value

    # ------------------------------------------------------ derived helpers
    @property
    def cors_origin_list(self) -> list[str]:
        return _split_list(self.cors_origins)

    @property
    def allowed_root_paths(self) -> list[Path]:
        return [Path(p).expanduser().resolve() for p in _split_list(self.allowed_roots)]

    @property
    def ignored_dirs(self) -> frozenset[str]:
        return frozenset(name.lower() for name in _split_list(self.ignored_dir_names))

    @property
    def junk_names(self) -> frozenset[str]:
        return frozenset(name.lower() for name in _split_list(self.junk_file_names))

    @property
    def junk_exts(self) -> frozenset[str]:
        return frozenset(ext.lower() for ext in _split_list(self.junk_extensions))

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def manifest_dir(self) -> Path:
        return self.data_dir / "manifests"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.chroma_dir, self.manifest_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton.

    ``lru_cache`` gives us a thread-safe, lazily-constructed singleton without
    the boilerplate (and testability problems) of a classic ``__new__`` guard;
    ``get_settings.cache_clear()`` restores a clean slate in tests.
    """
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings_dependency = get_settings
