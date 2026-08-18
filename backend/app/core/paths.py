"""Path normalisation and containment checks.

Every filesystem path that arrives from the network crosses this module first.
The invariant we enforce is *containment*: a resolved path must live under an
approved root. Resolving before comparing is what defeats both ``..`` traversal
and symlink escapes, since ``Path.resolve()`` collapses ``..`` segments and
follows links to their real target.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from app.config import Settings


class PathSecurityError(ValueError):
    """Raised when a path is malformed or escapes its permitted root."""


def normalize_root(raw: str, settings: Settings) -> Path:
    """Validate a user-supplied root directory.

    Raises:
        PathSecurityError: if the path is missing, not a directory, or outside
            the configured ``allowed_roots`` allow-list.
    """
    candidate = (raw or "").strip().strip('"').strip("'")
    if not candidate:
        raise PathSecurityError("A folder path is required.")

    expanded = Path(os.path.expandvars(candidate)).expanduser()
    try:
        resolved = expanded.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PathSecurityError(f"Cannot access path: {candidate}") from exc

    if not resolved.is_dir():
        raise PathSecurityError(f"Not a directory: {resolved}")

    allowed = settings.allowed_root_paths
    if allowed and not any(is_within(resolved, root) for root in allowed):
        raise PathSecurityError(
            f"Path {resolved} is outside the configured allow-list."
        )
    return resolved


def is_within(child: Path, parent: Path) -> bool:
    """True when ``child`` is ``parent`` or lives beneath it."""
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def resolve_within(root: Path, raw: str) -> Path:
    """Resolve ``raw`` and assert it stays inside ``root``.

    Used for any path the client names *after* a root has been approved, e.g.
    the destination directory chosen by the filing agent.
    """
    candidate = Path(os.path.expandvars((raw or "").strip())).expanduser()
    absolute = candidate if candidate.is_absolute() else root / candidate

    # strict=False: the target may not exist yet (a directory we are about to
    # create). We still resolve so that any ".." or symlink is collapsed.
    resolved = absolute.resolve(strict=False)
    if not is_within(resolved, root.resolve(strict=False)):
        raise PathSecurityError(f"{resolved} escapes the indexed root {root}.")
    return resolved


def root_id(root: Path) -> str:
    """Stable, filesystem-safe identifier for a root directory.

    Case-folded because Windows and macOS default to case-insensitive volumes,
    so ``C:\\Docs`` and ``c:\\docs`` must map to the same index.
    """
    digest = hashlib.sha1(str(root).casefold().encode("utf-8")).hexdigest()
    return digest[:16]


def collection_name(root: Path) -> str:
    """Chroma collection names must be alphanumeric/underscore/hyphen."""
    return f"root_{root_id(root)}"


def unique_destination(directory: Path, filename: str) -> Path:
    """Return a non-colliding path inside ``directory``.

    Mirrors the OS convention of ``report (1).pdf``. Never overwrites, which is
    the single most important safety property of the auto-filing feature.
    """
    target = directory / filename
    if not target.exists():
        return target

    stem, suffix = target.stem, target.suffix
    for counter in range(1, 1000):
        candidate = directory / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
    raise PathSecurityError(f"Could not find a free filename for {filename}.")


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0 or unit == "TB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"
