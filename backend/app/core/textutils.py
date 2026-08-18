"""Helpers for coping with loosely-structured LLM output."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def message_text(message: Any) -> str:
    """Flatten a LangChain message (or raw chunk) into plain text.

    LangChain 1.x allows ``content`` to be either a ``str`` or a list of
    content blocks (``{"type": "text", "text": ...}``) depending on the
    provider. Normalising here keeps that detail out of every call site.
    """
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return "" if content is None else str(content)


def _scan_balanced(text: str, opener: str, closer: str) -> str | None:
    """Return the first balanced ``opener``..``closer`` span in ``text``.

    A depth counter that is string- and escape-aware. Naive ``text[find('{'):
    rfind('}')+1]`` breaks the moment a brace appears inside a JSON string
    value, which happens often when an LLM quotes a Windows path or a snippet
    of code. Runs in O(n) over the response.
    """
    start = text.find(opener)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json(text: str, expect: type = dict) -> Any | None:
    """Best-effort recovery of a JSON value from a chat completion.

    Tries, in order: the whole string, the contents of a fenced code block,
    then the first balanced brace/bracket span. Returns ``None`` rather than
    raising so callers can fall back to a deterministic code path.
    """
    if not text:
        return None

    candidates: list[str] = [text.strip()]
    for match in _FENCE_RE.finditer(text):
        candidates.append(match.group(1).strip())

    opener, closer = ("{", "}") if expect is dict else ("[", "]")
    span = _scan_balanced(text, opener, closer)
    if span:
        candidates.append(span)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, expect):
            return parsed
    return None


def truncate(text: str, limit: int, marker: str = "\n...[truncated]") -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + marker


_WS_RE = re.compile(r"[ \t\x0b\f\r]+")
_BLANKLINE_RE = re.compile(r"\n{3,}")


def collapse_whitespace(text: str) -> str:
    """Squeeze runs of spaces and blank lines.

    Extracted document text is full of layout padding; removing it typically
    cuts token counts by 20-40% with no loss of meaning, which directly
    reduces embedding and inference time.
    """
    cleaned = _WS_RE.sub(" ", text.replace("\x00", ""))
    cleaned = _BLANKLINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()
