"""Request/response contracts.

Pydantic models are the API's validation boundary: anything that reaches a
handler has already been type-checked and range-checked, so handlers contain
business logic only. They also generate the OpenAPI schema served at ``/docs``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    path: str = Field(..., min_length=1, description="Absolute folder path to index.")
    force: bool = Field(False, description="Rebuild from scratch, ignoring the manifest.")
    summarize: bool | None = Field(
        None, description="Override the folder-summary setting for this run."
    )


class ChatRequest(BaseModel):
    path: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=30)
    stream: bool = True


class SearchRequest(BaseModel):
    path: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=50)


class FilingRequest(BaseModel):
    path: str = Field(..., min_length=1, description="Indexed root folder.")
    file: str = Field(..., min_length=1, description="File to be filed.")
    apply: bool = Field(False, description="Actually move the file.")
    destination: str | None = Field(
        None, description="Override the agent's chosen destination folder."
    )
    filename: str | None = Field(None, description="Override the target filename.")


class WatchRequest(BaseModel):
    path: str = Field(..., min_length=1, description="Indexed root folder.")
    enabled: bool = Field(..., description="Start (true) or stop (false) live watching.")


class JobResponse(BaseModel):
    id: str
    kind: str
    target: str
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    progress: float
    message: str
    result: dict[str, Any] | None = None
    error: str = ""
    createdAt: float
    finishedAt: float | None = None
    elapsedSeconds: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama: dict[str, Any]
    indexedRoots: int
    chatModel: str
    embeddingModel: str
    dataDir: str


class ErrorResponse(BaseModel):
    detail: str
