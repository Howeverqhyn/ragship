"""Request and response models."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    embedding_provider: str
    llm_mode: str
    version: str


class IngestResponse(BaseModel):
    source: str
    chunks: int
    ids: list[str]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    top_k: int | None = Field(default=None, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty")
        return stripped


class SourceHit(BaseModel):
    chunk_id: str
    source: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    mode: str
    sources: list[SourceHit]
