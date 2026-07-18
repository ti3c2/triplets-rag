"""Record types for data flowing through the pipeline.

These mirror the parquet schemas. We use pydantic for validation when reading
from sources but write to parquet via pandas for performance.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Document(_Strict):
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Query(_Strict):
    query_id: str
    query: str
    gold_answer: str | None = None
    gold_answers: list[str] = Field(default_factory=list)
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QRel(_Strict):
    query_id: str
    doc_id: str
    relevance: int = 1


class Chunk(_Strict):
    chunk_id: str
    doc_id: str
    text: str
    span_start: int = 0
    span_end: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedQuestion(_Strict):
    question_id: str
    chunk_id: str
    question: str
    seed_answer: str | None = None  # the answer extracted at generation time, if any
    generator_model: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Triplet(_Strict):
    triplet_id: str
    question_id: str
    question: str
    retrieved_chunk_ids: list[str]
    retrieved_chunk_texts: list[str]  # denormalized for prompt assembly
    teacher_answer: str
    teacher_model: str = ""
    faithfulness_score: float | None = None
    kept: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class Prediction(_Strict):
    """One row of predictions.jsonl per query."""

    query_id: str
    query: str
    strategy: str
    retrieved_ids: list[str]  # for retrieval metrics
    retrieved_texts: list[str]  # for ragas
    retrieved_triplet_ids: list[str] = Field(default_factory=list)
    prompt: str
    prediction: str
    gold_answers: list[str] = Field(default_factory=list)
    gold_doc_ids: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
