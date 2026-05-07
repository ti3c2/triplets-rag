"""Pydantic schemas for every configurable component.

All configs are pure data and serialize to JSON. The hashes derived from them
key the artifacts/indices/experiments folders.

Layered hashes:
- preprocessing_hash = hash(dataset, chunking, generator, embedder, preprocessing_params)
  -> chunks, questions, triplets, embeddings live under this.
- index_hash = hash(preprocessing_hash, indexer, indexing_strategy)
  -> faiss index lives under this.
- experiment_hash = hash(everything above + retriever + student + inference_strategy + budget + filtering + metrics + seed)
  -> predictions, metrics, logs live under this.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .utils.hashing import stable_hash


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------- Datasets ----------


class DatasetConfig(_Frozen):
    name: Literal["squad", "natural_questions", "multihop_rag", "fixture"]
    split: str = "validation"
    max_queries: int | None = None  # for pilots
    max_documents: int | None = None
    seed: int = 42


# ---------- Chunking ----------


class ChunkingConfig(_Frozen):
    strategy: Literal["sliding_window", "sentence", "paragraph"] = "sliding_window"
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_chars: int = 50


# ---------- Models ----------


class ModelKind(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VLLM = "vllm"  # local OpenAI-compatible server
    LOCAL_HF = "local_hf"  # we launch vLLM on a HF model
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI_EMBED = "openai_embed"


class LLMConfig(_Frozen):
    """Configuration for a generative LLM (teacher, student, or judge)."""

    kind: Literal["openai", "anthropic", "vllm", "local_hf"]
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    # Only used for local_hf:
    vllm_port: int = 8000
    vllm_gpu_memory_utilization: float | None = None  # falls back to settings
    vllm_dtype: str = "auto"
    vllm_max_model_len: int | None = None
    vllm_extra_args: list[str] = Field(default_factory=list)


class EmbedderConfig(_Frozen):
    kind: Literal["sentence_transformers", "openai_embed"]
    model_name: str
    batch_size: int = 64
    normalize: bool = True
    # OpenAI embedding dim if applicable; for sbert it's read from model
    dim: int | None = None


# ---------- Index / retrieval ----------


class IndexingStrategy(str, Enum):
    CHUNKS_ONLY = "chunks_only"  # vanilla
    QUESTIONS_ONLY = "questions_only"  # extreme QuOTE
    CHUNKS_AND_QUESTIONS = "chunks_and_questions"  # full QuOTE
    TRIPLETS = "triplets"  # the proposed approach
    QA_PAIRS = "qa_pairs"  # ablation: drop contexts, keep Q+A


class TripletRetrievalMode(str, Enum):
    Q2Q = "q2q"
    CHUNK_MEDIATED = "chunk_mediated"


class IndexerConfig(_Frozen):
    kind: Literal["faiss_flat", "faiss_hnsw"] = "faiss_flat"
    metric: Literal["ip", "l2"] = "ip"  # inner product on normalized vecs == cosine
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    indexing_strategy: IndexingStrategy = IndexingStrategy.CHUNKS_ONLY


class RetrieverConfig(_Frozen):
    top_k: int = 10
    triplet_retrieval_mode: TripletRetrievalMode = TripletRetrievalMode.Q2Q
    over_fetch_factor: int = 5  # over-fetch then dedupe (QuOTE-style)
    dedupe_by: Literal["chunk_id", "doc_id", "none"] = "chunk_id"


# ---------- Inference ----------


class InferenceStrategy(str, Enum):
    RETRIEVAL_ONLY = "retrieval_only"  # no LLM call; for retrieval metrics
    VANILLA_RAG = "vanilla_rag"  # query + retrieved chunks
    TRIPLET_RAG = "triplet_rag"  # query + retrieved triplets as demos
    QA_DEMO_RAG = "qa_demo_rag"  # query + retrieved (Q,A) demos (no contexts)


class InferenceConfig(_Frozen):
    strategy: InferenceStrategy = InferenceStrategy.VANILLA_RAG
    prompt_version: str = "v1"
    include_fresh_contexts: bool = False  # for triplet_rag: also append test query's retrieved chunks


# ---------- Budget ----------


class BudgetConfig(_Frozen):
    """How much context the student gets at inference.

    For vanilla_rag: total_context_items chunks are retrieved.
    For triplet_rag: num_triplets triplets, each with per_triplet_contexts contexts.
    Default values are matched: 2 * 5 = 10.
    """

    total_context_items: int = 10  # for vanilla_rag
    num_triplets: int = 2  # for triplet_rag
    per_triplet_contexts: int = 5
    matching_mode: Literal["item", "token"] = "item"


# ---------- Filtering ----------


class FilteringConfig(_Frozen):
    enabled: bool = False
    faithfulness_threshold: float = 0.7
    judge_model: LLMConfig | None = None  # if None, reuse generator


# ---------- Preprocessing ----------


class PreprocessingConfig(_Frozen):
    num_questions_per_chunk: int = 5
    question_prompt: Literal["basic", "complex", "multihop"] = "complex"
    answer_prompt: str = "rag_default"
    max_questions_total: int | None = None  # for pilots
    dedupe_questions: bool = True
    dedupe_threshold: float = 0.95


# ---------- Metrics ----------


class MetricsConfig(_Frozen):
    retrieval_metrics: list[str] = Field(
        default_factory=lambda: ["nDCG@10", "Recall@5", "Recall@10", "Recall@20", "RR", "P@1", "P@5"]
    )
    generation_lexical: list[str] = Field(default_factory=lambda: ["em", "f1", "rouge_l"])
    use_ragas: bool = True
    ragas_metrics: list[str] = Field(
        default_factory=lambda: ["faithfulness", "answer_relevancy", "answer_correctness"]
    )
    judge_model: LLMConfig | None = None  # if None, ragas uses default
    bootstrap_n: int = 1000
    bootstrap_seed: int = 12345


# ---------- Top-level ----------


class ExperimentConfig(_Frozen):
    """The full, frozen config that defines an experiment."""

    experiment_name: str
    seed: int = 42

    dataset: DatasetConfig
    chunking: ChunkingConfig
    generator: LLMConfig
    embedder: EmbedderConfig
    indexer: IndexerConfig
    retriever: RetrieverConfig
    student: LLMConfig
    inference: InferenceConfig
    budget: BudgetConfig
    preprocessing: PreprocessingConfig
    filtering: FilteringConfig
    metrics: MetricsConfig

    # Optional notes
    notes: str = ""
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # ---------- Derived hashes ----------

    def preprocessing_payload(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.model_dump(),
            "chunking": self.chunking.model_dump(),
            "generator": self.generator.model_dump(),
            "embedder": self.embedder.model_dump(),
            "preprocessing": self.preprocessing.model_dump(),
            "filtering": self.filtering.model_dump(),
            "seed": self.seed,
        }

    def index_payload(self) -> dict[str, Any]:
        return {
            "preprocessing_hash": self.preprocessing_hash,
            "indexer": self.indexer.model_dump(),
        }

    def experiment_payload(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "preprocessing_hash": self.preprocessing_hash,
            "index_hash": self.index_hash,
            "retriever": self.retriever.model_dump(),
            "student": self.student.model_dump(),
            "inference": self.inference.model_dump(),
            "budget": self.budget.model_dump(),
            "metrics": self.metrics.model_dump(),
            "seed": self.seed,
        }

    @property
    def preprocessing_hash(self) -> str:
        return stable_hash(self.preprocessing_payload())

    @property
    def index_hash(self) -> str:
        return stable_hash(self.index_payload())

    @property
    def experiment_hash(self) -> str:
        return stable_hash(self.experiment_payload())

    @property
    def experiment_id(self) -> str:
        # Human-readable + unique
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        slug = self.experiment_name.replace("/", "_").replace(" ", "_")
        return f"{ts}_{self.experiment_hash}_{slug}"

    # ---------- Path helpers (storage roots come from settings) ----------

    def artifact_dir(self, root: Path) -> Path:
        return root / "artifacts" / self.preprocessing_hash

    def index_dir(self, root: Path) -> Path:
        return (
            root
            / "indices"
            / f"{self.preprocessing_hash}_{self.indexer.indexing_strategy.value}_{self.index_hash}"
        )

    def experiment_dir(self, root: Path) -> Path:
        return root / "experiments" / self.experiment_id

    @model_validator(mode="after")
    def _check_budget_consistency(self) -> ExperimentConfig:
        if self.budget.matching_mode == "item":
            if (
                self.inference.strategy == InferenceStrategy.TRIPLET_RAG
                and self.budget.num_triplets * self.budget.per_triplet_contexts
                != self.budget.total_context_items
            ):
                # not fatal; we just warn via a stored note
                object.__setattr__(
                    self,
                    "notes",
                    (
                        self.notes
                        + f" [warning: triplet budget {self.budget.num_triplets}*{self.budget.per_triplet_contexts}"
                        f" != total {self.budget.total_context_items}]"
                    ).strip(),
                )
        return self
