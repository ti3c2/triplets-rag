"""Retrievers operate on built indices.

A retriever takes embedded queries and produces a list of RetrievalResult
objects, one per query. Each result carries the items needed by downstream
inference and metric computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..config import IndexingStrategy, RetrieverConfig, TripletRetrievalMode
from ..index.store import IndexBundle, search


@dataclass
class RetrievedItem:
    item_id: str
    item_type: str  # 'chunk' | 'question' | 'triplet' | 'qa_pair'
    score: float
    text: str
    chunk_id: str | None = None
    doc_id: str | None = None
    chunk_ids: list[str] = field(default_factory=list)  # for triplet
    chunk_texts: list[str] = field(default_factory=list)
    teacher_answer: str | None = None
    question: str | None = None  # for triplet/qa_pair
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query_id: str
    items: list[RetrievedItem]


# ----- Helpers for dedupe -----


def _dedupe(items: list[RetrievedItem], key: str) -> list[RetrievedItem]:
    if key == "none":
        return items
    seen: set[str] = set()
    out: list[RetrievedItem] = []
    for it in items:
        if key == "chunk_id":
            k = it.chunk_id or it.item_id
        elif key == "doc_id":
            k = it.doc_id or it.item_id
        else:
            k = it.item_id
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


# ----- Dense retrievers -----


def retrieve_dense(
    bundle: IndexBundle,
    query_embeddings: np.ndarray,
    query_ids: list[str],
    cfg: RetrieverConfig,
    *,
    triplet_chunks: pd.DataFrame | None = None,
) -> list[RetrievalResult]:
    """Generic dense retriever for chunk / question / triplet / qa_pair items."""
    over_k = cfg.top_k * cfg.over_fetch_factor
    distances, indices = search(bundle, query_embeddings, over_k)

    id_map = bundle.id_map
    type_value = id_map["item_type"].iloc[0] if len(id_map) else "chunk"

    # Build chunk-id -> text lookup if we have triplets-as-rows and need their contexts
    chunk_text_lookup: dict[str, str] = {}
    if triplet_chunks is not None:
        chunk_text_lookup = triplet_chunks.set_index("chunk_id")["text"].to_dict()

    results: list[RetrievalResult] = []
    for q_idx, qid in enumerate(query_ids):
        items: list[RetrievedItem] = []
        for rank, (rowid_arr, score_arr) in enumerate(
            zip(indices[q_idx], distances[q_idx], strict=True)
        ):
            r = int(rowid_arr)
            if r < 0:
                continue
            row = id_map.iloc[r]
            it_type = row["item_type"]
            if it_type == "chunk":
                items.append(
                    RetrievedItem(
                        item_id=row["item_id"],
                        item_type="chunk",
                        score=float(score_arr),
                        text=row["text"],
                        chunk_id=row["item_id"],
                        doc_id=row.get("doc_id"),
                    )
                )
            elif it_type == "question":
                items.append(
                    RetrievedItem(
                        item_id=row["item_id"],
                        item_type="question",
                        score=float(score_arr),
                        text=row.get("chunk_text", ""),
                        chunk_id=row.get("chunk_id"),
                        doc_id=row.get("doc_id"),
                        question=row["text"],
                    )
                )
            elif it_type == "triplet":
                cids = (
                    list(row["chunk_ids"])
                    if isinstance(row["chunk_ids"], (list, np.ndarray))
                    else []
                )
                items.append(
                    RetrievedItem(
                        item_id=row["item_id"],
                        item_type="triplet",
                        score=float(score_arr),
                        text=row["text"],
                        chunk_ids=list(cids),
                        chunk_texts=[chunk_text_lookup.get(c, "") for c in cids],
                        teacher_answer=row.get("teacher_answer"),
                        question=row["text"],
                    )
                )
            elif it_type == "qa_pair":
                items.append(
                    RetrievedItem(
                        item_id=row["item_id"],
                        item_type="qa_pair",
                        score=float(score_arr),
                        text=row["text"],
                        teacher_answer=row.get("teacher_answer"),
                    )
                )
            else:
                logger.warning(f"Unknown item_type {it_type}; skipping")

        items = _dedupe(items, cfg.dedupe_by)[: cfg.top_k]
        results.append(RetrievalResult(query_id=qid, items=items))
    return results


def retrieve_triplet_chunk_mediated(
    chunk_bundle: IndexBundle,
    query_embeddings: np.ndarray,
    query_ids: list[str],
    cfg: RetrieverConfig,
    triplets: pd.DataFrame,
    chunks: pd.DataFrame,
) -> list[RetrievalResult]:
    """Find chunks first, then surface triplets attached to them.

    For each test query:
      1. retrieve top-(k * over_fetch_factor) chunks
      2. for each chunk in rank order, gather all triplets whose retrieved_chunk_ids
         contain that chunk
      3. dedupe by triplet_id, keep top_k triplets
    """
    if "retrieved_chunk_ids" not in triplets.columns:
        raise ValueError("triplets must have retrieved_chunk_ids column")

    # Build chunk_id -> [triplet rows] index
    chunk_to_triplets: dict[str, list[dict[str, Any]]] = {}
    for t in triplets.to_dict(orient="records"):
        for cid in t["retrieved_chunk_ids"]:
            chunk_to_triplets.setdefault(cid, []).append(t)
    chunk_text_lookup = chunks.set_index("chunk_id")["text"].to_dict()

    over_k = cfg.top_k * cfg.over_fetch_factor
    distances, indices = search(chunk_bundle, query_embeddings, over_k)
    chunk_id_by_row = chunk_bundle.id_map.set_index("rowid")["item_id"].to_dict()

    results: list[RetrievalResult] = []
    for q_idx, qid in enumerate(query_ids):
        seen_triplets: set[str] = set()
        items: list[RetrievedItem] = []
        for rank, (rowid_arr, score_arr) in enumerate(
            zip(indices[q_idx], distances[q_idx], strict=True)
        ):
            r = int(rowid_arr)
            if r < 0:
                continue
            cid = chunk_id_by_row.get(r)
            if cid is None:
                continue
            for t in chunk_to_triplets.get(cid, []):
                if t["triplet_id"] in seen_triplets:
                    continue
                seen_triplets.add(t["triplet_id"])
                tcids = list(t["retrieved_chunk_ids"])
                items.append(
                    RetrievedItem(
                        item_id=t["triplet_id"],
                        item_type="triplet",
                        score=float(score_arr),
                        text=t["question"],
                        chunk_ids=tcids,
                        chunk_texts=[chunk_text_lookup.get(c, "") for c in tcids],
                        teacher_answer=t["teacher_answer"],
                        question=t["question"],
                    )
                )
                if len(items) >= cfg.top_k:
                    break
            if len(items) >= cfg.top_k:
                break
        results.append(RetrievalResult(query_id=qid, items=items))
    return results


def retrieve(
    indexing_strategy: IndexingStrategy,
    cfg: RetrieverConfig,
    bundle: IndexBundle,
    query_embeddings: np.ndarray,
    query_ids: list[str],
    *,
    triplets: pd.DataFrame | None = None,
    chunks: pd.DataFrame | None = None,
) -> list[RetrievalResult]:
    """Top-level dispatch."""
    if indexing_strategy == IndexingStrategy.TRIPLETS:
        if cfg.triplet_retrieval_mode == TripletRetrievalMode.Q2Q:
            return retrieve_dense(bundle, query_embeddings, query_ids, cfg, triplet_chunks=chunks)
        else:
            assert triplets is not None and chunks is not None
            return retrieve_triplet_chunk_mediated(
                bundle, query_embeddings, query_ids, cfg, triplets, chunks
            )
    return retrieve_dense(bundle, query_embeddings, query_ids, cfg, triplet_chunks=chunks)
