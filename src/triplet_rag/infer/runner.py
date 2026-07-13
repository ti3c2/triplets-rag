"""Inference runner: takes the experiment config and runs all queries through
the chosen strategy, writing predictions.jsonl.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
from loguru import logger

from ..config import ExperimentConfig
from ..index.store import IndexBundle
from ..models import EmbedderClient, LLMClient
from ..retrieve.retriever import retrieve
from ..settings import get_settings
from ..utils.io import write_jsonl
from .strategies import arun_inference_for_query


def run_inference(
    cfg: ExperimentConfig,
    *,
    queries: pd.DataFrame,
    chunks: pd.DataFrame,
    triplets: pd.DataFrame | None,
    embedder: EmbedderClient,
    bundle: IndexBundle,
    student: LLMClient | None,
    fresh_bundle: IndexBundle | None = None,
) -> Path:
    """Returns the path to predictions.jsonl."""
    s = get_settings()
    exp_dir = cfg.experiment_dir(s.storage_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    pred_path = exp_dir / "predictions.jsonl"

    logger.info(
        f"Running inference: strategy={cfg.inference.strategy.value}, "
        f"queries={len(queries)}, top_k={cfg.retriever.top_k}"
    )

    # Embed all queries
    query_texts = queries["query"].tolist()
    query_ids = queries["query_id"].tolist()
    q_emb = embedder.embed(query_texts)

    # Retrieve
    results = retrieve(
        cfg.indexer.indexing_strategy,
        cfg.retriever,
        bundle,
        q_emb,
        query_ids,
        triplets=triplets,
        chunks=chunks,
    )
    fresh_results = None
    if cfg.inference.include_fresh_contexts and fresh_bundle is not None:
        from ..config import RetrieverConfig

        fresh_cfg = RetrieverConfig(
            top_k=cfg.budget.total_context_items,
            triplet_retrieval_mode=cfg.retriever.triplet_retrieval_mode,
            over_fetch_factor=cfg.retriever.over_fetch_factor,
            dedupe_by="chunk_id",
        )
        from ..config import IndexingStrategy

        fresh_results = retrieve(
            IndexingStrategy.CHUNKS_ONLY,
            fresh_cfg,
            fresh_bundle,
            q_emb,
            query_ids,
            triplets=None,
            chunks=chunks,
        )

    # Build the per-query inputs
    queries_dict = queries.set_index("query_id").to_dict(orient="index")

    # Run student calls. Remote model fanout uses async I/O bounded by llm_concurrency.
    # Retrieval-only also uses this path, but does not perform model calls.

    async def _run_one_async(idx: int):
        qid = query_ids[idx]
        meta = queries_dict[qid]
        return await arun_inference_for_query(
            query=meta["query"],
            query_id=qid,
            gold_answers=list(meta.get("gold_answers", [])) or [meta.get("gold_answer", "") or ""],
            gold_doc_ids=list(meta.get("gold_doc_ids", [])),
            retrieval=results[idx],
            inference_cfg=cfg.inference,
            budget=cfg.budget,
            student=student,
            fresh_retrieval=fresh_results[idx] if fresh_results else None,
        )

    async def _run_all_async() -> list:
        c = s.llm_concurrency
        semaphore = asyncio.Semaphore(c)
        out = []
        done = 0
        log_every = max(1, len(query_ids) // 20)

        async def _bounded(idx: int):
            async with semaphore:
                return await _run_one_async(idx)

        tasks = [asyncio.create_task(_bounded(i)) for i in range(len(query_ids))]
        for task in asyncio.as_completed(tasks):
            pred = await task
            out.append(pred)
            done += 1
            if done % log_every == 0:
                logger.info(f"inference: {done}/{len(query_ids)}")
        return out

    predictions = asyncio.run(_run_all_async())

    # Sort to deterministic order then write
    predictions.sort(key=lambda p: p.query_id)
    write_jsonl([p.model_dump() for p in predictions], pred_path)
    logger.info(f"Wrote {len(predictions)} predictions to {pred_path}")
    return pred_path
