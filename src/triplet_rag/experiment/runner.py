"""End-to-end experiment orchestrator.

Phases run in sequence; each writes a `_SUCCESS.json` marker and is skipped
when its inputs are unchanged. Models are spun up only when needed (via
ManagedLLM / ManagedEmbedder) and torn down before the next phase.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from ..config import ExperimentConfig, IndexingStrategy, InferenceStrategy
from ..data.chunking import chunk_corpus
from ..data.loaders import get_loader
from ..evaluate import (
    aggregate_and_persist,
    compute_lexical_metrics,
    compute_ragas_metrics,
    compute_retrieval_metrics,
)
from ..index import build_index_for_strategy, index_exists, load_bundle, save_bundle
from ..infer import run_inference
from ..models import ManagedEmbedder, ManagedLLM
from ..preprocess import build_triplets, filter_triplets, generate_questions
from ..settings import get_settings
from ..utils.io import (
    has_success,
    read_json,
    read_jsonl,
    read_parquet,
    touch_success,
    write_json,
    write_npy,
    write_parquet,
)
from ..utils.logging import setup_logging
from ..utils.seeds import set_global_seed
from .registry import register_experiment


@dataclass
class _PathSet:
    raw_dir: Path
    art_dir: Path
    chunks_path: Path
    questions_path: Path
    triplets_path: Path
    chunk_emb_path: Path
    question_emb_path: Path
    index_dir: Path
    chunk_only_index_dir: Path
    exp_dir: Path
    pred_path: Path
    metrics_dir: Path
    status_path: Path

    @classmethod
    def from_cfg(cls, cfg: ExperimentConfig) -> _PathSet:
        s = get_settings()
        art = cfg.artifact_dir(s.storage_dir)
        idx = cfg.index_dir(s.storage_dir)
        # The chunks-only index is reused for triplet building and as a "fresh" index
        chunk_only_dir = (
            s.storage_dir
            / "indices"
            / f"{cfg.preprocessing_hash}_{IndexingStrategy.CHUNKS_ONLY.value}_chunkonly"
        )
        exp = cfg.experiment_dir(s.storage_dir)
        return cls(
            raw_dir=s.storage_dir / "raw",
            art_dir=art,
            chunks_path=art / "chunks.parquet",
            questions_path=art / "questions.parquet",
            triplets_path=art / "triplets.parquet",
            chunk_emb_path=art / "embeddings" / "chunks.npy",
            question_emb_path=art / "embeddings" / "questions.npy",
            index_dir=idx,
            chunk_only_index_dir=chunk_only_dir,
            exp_dir=exp,
            pred_path=exp / "predictions.jsonl",
            metrics_dir=exp / "metrics",
            status_path=exp / "status.json",
        )


def _set_status(path: Path, status: str, **extra) -> None:
    payload = {
        "status": status,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        **extra,
    }
    write_json(payload, path)


def _phase_success(art_dir: Path, name: str) -> Path:
    return art_dir / f"_SUCCESS_{name}.json"


def _needs_questions(cfg: ExperimentConfig) -> bool:
    """Phases 2-5 only run if the indexing strategy or filtering needs questions/triplets."""
    needs_q = cfg.indexer.indexing_strategy in (
        IndexingStrategy.QUESTIONS_ONLY,
        IndexingStrategy.CHUNKS_AND_QUESTIONS,
        IndexingStrategy.TRIPLETS,
        IndexingStrategy.QA_PAIRS,
    )
    return needs_q


def _needs_triplets(cfg: ExperimentConfig) -> bool:
    return cfg.indexer.indexing_strategy in (IndexingStrategy.TRIPLETS, IndexingStrategy.QA_PAIRS)


TRIPLET_ANSWER_SOURCE = "seed_answer"


def _triplets_current(paths: _PathSet) -> bool:
    success_path = _phase_success(paths.art_dir, "triplets")
    if not has_success(success_path) or not paths.triplets_path.exists():
        return False
    try:
        marker = read_json(success_path)
    except Exception:
        return False
    return marker.get("answer_source") == TRIPLET_ANSWER_SOURCE


# ----- Phase implementations -----


def phase_ingest(cfg: ExperimentConfig, paths: _PathSet, force: bool) -> None:
    loader = get_loader(cfg.dataset, paths.raw_dir)
    loader.load(force=force)


def phase_chunk(cfg: ExperimentConfig, paths: _PathSet, force: bool) -> None:
    if not force and has_success(_phase_success(paths.art_dir, "chunk")):
        logger.info("[phase: chunk] skipped (success marker present)")
        return
    loader = get_loader(cfg.dataset, paths.raw_dir)
    corpus = read_parquet(loader.corpus_path)
    chunks = chunk_corpus(corpus, cfg.chunking)
    paths.chunks_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(chunks, paths.chunks_path)
    write_json(cfg.chunking.model_dump(), paths.art_dir / "chunking.json")
    touch_success(_phase_success(paths.art_dir, "chunk"), {"n_chunks": len(chunks)})


def phase_generate_questions(cfg: ExperimentConfig, paths: _PathSet, force: bool) -> None:
    if not _needs_questions(cfg):
        logger.info("[phase: generate_questions] not needed for this strategy")
        return
    if not force and has_success(_phase_success(paths.art_dir, "questions")):
        logger.info("[phase: generate_questions] skipped")
        return
    chunks = read_parquet(paths.chunks_path)
    lifecycle_log = paths.exp_dir / "logs" / "model_lifecycle.log"
    question_gen_concurrency = get_settings().question_gen_concurrency
    if question_gen_concurrency is not None:
        logger.info(f"[phase: generate_questions] concurrency={question_gen_concurrency}")
    with ManagedLLM(cfg.generator, lifecycle_log=lifecycle_log) as teacher:
        questions = asyncio.run(
            generate_questions(
                chunks,
                teacher,
                cfg.generator,
                cfg.preprocessing,
                concurrency=question_gen_concurrency,
            )
        )
    write_parquet(questions, paths.questions_path)
    touch_success(_phase_success(paths.art_dir, "questions"), {"n_questions": len(questions)})


def phase_embed(cfg: ExperimentConfig, paths: _PathSet, force: bool) -> None:
    chunks_done = (
        not force
        and has_success(_phase_success(paths.art_dir, "embed_chunks"))
        and paths.chunk_emb_path.exists()
    )
    questions_done = (
        not force
        and has_success(_phase_success(paths.art_dir, "embed_questions"))
        and paths.question_emb_path.exists()
    )
    needs_q = _needs_questions(cfg)
    if chunks_done and (questions_done or not needs_q):
        logger.info("[phase: embed] skipped")
        return

    chunks = read_parquet(paths.chunks_path)
    questions = read_parquet(paths.questions_path) if needs_q else None
    lifecycle_log = paths.exp_dir / "logs" / "model_lifecycle.log"
    with ManagedEmbedder(cfg.embedder, lifecycle_log=lifecycle_log) as embedder:
        if not chunks_done:
            chunk_emb = embedder.embed(chunks["text"].tolist())
            write_npy(chunk_emb, paths.chunk_emb_path)
            touch_success(_phase_success(paths.art_dir, "embed_chunks"), {"n": len(chunk_emb)})
        if needs_q and not questions_done:
            assert questions is not None
            q_emb = embedder.embed(questions["question"].tolist())
            write_npy(q_emb, paths.question_emb_path)
            touch_success(_phase_success(paths.art_dir, "embed_questions"), {"n": len(q_emb)})


def phase_build_chunk_only_index(cfg: ExperimentConfig, paths: _PathSet, force: bool) -> None:
    if not force and index_exists(paths.chunk_only_index_dir):
        return
    chunks = read_parquet(paths.chunks_path)
    chunk_emb = np.load(paths.chunk_emb_path)
    bundle = build_index_for_strategy(
        IndexingStrategy.CHUNKS_ONLY,
        cfg.indexer,
        chunks=chunks,
        questions=None,
        triplets=None,
        chunk_emb=chunk_emb,
        question_emb=None,
    )
    save_bundle(bundle, paths.chunk_only_index_dir)


def phase_build_triplets(cfg: ExperimentConfig, paths: _PathSet, force: bool) -> bool:
    if not _needs_triplets(cfg):
        logger.info("[phase: triplets] not needed")
        return False
    if not force and _triplets_current(paths):
        logger.info("[phase: triplets] skipped")
        return False

    chunks = read_parquet(paths.chunks_path)
    questions = read_parquet(paths.questions_path)
    chunk_index = load_bundle(paths.chunk_only_index_dir)
    q_emb = np.load(paths.question_emb_path)

    lifecycle_log = paths.exp_dir / "logs" / "model_lifecycle.log"
    has_seed_answers = (
        "seed_answer" in questions.columns
        and questions["seed_answer"].fillna("").astype(str).str.strip().all()
    )
    if has_seed_answers:
        logger.info("[phase: triplets] using seed answers from generated questions")
        triplets = asyncio.run(
            build_triplets(
                questions,
                chunks,
                chunk_index,
                q_emb,
                teacher=None,
                teacher_cfg=cfg.generator,
                contexts_per_question=cfg.budget.per_triplet_contexts,
                answer_prompt=cfg.preprocessing.answer_prompt,
            )
        )
    else:
        logger.info("[phase: triplets] missing seed answers; falling back to generator answers")
        with ManagedLLM(cfg.generator, lifecycle_log=lifecycle_log) as teacher:
            triplets = asyncio.run(
                build_triplets(
                    questions,
                    chunks,
                    chunk_index,
                    q_emb,
                    teacher=teacher,
                    teacher_cfg=cfg.generator,
                    contexts_per_question=cfg.budget.per_triplet_contexts,
                    answer_prompt=cfg.preprocessing.answer_prompt,
                )
            )

    if cfg.filtering.enabled:
        judge_cfg = cfg.filtering.judge_model or cfg.generator
        with ManagedLLM(judge_cfg, lifecycle_log=lifecycle_log) as judge:
            triplets = asyncio.run(filter_triplets(triplets, cfg.filtering, judge))

    write_parquet(triplets, paths.triplets_path)
    touch_success(
        _phase_success(paths.art_dir, "triplets"),
        {"n_triplets": len(triplets), "answer_source": TRIPLET_ANSWER_SOURCE},
    )
    return True


def phase_build_index(cfg: ExperimentConfig, paths: _PathSet, force: bool) -> None:
    if not force and index_exists(paths.index_dir):
        logger.info("[phase: build_index] skipped")
        return
    chunks = read_parquet(paths.chunks_path)
    questions = read_parquet(paths.questions_path) if paths.questions_path.exists() else None
    triplets = read_parquet(paths.triplets_path) if paths.triplets_path.exists() else None
    chunk_emb = np.load(paths.chunk_emb_path) if paths.chunk_emb_path.exists() else None
    q_emb = np.load(paths.question_emb_path) if paths.question_emb_path.exists() else None
    if cfg.indexer.indexing_strategy == IndexingStrategy.TRIPLETS and triplets is not None:
        # filter to kept triplets
        triplets = triplets[triplets["kept"].astype(bool)].reset_index(drop=True)
    bundle = build_index_for_strategy(
        cfg.indexer.indexing_strategy,
        cfg.indexer,
        chunks=chunks,
        questions=questions,
        triplets=triplets,
        chunk_emb=chunk_emb,
        question_emb=q_emb,
    )
    save_bundle(bundle, paths.index_dir)


def phase_run_inference(cfg: ExperimentConfig, paths: _PathSet, force: bool) -> None:
    if not force and paths.pred_path.exists():
        logger.info("[phase: inference] predictions exist, skipping")
        return
    loader = get_loader(cfg.dataset, paths.raw_dir)
    queries = read_parquet(loader.queries_path)
    chunks = read_parquet(paths.chunks_path)
    triplets = read_parquet(paths.triplets_path) if paths.triplets_path.exists() else None
    bundle = load_bundle(paths.index_dir)
    fresh_bundle = (
        load_bundle(paths.chunk_only_index_dir) if cfg.inference.include_fresh_contexts else None
    )

    lifecycle_log = paths.exp_dir / "logs" / "model_lifecycle.log"
    paths.exp_dir.mkdir(parents=True, exist_ok=True)
    (paths.exp_dir / "logs").mkdir(parents=True, exist_ok=True)

    with ManagedEmbedder(cfg.embedder, lifecycle_log=lifecycle_log) as embedder:
        student_ctx = (
            ManagedLLM(cfg.student, lifecycle_log=lifecycle_log)
            if cfg.inference.strategy != InferenceStrategy.RETRIEVAL_ONLY
            else None
        )
        if student_ctx is None:
            run_inference(
                cfg,
                queries=queries,
                chunks=chunks,
                triplets=triplets,
                embedder=embedder,
                bundle=bundle,
                student=None,
                fresh_bundle=fresh_bundle,
            )
        else:
            with student_ctx as student:
                run_inference(
                    cfg,
                    queries=queries,
                    chunks=chunks,
                    triplets=triplets,
                    embedder=embedder,
                    bundle=bundle,
                    student=student,
                    fresh_bundle=fresh_bundle,
                )


def phase_compute_metrics(cfg: ExperimentConfig, paths: _PathSet, force: bool) -> None:
    if not force and (paths.metrics_dir / "aggregate.json").exists():
        logger.info("[phase: metrics] skipped")
        return

    predictions = list(read_jsonl(paths.pred_path))
    loader = get_loader(cfg.dataset, paths.raw_dir)
    qrels = read_parquet(loader.qrels_path)
    chunks = read_parquet(paths.chunks_path)

    retrieval_agg, retrieval_pq = compute_retrieval_metrics(
        predictions, qrels, cfg.metrics.retrieval_metrics, chunks
    )
    lexical_agg, lexical_pq = compute_lexical_metrics(predictions, cfg.metrics.generation_lexical)
    judge_agg, judge_pq = compute_ragas_metrics(predictions, cfg.metrics)

    aggregate_and_persist(
        out_dir=paths.metrics_dir,
        retrieval_pq=retrieval_pq,
        lexical_pq=lexical_pq,
        judge_pq=judge_pq,
        bootstrap_n=cfg.metrics.bootstrap_n,
        bootstrap_seed=cfg.metrics.bootstrap_seed,
    )

    # Convenience: write headline numbers as a simple flat dict too
    headline = {**retrieval_agg, **lexical_agg, **judge_agg}
    write_json(headline, paths.metrics_dir / "headline.json")


# ----- Top-level run -----


def run_experiment(cfg: ExperimentConfig, force: bool = False) -> Path:
    if force:
        cfg = cfg.model_copy(update={"created_at": datetime.utcnow().isoformat() + "Z"})
    s = get_settings()
    paths = _PathSet.from_cfg(cfg)

    paths.exp_dir.mkdir(parents=True, exist_ok=True)
    (paths.exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    setup_logging(s.log_level, log_file=paths.exp_dir / "logs" / "run.log")
    set_global_seed(cfg.seed)

    # Persist the resolved config
    write_json(cfg.model_dump(), paths.exp_dir / "config.yaml.json")
    write_json(
        {
            "preprocessing_hash": cfg.preprocessing_hash,
            "index_hash": cfg.index_hash,
            "experiment_hash": cfg.experiment_hash,
            "artifact_dir": str(paths.art_dir),
            "index_dir": str(paths.index_dir),
        },
        paths.exp_dir / "refs.json",
    )

    _set_status(paths.status_path, "RUNNING")
    register_experiment(s.storage_dir, cfg.experiment_id, cfg.model_dump(), "RUNNING")

    try:
        logger.info(f"=== Experiment {cfg.experiment_id} ===")
        logger.info(f"preprocessing_hash={cfg.preprocessing_hash}")
        logger.info(f"index_hash={cfg.index_hash}")

        phase_ingest(cfg, paths, force)
        phase_chunk(cfg, paths, force)
        phase_generate_questions(cfg, paths, force)
        phase_embed(cfg, paths, force)
        phase_build_chunk_only_index(cfg, paths, force)
        triplets_rebuilt = phase_build_triplets(cfg, paths, force)
        phase_build_index(cfg, paths, force or triplets_rebuilt)
        phase_run_inference(cfg, paths, force)
        phase_compute_metrics(cfg, paths, force)

        _set_status(paths.status_path, "DONE")
        register_experiment(
            s.storage_dir,
            cfg.experiment_id,
            cfg.model_dump(),
            "DONE",
            aggregate_path=paths.metrics_dir / "aggregate.json",
        )
        logger.info(f"=== Experiment {cfg.experiment_id} DONE ===")
        return paths.exp_dir

    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        _set_status(paths.status_path, "FAILED", error=str(e))
        register_experiment(s.storage_dir, cfg.experiment_id, cfg.model_dump(), "FAILED")
        raise
