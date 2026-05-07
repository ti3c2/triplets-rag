"""End-to-end integration test on the fixture corpus.

Verifies:
- All phases run without error
- Files land in the expected storage layout
- Artifacts are shared across experiments with the same preprocessing_hash
- Skipping works on re-runs
"""

from __future__ import annotations

from pathlib import Path

import pytest

from triplet_rag.config import (
    BudgetConfig,
    ChunkingConfig,
    DatasetConfig,
    EmbedderConfig,
    ExperimentConfig,
    FilteringConfig,
    IndexerConfig,
    IndexingStrategy,
    InferenceConfig,
    InferenceStrategy,
    LLMConfig,
    MetricsConfig,
    PreprocessingConfig,
    RetrieverConfig,
    TripletRetrievalMode,
)
from triplet_rag.experiment import run_experiment
from triplet_rag.utils.io import read_json, read_jsonl, read_parquet


def _vanilla_cfg(name: str = "fixture_vanilla") -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=name,
        seed=42,
        dataset=DatasetConfig(name="fixture"),
        chunking=ChunkingConfig(chunk_size=200, chunk_overlap=30, min_chunk_chars=20),
        generator=LLMConfig(kind="openai", model_name="gpt-4o-mini"),
        embedder=EmbedderConfig(kind="sentence_transformers", model_name="stub-sbert"),
        indexer=IndexerConfig(indexing_strategy=IndexingStrategy.CHUNKS_ONLY),
        retriever=RetrieverConfig(top_k=2, over_fetch_factor=3),
        student=LLMConfig(kind="openai", model_name="gpt-4o-mini"),
        inference=InferenceConfig(strategy=InferenceStrategy.VANILLA_RAG),
        budget=BudgetConfig(total_context_items=2, num_triplets=1, per_triplet_contexts=1),
        preprocessing=PreprocessingConfig(num_questions_per_chunk=2, question_prompt="basic"),
        filtering=FilteringConfig(enabled=False),
        metrics=MetricsConfig(use_ragas=False, retrieval_metrics=["nDCG@5", "R@5", "RR"]),
    )


def _triplet_cfg(name: str = "fixture_triplet", mode: TripletRetrievalMode = TripletRetrievalMode.Q2Q) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=name,
        seed=42,
        dataset=DatasetConfig(name="fixture"),
        chunking=ChunkingConfig(chunk_size=200, chunk_overlap=30, min_chunk_chars=20),
        generator=LLMConfig(kind="openai", model_name="gpt-4o-mini"),
        embedder=EmbedderConfig(kind="sentence_transformers", model_name="stub-sbert"),
        indexer=IndexerConfig(indexing_strategy=IndexingStrategy.TRIPLETS),
        retriever=RetrieverConfig(top_k=2, over_fetch_factor=5, triplet_retrieval_mode=mode),
        student=LLMConfig(kind="openai", model_name="gpt-4o-mini"),
        inference=InferenceConfig(strategy=InferenceStrategy.TRIPLET_RAG),
        budget=BudgetConfig(total_context_items=2, num_triplets=2, per_triplet_contexts=1),
        preprocessing=PreprocessingConfig(num_questions_per_chunk=2, question_prompt="basic"),
        filtering=FilteringConfig(enabled=False),
        metrics=MetricsConfig(use_ragas=False, retrieval_metrics=["nDCG@5", "R@5", "RR"]),
    )


def test_vanilla_end_to_end(stub_llm_chat, stub_embedder, _isolated_storage):
    cfg = _vanilla_cfg()
    exp_dir = run_experiment(cfg)

    # Storage layout
    assert exp_dir.exists()
    assert (exp_dir / "config.yaml.json").exists()
    assert (exp_dir / "refs.json").exists()
    assert (exp_dir / "predictions.jsonl").exists()
    assert (exp_dir / "metrics" / "aggregate.json").exists()
    assert (exp_dir / "status.json").exists()

    # Status is DONE
    status = read_json(exp_dir / "status.json")
    assert status["status"] == "DONE"

    # Artifacts under the shared preprocessing_hash dir
    storage = _isolated_storage
    art_dir = storage / "artifacts" / cfg.preprocessing_hash
    assert (art_dir / "chunks.parquet").exists()
    # vanilla doesn't need questions or triplets
    chunks_df = read_parquet(art_dir / "chunks.parquet")
    assert len(chunks_df) >= 1

    # Predictions: one row per query in the fixture (3)
    preds = list(read_jsonl(exp_dir / "predictions.jsonl"))
    assert len(preds) == 3
    for p in preds:
        assert p["strategy"] == "vanilla_rag"
        assert p["prediction"]
        assert p["retrieved_ids"]

    # Aggregate metrics: at least retrieval and lexical present
    agg = read_json(exp_dir / "metrics" / "aggregate.json")
    assert any("R@5" in k or "nDCG" in k for k in agg)
    assert any(k.lower() in ("em", "f1", "rouge_l") for k in agg)

    # Registry
    reg = storage / "registry.parquet"
    assert reg.exists()


def test_triplet_q2q_end_to_end(stub_llm_chat, stub_embedder, _isolated_storage):
    cfg = _triplet_cfg(mode=TripletRetrievalMode.Q2Q)
    exp_dir = run_experiment(cfg)
    storage = _isolated_storage

    # Triplet artifacts must exist
    art_dir = storage / "artifacts" / cfg.preprocessing_hash
    assert (art_dir / "questions.parquet").exists()
    assert (art_dir / "triplets.parquet").exists()
    assert (art_dir / "embeddings" / "chunks.npy").exists()
    assert (art_dir / "embeddings" / "questions.npy").exists()

    # Predictions
    preds = list(read_jsonl(exp_dir / "predictions.jsonl"))
    assert len(preds) == 3
    for p in preds:
        assert p["strategy"] == "triplet_rag"
        assert p["retrieved_triplet_ids"]


def test_triplet_chunk_mediated_end_to_end(stub_llm_chat, stub_embedder, _isolated_storage):
    cfg = _triplet_cfg(name="fixture_triplet_chunkmed", mode=TripletRetrievalMode.CHUNK_MEDIATED)
    exp_dir = run_experiment(cfg)
    preds = list(read_jsonl(exp_dir / "predictions.jsonl"))
    assert len(preds) == 3
    for p in preds:
        assert p["strategy"] == "triplet_rag"


def test_artifacts_are_shared_across_experiments(stub_llm_chat, stub_embedder, _isolated_storage):
    """Two experiments with the same preprocessing config should share artifacts."""
    storage = _isolated_storage

    # Run first
    cfg_a = _triplet_cfg(name="exp_a")
    exp_dir_a = run_experiment(cfg_a)
    art_dir = storage / "artifacts" / cfg_a.preprocessing_hash
    triplets_path = art_dir / "triplets.parquet"
    assert triplets_path.exists()
    mtime_first = triplets_path.stat().st_mtime

    # Run second with a different student (same preprocessing_hash)
    cfg_b = _triplet_cfg(name="exp_b")
    cfg_b_dict = cfg_b.model_dump()
    cfg_b_dict["student"] = LLMConfig(
        kind="openai", model_name="gpt-4o", temperature=0.0
    ).model_dump()
    cfg_b_dict["inference"] = InferenceConfig(
        strategy=InferenceStrategy.QA_DEMO_RAG
    ).model_dump()
    cfg_b2 = ExperimentConfig(**cfg_b_dict)
    assert cfg_a.preprocessing_hash == cfg_b2.preprocessing_hash
    assert cfg_a.experiment_hash != cfg_b2.experiment_hash

    exp_dir_b = run_experiment(cfg_b2)
    # Triplets file unchanged
    assert triplets_path.stat().st_mtime == mtime_first
    # Different experiment dirs
    assert exp_dir_a != exp_dir_b


def test_resumes_when_predictions_exist(stub_llm_chat, stub_embedder, _isolated_storage):
    """When the experiment dir already has predictions.jsonl, re-running into
    the same dir leaves the file alone. We test this by calling phase_run_inference
    directly on a known dir.
    """
    from triplet_rag.experiment.runner import (
        _PathSet,
        phase_chunk,
        phase_compute_metrics,
        phase_embed,
        phase_ingest,
        phase_run_inference,
        phase_build_chunk_only_index,
        phase_build_index,
    )

    cfg = _vanilla_cfg(name="fixture_resume")
    paths = _PathSet.from_cfg(cfg)
    paths.exp_dir.mkdir(parents=True, exist_ok=True)
    (paths.exp_dir / "logs").mkdir(exist_ok=True)

    phase_ingest(cfg, paths, force=False)
    phase_chunk(cfg, paths, force=False)
    phase_embed(cfg, paths, force=False)
    phase_build_chunk_only_index(cfg, paths, force=False)
    phase_build_index(cfg, paths, force=False)
    phase_run_inference(cfg, paths, force=False)
    pred_path = paths.pred_path
    assert pred_path.exists()
    mtime_first = pred_path.stat().st_mtime

    # Second run on the SAME paths object: should skip
    phase_run_inference(cfg, paths, force=False)
    assert pred_path.stat().st_mtime == mtime_first


def test_force_rebuilds(stub_llm_chat, stub_embedder, _isolated_storage):
    cfg = _vanilla_cfg(name="fixture_force")
    exp_dir1 = run_experiment(cfg)

    # New experiment_id (timestamp differs), so re-run with force creates new dir
    import time

    time.sleep(1.1)
    exp_dir2 = run_experiment(cfg, force=True)
    assert exp_dir1 != exp_dir2  # different timestamps in id
