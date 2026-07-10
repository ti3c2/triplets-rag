"""ExperimentConfig hash layering: preprocessing/index/experiment hashes
should change appropriately when their respective inputs change.
"""

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
)


def _base_cfg(**overrides) -> ExperimentConfig:
    base = dict(
        experiment_name="test",
        dataset=DatasetConfig(name="fixture"),
        chunking=ChunkingConfig(),
        generator=LLMConfig(kind="openai", model_name="gpt-4o-mini"),
        embedder=EmbedderConfig(kind="sentence_transformers", model_name="bge"),
        indexer=IndexerConfig(indexing_strategy=IndexingStrategy.CHUNKS_ONLY),
        retriever=RetrieverConfig(),
        student=LLMConfig(kind="openai", model_name="gpt-4o-mini"),
        inference=InferenceConfig(strategy=InferenceStrategy.VANILLA_RAG),
        budget=BudgetConfig(),
        preprocessing=PreprocessingConfig(),
        filtering=FilteringConfig(),
        metrics=MetricsConfig(),
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def test_changing_student_does_not_change_preprocessing_hash():
    """Key invariant: artifacts must be shareable across students."""
    a = _base_cfg(student=LLMConfig(kind="openai", model_name="gpt-4o-mini"))
    b = _base_cfg(student=LLMConfig(kind="openai", model_name="gpt-4o"))
    assert a.preprocessing_hash == b.preprocessing_hash
    # But experiment hashes must differ
    assert a.experiment_hash != b.experiment_hash


def test_changing_student_base_url_does_not_change_experiment_hash():
    """The same model served at a different endpoint is the same experiment."""
    a = _base_cfg(
        student=LLMConfig(
            kind="vllm",
            model_name="Qwen/Qwen2.5-7B-Instruct",
            base_url="http://localhost:7113/v1",
        )
    )
    b = _base_cfg(
        student=LLMConfig(
            kind="vllm",
            model_name="Qwen/Qwen2.5-7B-Instruct",
            base_url="http://other-host:8000/v1",
        )
    )
    assert a.preprocessing_hash == b.preprocessing_hash
    assert a.index_hash == b.index_hash
    assert a.experiment_hash == b.experiment_hash


def test_changing_chunking_changes_preprocessing_hash():
    a = _base_cfg(chunking=ChunkingConfig(chunk_size=512))
    b = _base_cfg(chunking=ChunkingConfig(chunk_size=256))
    assert a.preprocessing_hash != b.preprocessing_hash


def test_changing_generator_changes_preprocessing_hash():
    """Different teacher models => different artifacts (different generated questions)."""
    a = _base_cfg(generator=LLMConfig(kind="openai", model_name="gpt-4o-mini"))
    b = _base_cfg(generator=LLMConfig(kind="openai", model_name="gpt-4o"))
    assert a.preprocessing_hash != b.preprocessing_hash


def test_changing_generator_base_url_does_not_change_preprocessing_hash():
    """Endpoint routing is runtime wiring, not generated-content identity."""
    a = _base_cfg(
        generator=LLMConfig(
            kind="vllm",
            model_name="Qwen/Qwen2.5-32B-Instruct",
            base_url="http://localhost:7112/v1",
        )
    )
    b = _base_cfg(
        generator=LLMConfig(
            kind="vllm",
            model_name="Qwen/Qwen2.5-32B-Instruct",
            base_url="http://other-host:8000/v1",
        )
    )
    assert a.preprocessing_hash == b.preprocessing_hash


def test_changing_embedder_base_url_does_not_change_preprocessing_hash():
    """The same embedding model served elsewhere should reuse artifacts."""
    a = _base_cfg(
        embedder=EmbedderConfig(
            kind="openai_compatible",
            model_name="jinaai/jina-embeddings-v3",
            base_url="http://localhost:3300/v1",
            dim=1024,
        )
    )
    b = _base_cfg(
        embedder=EmbedderConfig(
            kind="openai_compatible",
            model_name="jinaai/jina-embeddings-v3",
            base_url="http://other-host:3300/v1",
            dim=1024,
        )
    )
    assert a.preprocessing_hash == b.preprocessing_hash


def test_changing_embedder_model_changes_preprocessing_hash():
    a = _base_cfg(embedder=EmbedderConfig(kind="openai_embed", model_name="model-a", dim=1536))
    b = _base_cfg(embedder=EmbedderConfig(kind="openai_embed", model_name="model-b", dim=1536))
    assert a.preprocessing_hash != b.preprocessing_hash


def test_index_hash_includes_indexer_kind():
    a = _base_cfg(indexer=IndexerConfig(indexing_strategy=IndexingStrategy.CHUNKS_ONLY))
    b = _base_cfg(indexer=IndexerConfig(indexing_strategy=IndexingStrategy.TRIPLETS))
    assert a.index_hash != b.index_hash


def test_retriever_change_affects_experiment_only():
    a = _base_cfg(retriever=RetrieverConfig(top_k=5))
    b = _base_cfg(retriever=RetrieverConfig(top_k=10))
    assert a.preprocessing_hash == b.preprocessing_hash
    assert a.index_hash == b.index_hash
    assert a.experiment_hash != b.experiment_hash


def test_experiment_id_format():
    cfg = _base_cfg()
    eid = cfg.experiment_id
    parts = eid.split("_")
    # YYYYMMDD_HHMMSS_<hash>_<slug>
    assert len(parts) >= 4
