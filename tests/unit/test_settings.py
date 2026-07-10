from __future__ import annotations

from triplet_rag.settings import Settings


def test_question_gen_concurrency_empty_env_is_none(monkeypatch):
    monkeypatch.setenv("TRIPLET_RAG_QUESTION_GEN_CONCURRENCY", "")

    settings = Settings(_env_file=None)

    assert settings.question_gen_concurrency is None


def test_question_gen_concurrency_env_value(monkeypatch):
    monkeypatch.setenv("TRIPLET_RAG_QUESTION_GEN_CONCURRENCY", "12")

    settings = Settings(_env_file=None)

    assert settings.question_gen_concurrency == 12


def test_embed_batch_size_empty_env_is_none(monkeypatch):
    monkeypatch.setenv("TRIPLET_RAG_EMBED_BATCH_SIZE", "")

    settings = Settings(_env_file=None)

    assert settings.embed_batch_size is None


def test_embed_batch_size_env_value(monkeypatch):
    monkeypatch.setenv("TRIPLET_RAG_EMBED_BATCH_SIZE", "4")

    settings = Settings(_env_file=None)

    assert settings.embed_batch_size == 4
