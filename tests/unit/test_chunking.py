"""Chunking should respect size, overlap, and minimum size."""

import pandas as pd

from triplet_rag.config import ChunkingConfig
from triplet_rag.data.chunking import chunk_corpus


def _corpus_one_doc(text: str) -> pd.DataFrame:
    return pd.DataFrame([{"doc_id": "d1", "title": "t", "text": text, "metadata": {}}])


def test_sliding_window_basic():
    text = "a" * 1000
    cfg = ChunkingConfig(strategy="sliding_window", chunk_size=100, chunk_overlap=20, min_chunk_chars=10)
    chunks = chunk_corpus(_corpus_one_doc(text), cfg)
    assert len(chunks) > 0
    # All chunks should be at most chunk_size
    assert (chunks["text"].str.len() <= 100).all()
    # Overlap: consecutive starts should differ by step = size - overlap = 80
    starts = sorted(chunks["span_start"].tolist())
    if len(starts) >= 2:
        assert starts[1] - starts[0] == 80


def test_sliding_window_min_chars():
    text = "a" * 50
    cfg = ChunkingConfig(strategy="sliding_window", chunk_size=100, chunk_overlap=10, min_chunk_chars=200)
    chunks = chunk_corpus(_corpus_one_doc(text), cfg)
    # Below min_chunk_chars threshold => no chunks
    assert len(chunks) == 0


def test_sentence_chunks():
    text = "First sentence. Second sentence. Third sentence."
    cfg = ChunkingConfig(strategy="sentence", chunk_size=30, chunk_overlap=0, min_chunk_chars=5)
    chunks = chunk_corpus(_corpus_one_doc(text), cfg)
    assert len(chunks) >= 1


def test_chunk_ids_are_unique():
    text = "abcdefg " * 200
    cfg = ChunkingConfig(strategy="sliding_window", chunk_size=100, chunk_overlap=20)
    chunks = chunk_corpus(_corpus_one_doc(text), cfg)
    assert chunks["chunk_id"].is_unique
