"""Chunking strategies for the corpus."""

from __future__ import annotations

import re
from collections.abc import Iterator

import pandas as pd
from loguru import logger

from ..config import ChunkingConfig
from ..utils.hashing import stable_hash_str


def _sliding_chunks(text: str, size: int, overlap: int) -> Iterator[tuple[int, int, str]]:
    """Sliding window over characters. Yields (start, end, text)."""
    if not text:
        return
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")
    step = size - overlap
    n = len(text)
    start = 0
    while start < n:
        end = min(start + size, n)
        yield start, end, text[start:end]
        if end == n:
            break
        start += step


def _sentence_chunks(text: str, max_chars: int) -> Iterator[tuple[int, int, str]]:
    """Sentence-bounded chunks; pack sentences up to max_chars."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    buf = ""
    buf_start = 0
    cur = 0
    for s in sents:
        if not s:
            continue
        if buf and len(buf) + 1 + len(s) > max_chars:
            yield buf_start, buf_start + len(buf), buf
            buf_start = cur
            buf = s
        else:
            if buf:
                buf = f"{buf} {s}"
            else:
                buf_start = cur
                buf = s
        cur += len(s) + 1
    if buf:
        yield buf_start, buf_start + len(buf), buf


def _paragraph_chunks(text: str) -> Iterator[tuple[int, int, str]]:
    """Split on blank lines."""
    cur = 0
    for para in re.split(r"\n\s*\n", text):
        if para.strip():
            yield cur, cur + len(para), para.strip()
        cur += len(para) + 2  # rough


def chunk_corpus(corpus: pd.DataFrame, cfg: ChunkingConfig) -> pd.DataFrame:
    """Return chunks dataframe with columns: chunk_id, doc_id, text, span_start, span_end, metadata."""
    logger.info(f"Chunking {len(corpus)} docs with {cfg.strategy} (size={cfg.chunk_size})")
    rows = []
    for _, doc in corpus.iterrows():
        text = doc["text"]
        doc_id = doc["doc_id"]
        if cfg.strategy == "sliding_window":
            iterator = _sliding_chunks(text, cfg.chunk_size, cfg.chunk_overlap)
        elif cfg.strategy == "sentence":
            iterator = _sentence_chunks(text, cfg.chunk_size)
        elif cfg.strategy == "paragraph":
            iterator = _paragraph_chunks(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {cfg.strategy}")

        for start, end, chunk_text in iterator:
            if len(chunk_text) < cfg.min_chunk_chars:
                continue
            chunk_id = f"{doc_id}::{start}-{end}::{stable_hash_str(chunk_text, 6)}"
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "span_start": int(start),
                    "span_end": int(end),
                    "metadata": {},
                }
            )
    df = pd.DataFrame(rows)
    logger.info(f"Produced {len(df)} chunks")
    return df
