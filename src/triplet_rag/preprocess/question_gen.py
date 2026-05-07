"""Generate questions per chunk using the teacher LLM."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
from loguru import logger

from ..config import LLMConfig, PreprocessingConfig
from ..models import LLMClient
from ..prompts import question_gen_key, render
from ..utils.hashing import stable_hash_str


def _parse_qa_lines(text: str) -> list[tuple[str, str]]:
    """Parse 'Question? Answer' lines, robust to small format variations."""
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip numbering/bullets
        line = re.sub(r"^[\-\*\d\.\)\s]+", "", line)
        if "?" not in line:
            continue
        idx = line.find("?")
        q = line[: idx + 1].strip()
        a = line[idx + 1 :].strip().lstrip("-:.").strip()
        if not q or not a:
            continue
        pairs.append((q, a))
    return pairs


def generate_questions(
    chunks: pd.DataFrame,
    teacher: LLMClient,
    teacher_cfg: LLMConfig,
    pp: PreprocessingConfig,
    *,
    concurrency: int | None = None,
) -> pd.DataFrame:
    """For each chunk, ask the teacher for N QA pairs. Returns a flat dataframe.

    Columns: question_id, chunk_id, question, seed_answer, generator_model.
    """
    prompt_key = question_gen_key(pp.question_prompt)
    logger.info(
        f"Generating questions: {len(chunks)} chunks, "
        f"target {pp.num_questions_per_chunk} per chunk, prompt={prompt_key}"
    )
    prompts: list[list[dict[str, str]]] = []
    for _, ch in chunks.iterrows():
        text = render(prompt_key, n=pp.num_questions_per_chunk, chunk_text=ch["text"])
        prompts.append([{"role": "user", "content": text}])

    raw = teacher.chat_many(prompts, concurrency=concurrency, progress="generate_questions")

    rows = []
    for ch_row, response in zip(chunks.itertuples(index=False), raw, strict=True):
        chunk_id = ch_row.chunk_id
        for q, a in _parse_qa_lines(response):
            qid = f"{chunk_id}::Q-{stable_hash_str(q, 8)}"
            rows.append(
                {
                    "question_id": qid,
                    "chunk_id": chunk_id,
                    "question": q,
                    "seed_answer": a,
                    "generator_model": teacher_cfg.model_name,
                    "metadata": {},
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No questions parsed; check the model's output format.")
        return df

    # Deduplicate exact-text duplicates within a chunk
    df = df.drop_duplicates(subset=["chunk_id", "question"]).reset_index(drop=True)
    logger.info(f"Generated {len(df)} questions across {df['chunk_id'].nunique()} chunks")

    if pp.max_questions_total and len(df) > pp.max_questions_total:
        df = df.sample(n=pp.max_questions_total, random_state=42).reset_index(drop=True)
        logger.info(f"Subsampled to {len(df)} questions")

    return df
