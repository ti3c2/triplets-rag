"""Filtering triplets by teacher-answer faithfulness.

Uses RAGAS faithfulness when enabled, otherwise a lightweight LLM-as-judge
fallback that asks 'is this answer fully supported by these contexts? Score 0-1.'

For pilot speed, callers may pass a sample to the filter and threshold on
the score; below-threshold triplets are marked kept=False but retained in
storage for reproducibility / analysis.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from loguru import logger

from ..config import FilteringConfig, LLMConfig
from ..models import LLMClient


_JUDGE_PROMPT = """You are a strict factuality judge. Given a question, a list of contexts, and an answer, decide whether every claim in the answer is supported by the contexts.

Respond ONLY in JSON of the form:
{"faithfulness": <float in [0,1]>, "rationale": "<one short sentence>"}

A score of 1.0 means every claim is fully supported. 0.0 means the answer contradicts or invents information not in the contexts.

Question: {question}

Contexts:
{contexts}

Answer: {answer}

JSON:"""


def _score_with_judge(
    triplets: pd.DataFrame,
    judge: LLMClient,
    *,
    concurrency: int | None = None,
) -> list[float]:
    prompts: list[list[dict[str, str]]] = []
    for _, t in triplets.iterrows():
        ctxs = "\n\n".join(f"- {c}" for c in t["retrieved_chunk_texts"])
        prompts.append(
            [
                {
                    "role": "user",
                    "content": _JUDGE_PROMPT.format(
                        question=t["question"], contexts=ctxs, answer=t["teacher_answer"]
                    ),
                }
            ]
        )
    raw = judge.chat_many(prompts, concurrency=concurrency, progress="filter_judge")
    scores: list[float] = []
    for r in raw:
        try:
            r2 = r.strip()
            # Strip code fences if present
            if r2.startswith("```"):
                r2 = r2.strip("`")
                if r2.lower().startswith("json"):
                    r2 = r2[4:]
            obj: Any = json.loads(r2)
            score = float(obj.get("faithfulness", 0.0))
            scores.append(max(0.0, min(1.0, score)))
        except Exception as e:
            logger.warning(f"judge parse failed: {e}; raw={r[:100]}")
            scores.append(0.0)
    return scores


def filter_triplets(
    triplets: pd.DataFrame,
    cfg: FilteringConfig,
    judge: LLMClient | None,
    *,
    concurrency: int | None = None,
) -> pd.DataFrame:
    if not cfg.enabled or len(triplets) == 0:
        return triplets

    if judge is None:
        logger.warning("Filtering enabled but no judge provided; skipping")
        return triplets

    logger.info(f"Filtering {len(triplets)} triplets with threshold={cfg.faithfulness_threshold}")
    scores = _score_with_judge(triplets, judge, concurrency=concurrency)
    triplets = triplets.copy()
    triplets["faithfulness_score"] = scores
    triplets["kept"] = [s >= cfg.faithfulness_threshold for s in scores]
    n_kept = int(triplets["kept"].sum())
    logger.info(f"Kept {n_kept}/{len(triplets)} triplets after filtering")
    return triplets
