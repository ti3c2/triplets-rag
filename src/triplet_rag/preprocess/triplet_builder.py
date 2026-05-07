"""Build (question, contexts, answer) triplets.

For each generated question we:
  1. retrieve top-K chunks from a chunks-only index;
  2. ask the teacher LLM for an answer using those chunks;
  3. (optional) score faithfulness via a judge and filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from ..config import LLMConfig
from ..index.store import IndexBundle, search
from ..models import LLMClient
from ..prompts import answer_gen_key, render
from ..utils.hashing import stable_hash_str


def build_triplets(
    questions: pd.DataFrame,
    chunks: pd.DataFrame,
    chunk_index: IndexBundle,
    question_embeddings: np.ndarray,
    teacher: LLMClient,
    teacher_cfg: LLMConfig,
    *,
    contexts_per_question: int = 5,
    answer_prompt: str = "rag_default",
    concurrency: int | None = None,
) -> pd.DataFrame:
    """Returns triplets dataframe: triplet_id, question_id, question,
    retrieved_chunk_ids, retrieved_chunk_texts, teacher_answer, teacher_model.
    """
    n = len(questions)
    if n == 0:
        return pd.DataFrame()
    logger.info(f"Building triplets for {n} questions, top-{contexts_per_question} contexts each")

    # 1. Retrieve contexts via the chunk index
    distances, neighbor_ids = search(chunk_index, question_embeddings, contexts_per_question)
    chunk_lookup = chunks.set_index("chunk_id")["text"].to_dict()
    chunk_id_by_row = chunk_index.id_map.set_index("rowid")["item_id"].to_dict()

    retrieved_chunk_ids: list[list[str]] = []
    retrieved_chunk_texts: list[list[str]] = []
    for row in neighbor_ids:
        ids = []
        texts = []
        for r in row:
            r_int = int(r)
            if r_int < 0:
                continue
            cid = chunk_id_by_row.get(r_int)
            if cid is None:
                continue
            ids.append(cid)
            texts.append(chunk_lookup.get(cid, ""))
        retrieved_chunk_ids.append(ids)
        retrieved_chunk_texts.append(texts)

    # 2. Build prompts and call teacher
    prompt_key = answer_gen_key(answer_prompt)
    prompts: list[list[dict[str, str]]] = []
    for q_text, ctxs in zip(questions["question"].values, retrieved_chunk_texts, strict=True):
        msg = render(prompt_key, question=q_text, contexts=ctxs)
        prompts.append([{"role": "user", "content": msg}])

    answers = teacher.chat_many(
        prompts,
        concurrency=concurrency,
        progress="build_triplets",
    )

    rows = []
    for q_row, cids, ctxs, ans in zip(
        questions.itertuples(index=False),
        retrieved_chunk_ids,
        retrieved_chunk_texts,
        answers,
        strict=True,
    ):
        triplet_id = f"T-{stable_hash_str(q_row.question_id + ans, 12)}"
        rows.append(
            {
                "triplet_id": triplet_id,
                "question_id": q_row.question_id,
                "question": q_row.question,
                "retrieved_chunk_ids": cids,
                "retrieved_chunk_texts": ctxs,
                "teacher_answer": ans.strip(),
                "teacher_model": teacher_cfg.model_name,
                "faithfulness_score": None,
                "kept": True,
                "metadata": {},
            }
        )
    df = pd.DataFrame(rows)
    logger.info(f"Built {len(df)} triplets")
    return df
