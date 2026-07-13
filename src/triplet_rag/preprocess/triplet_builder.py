"""Build (question, contexts, answer) triplets.

For each generated question we:
  1. retrieve top-K chunks from a chunks-only index;
  2. reuse the seed answer produced during question generation;
  3. (optional) score faithfulness via a judge and filter.

If a legacy questions table lacks seed answers, we can fall back to a teacher
LLM answer call using the retrieved contexts.
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


async def build_triplets(
    questions: pd.DataFrame,
    chunks: pd.DataFrame,
    chunk_index: IndexBundle,
    question_embeddings: np.ndarray,
    teacher: LLMClient | None,
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

    # 2. Prefer the answer already generated with each synthetic question.
    # This keeps triplet construction retrieval-only in the normal path.
    if "seed_answer" in questions.columns and questions["seed_answer"].fillna("").astype(str).any():
        answers = questions["seed_answer"].fillna("").astype(str).tolist()
        missing = [idx for idx, answer in enumerate(answers) if not answer.strip()]
        if missing:
            if teacher is None:
                logger.warning(
                    f"{len(missing)} questions have empty seed_answer; leaving answers empty"
                )
            else:
                logger.info(
                    f"{len(missing)} questions have empty seed_answer; falling back to teacher"
                )
                fallback = await _answer_with_teacher(
                    questions.iloc[missing],
                    retrieved_chunk_texts=[retrieved_chunk_texts[idx] for idx in missing],
                    teacher=teacher,
                    answer_prompt=answer_prompt,
                    concurrency=concurrency,
                )
                for idx, answer in zip(missing, fallback, strict=True):
                    answers[idx] = answer
    else:
        if teacher is None:
            raise ValueError(
                "questions table has no usable seed_answer column and no teacher was provided"
            )
        logger.info("No seed_answer column found; falling back to teacher answer generation")
        answers = await _answer_with_teacher(
            questions,
            retrieved_chunk_texts=retrieved_chunk_texts,
            teacher=teacher,
            answer_prompt=answer_prompt,
            concurrency=concurrency,
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


async def _answer_with_teacher(
    questions: pd.DataFrame,
    *,
    retrieved_chunk_texts: list[list[str]],
    teacher: LLMClient,
    answer_prompt: str,
    concurrency: int | None,
) -> list[str]:
    prompt_key = answer_gen_key(answer_prompt)
    prompts: list[list[dict[str, str]]] = []
    for q_text, ctxs in zip(questions["question"].values, retrieved_chunk_texts, strict=True):
        msg = render(prompt_key, question=q_text, contexts=ctxs)
        prompts.append([{"role": "user", "content": msg}])
    return await teacher.chat_many_async(
        prompts,
        concurrency=concurrency,
        progress="build_triplets",
    )
