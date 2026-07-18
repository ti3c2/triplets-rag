"""Indexing strategies.

Each strategy produces an IndexBundle from precomputed embeddings and the
artifact dataframes (chunks, questions, triplets).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from ..config import IndexerConfig, IndexingStrategy
from .store import IndexBundle, build_faiss


def _id_map_for_chunks(chunks: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rowid": np.arange(len(chunks)),
            "item_type": "chunk",
            "item_id": chunks["chunk_id"].values,
            "doc_id": chunks["doc_id"].values,
            "text": chunks["text"].values,
        }
    )


def _id_map_for_questions(questions: pd.DataFrame, chunks: pd.DataFrame) -> pd.DataFrame:
    chunk_lookup = chunks.set_index("chunk_id")[["doc_id", "text"]]
    items = questions.join(chunk_lookup, on="chunk_id", rsuffix="_chunk")
    return pd.DataFrame(
        {
            "rowid": np.arange(len(items)),
            "item_type": "question",
            "item_id": items["question_id"].values,
            "chunk_id": items["chunk_id"].values,
            "doc_id": items["doc_id"].values,
            "text": items["question"].values,
            "chunk_text": items["text"].values,
        }
    )


def build_index_for_strategy(
    strategy: IndexingStrategy,
    cfg: IndexerConfig,
    *,
    chunks: pd.DataFrame,
    questions: pd.DataFrame | None,
    triplets: pd.DataFrame | None,
    chunk_emb: np.ndarray | None,
    question_emb: np.ndarray | None,
) -> IndexBundle:
    logger.info(f"Building index for strategy={strategy.value}")

    if strategy == IndexingStrategy.CHUNKS_ONLY:
        assert chunk_emb is not None
        index = build_faiss(chunk_emb, cfg)
        return IndexBundle(index=index, id_map=_id_map_for_chunks(chunks))

    if strategy == IndexingStrategy.QUESTIONS_ONLY:
        assert questions is not None and question_emb is not None
        index = build_faiss(question_emb, cfg)
        return IndexBundle(index=index, id_map=_id_map_for_questions(questions, chunks))

    if strategy == IndexingStrategy.CHUNKS_AND_QUESTIONS:
        assert chunk_emb is not None and question_emb is not None and questions is not None
        merged = np.vstack([chunk_emb, question_emb]).astype(np.float32)
        index = build_faiss(merged, cfg)
        chunk_map = _id_map_for_chunks(chunks)
        q_map = _id_map_for_questions(questions, chunks)
        q_map = q_map.copy()
        q_map["rowid"] = q_map["rowid"] + len(chunk_map)
        id_map = pd.concat([chunk_map, q_map], ignore_index=True)
        return IndexBundle(index=index, id_map=id_map)

    if strategy == IndexingStrategy.TRIPLETS:
        # We index the *question* embeddings, but each row points to a triplet
        assert triplets is not None and questions is not None and question_emb is not None
        # Triplets are keyed by question_id
        q_to_idx = {qid: i for i, qid in enumerate(questions["question_id"].values)}
        rows = []
        emb_rows = []
        for _, t in triplets.iterrows():
            qid = t["question_id"]
            if qid not in q_to_idx:
                continue
            emb_rows.append(question_emb[q_to_idx[qid]])
            rows.append(
                {
                    "rowid": len(rows),
                    "item_type": "triplet",
                    "item_id": t["triplet_id"],
                    "question_id": qid,
                    "text": t["question"],
                    "chunk_ids": list(t["retrieved_chunk_ids"]),
                    "teacher_answer": t["teacher_answer"],
                }
            )
        if not emb_rows:
            raise ValueError("No triplets with embeddings; check generation step.")
        emb = np.vstack(emb_rows).astype(np.float32)
        index = build_faiss(emb, cfg)
        return IndexBundle(index=index, id_map=pd.DataFrame(rows))

    if strategy == IndexingStrategy.QA_PAIRS:
        # Index the (question + answer) text using question embedding as proxy.
        # For purer ablation, the embedder should be re-run on Q||A; we accept
        # using question embedding here as a documented simplification.
        assert triplets is not None and questions is not None and question_emb is not None
        q_to_idx = {qid: i for i, qid in enumerate(questions["question_id"].values)}
        rows = []
        emb_rows = []
        for _, t in triplets.iterrows():
            qid = t["question_id"]
            if qid not in q_to_idx:
                continue
            emb_rows.append(question_emb[q_to_idx[qid]])
            rows.append(
                {
                    "rowid": len(rows),
                    "item_type": "qa_pair",
                    "item_id": t["triplet_id"],
                    "question_id": qid,
                    "text": f"Q: {t['question']} A: {t['teacher_answer']}",
                    "teacher_answer": t["teacher_answer"],
                }
            )
        emb = np.vstack(emb_rows).astype(np.float32)
        index = build_faiss(emb, cfg)
        return IndexBundle(index=index, id_map=pd.DataFrame(rows))

    raise ValueError(f"Unknown strategy: {strategy}")
