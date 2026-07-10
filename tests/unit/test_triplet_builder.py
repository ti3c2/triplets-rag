from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from triplet_rag.config import IndexerConfig, LLMConfig
from triplet_rag.index.store import IndexBundle, build_faiss
from triplet_rag.preprocess.triplet_builder import build_triplets


class _ExplodingTeacher:
    def chat_many(self, *args, **kwargs):
        raise AssertionError("teacher should not be called when seed_answer is present")


def _chunk_bundle(chunks: pd.DataFrame, embeddings: np.ndarray) -> IndexBundle:
    index = build_faiss(embeddings, IndexerConfig())
    id_map = pd.DataFrame(
        {
            "rowid": list(range(len(chunks))),
            "item_type": ["chunk"] * len(chunks),
            "item_id": chunks["chunk_id"].tolist(),
        }
    )
    return IndexBundle(index=index, id_map=id_map)


def test_build_triplets_uses_seed_answer_without_teacher_call():
    chunks = pd.DataFrame(
        [
            {"chunk_id": "c1", "text": "Notre Dame has a copper statue of Christ."},
            {"chunk_id": "c2", "text": "The Grotto is a Marian place of prayer."},
        ]
    )
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    questions = pd.DataFrame(
        [
            {
                "question_id": "q1",
                "chunk_id": "c1",
                "question": "What is in front of the Main Building?",
                "seed_answer": "a copper statue of Christ",
            }
        ]
    )
    question_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

    triplets = build_triplets(
        questions,
        chunks,
        _chunk_bundle(chunks, embeddings),
        question_embeddings,
        teacher=_ExplodingTeacher(),
        teacher_cfg=LLMConfig(kind="openai", model_name="teacher"),
        contexts_per_question=1,
    )

    assert len(triplets) == 1
    assert triplets.loc[0, "teacher_answer"] == "a copper statue of Christ"
    assert triplets.loc[0, "retrieved_chunk_ids"] == ["c1"]


def test_build_triplets_requires_teacher_without_seed_answer():
    chunks = pd.DataFrame([{"chunk_id": "c1", "text": "Context"}])
    embeddings = np.array([[1.0]], dtype=np.float32)
    questions = pd.DataFrame([{"question_id": "q1", "chunk_id": "c1", "question": "Question?"}])

    with pytest.raises(ValueError, match="seed_answer"):
        build_triplets(
            questions,
            chunks,
            _chunk_bundle(chunks, embeddings),
            np.array([[1.0]], dtype=np.float32),
            teacher=None,
            teacher_cfg=LLMConfig(kind="openai", model_name="teacher"),
            contexts_per_question=1,
        )
