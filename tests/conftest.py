"""Pytest config + stubbed LLM/embedder for offline integration testing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _isolated_storage(monkeypatch, tmp_path):
    """Each test gets its own storage dir so tests don't pollute each other."""
    monkeypatch.setenv("TRIPLET_RAG_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "stub-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stub-key")
    # Reset settings cache
    from triplet_rag.settings import get_settings

    get_settings.cache_clear()
    yield tmp_path


@pytest.fixture
def stub_llm_chat(monkeypatch):
    """Patch litellm.completion so no real API calls are made.

    The stub returns deterministic content based on the prompt:
    - For question_gen prompts (containing "generate ... question-answer pairs"): emits
      two "Q? A" lines.
    - For answer_gen prompts (containing "Use ONLY the information"): emits a stub
      answer derived from the first context line.
    - For triplet_rag/qa_demo_rag inference prompts: emits a stub student answer.
    - For judge prompts (containing "JSON:"): emits {"faithfulness": 0.9, ...}.
    """

    def _fake_completion(**kwargs: Any) -> dict[str, Any]:
        msgs = kwargs.get("messages", [])
        prompt = msgs[-1]["content"] if msgs else ""

        if "JSON:" in prompt and "faithfulness" in prompt:
            content = '{"faithfulness": 0.9, "rationale": "stub"}'
        elif "generate" in prompt and "question-answer" in prompt:
            # Two QA pairs
            content = (
                "What is mentioned in the text? Some content is mentioned\n"
                "Who or what is described? An entity is described"
            )
        elif "Use ONLY the information" in prompt or "Answer ONLY from the contexts" in prompt:
            # Use a fragment of the prompt as the answer
            if "ATP" in prompt:
                content = "ATP"
            elif "photosynthesis" in prompt or "sunlight" in prompt.lower():
                content = "sunlight, water and carbon dioxide"
            elif "Eiffel" in prompt:
                content = "Gustave Eiffel"
            else:
                content = "I don't know."
        else:
            content = "stub response"

        return {
            "choices": [{"message": {"content": content}}],
            "model": kwargs.get("model"),
        }

    async def _fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        return _fake_completion(**kwargs)

    monkeypatch.setattr("litellm.completion", _fake_completion)
    monkeypatch.setattr("litellm.acompletion", _fake_acompletion)
    yield _fake_completion


@pytest.fixture
def stub_embedder(monkeypatch):
    """Patch SentenceTransformer to a tiny deterministic hash-based embedder.

    Avoids downloading models during tests.
    """
    import hashlib

    dim = 32

    class _StubModel:
        def __init__(self, name):
            self.name = name

        def get_sentence_embedding_dimension(self) -> int:
            return dim

        def encode(
            self,
            texts,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ):
            outs = []
            for t in texts:
                # Hash-based deterministic vector; same text -> same vector
                h = hashlib.sha256(t.encode("utf-8")).digest()
                vec = np.frombuffer(h * 8, dtype=np.uint8)[:dim].astype(np.float32)
                vec = vec - 128.0
                if normalize_embeddings:
                    norm = np.linalg.norm(vec) + 1e-12
                    vec = vec / norm
                outs.append(vec)
            return np.vstack(outs)

    monkeypatch.setattr(
        "triplet_rag.models.embedder_client.EmbedderClient._ensure_loaded",
        lambda self: _stub_ensure_loaded(self, _StubModel),
    )
    yield


def _stub_ensure_loaded(self, model_cls):
    if self._model is not None:
        return
    self._model = model_cls(self.cfg.model_name)
    self._dim = self._model.get_sentence_embedding_dimension()
