"""Embedder clients: sentence-transformers (local) and OpenAI embeddings (remote).

Both produce normalized float32 numpy arrays of shape (n, dim).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import EmbedderConfig
from ..settings import get_settings


class EmbedderClient:
    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        self.settings = get_settings()
        self._model: Any = None
        self._dim: int | None = cfg.dim

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self.cfg.kind == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading SBERT model: {self.cfg.model_name}")
            self._model = SentenceTransformer(self.cfg.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        elif self.cfg.kind == "openai_embed":
            self._model = "remote"
            if self._dim is None:
                # default dims for common openai embed models
                defaults = {
                    "text-embedding-3-small": 1536,
                    "text-embedding-3-large": 3072,
                    "text-embedding-ada-002": 1536,
                }
                self._dim = defaults.get(self.cfg.model_name, 1536)
        else:
            raise ValueError(f"Unknown embedder kind: {self.cfg.kind}")

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        t0 = time.time()
        if self.cfg.kind == "sentence_transformers":
            arr = self._model.encode(
                texts,
                batch_size=self.cfg.batch_size,
                normalize_embeddings=self.cfg.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
        else:
            arr = self._embed_openai(texts)

        logger.debug(f"Embedded {len(texts)} texts in {time.time()-t0:.1f}s, dim={arr.shape[1]}")
        return arr

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=2, max=60))
    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        from litellm import embedding

        s = self.settings
        outs: list[list[float]] = []
        bs = self.cfg.batch_size
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            resp = embedding(
                model=self.cfg.model_name,
                input=chunk,
                api_key=s.openai_api_key,
                timeout=s.llm_request_timeout,
            )
            for d in resp["data"]:
                outs.append(d["embedding"])
        arr = np.array(outs, dtype=np.float32)
        if self.cfg.normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr = arr / norms
        return arr

    def close(self) -> None:
        # Free GPU memory if held
        if self.cfg.kind == "sentence_transformers" and self._model is not None:
            try:
                import torch

                del self._model
                self._model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
