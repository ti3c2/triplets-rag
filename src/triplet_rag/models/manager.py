"""Model lifecycle manager.

Use as a context manager. Cleans up subprocess servers and frees GPU memory on
exit. The orchestrator wraps each phase that needs a model in one of these.

For remote APIs (openai, anthropic) start/stop are no-ops; for local_hf, a
vLLM subprocess is spawned.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from ..config import EmbedderConfig, LLMConfig
from .embedder_client import EmbedderClient
from .llm_client import LLMClient
from .vllm_server import VLLMServer


class _LifecycleLogger:
    """Append-only JSONL log of model start/stop events."""

    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, payload: dict[str, Any]) -> None:
        rec = {"ts": datetime.utcnow().isoformat() + "Z", "event": event, **payload}
        logger.info(f"[lifecycle] {event} {payload}")
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(rec) + "\n")


class ManagedLLM:
    """Context manager that yields an LLMClient, starting/stopping a vLLM
    subprocess if needed.
    """

    def __init__(self, cfg: LLMConfig, lifecycle_log: Path | None = None):
        self.cfg = cfg
        self._server: VLLMServer | None = None
        self._client: LLMClient | None = None
        self._log = _LifecycleLogger(lifecycle_log)

    def __enter__(self) -> LLMClient:
        self._log.log("llm_start_begin", {"kind": self.cfg.kind, "model": self.cfg.model_name})
        if self.cfg.kind == "local_hf":
            self._server = VLLMServer(self.cfg)
            self._server.start()
            # Build a synthetic LLMConfig of kind=vllm pointing at this port
            client_cfg = LLMConfig(
                kind="vllm",
                model_name=self.cfg.model_name,
                base_url=self._server.base_url,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                top_p=self.cfg.top_p,
                vllm_port=self.cfg.vllm_port,
            )
            self._client = LLMClient(client_cfg)
        else:
            self._client = LLMClient(self.cfg)
        self._log.log("llm_start_done", {"kind": self.cfg.kind, "model": self.cfg.model_name})
        return self._client

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._server is not None:
            try:
                self._server.stop()
            finally:
                self._server = None
        self._client = None
        self._log.log("llm_stopped", {"kind": self.cfg.kind, "model": self.cfg.model_name})


class ManagedEmbedder:
    """Context manager that yields an EmbedderClient and frees GPU memory on exit."""

    def __init__(self, cfg: EmbedderConfig, lifecycle_log: Path | None = None):
        self.cfg = cfg
        self._client: EmbedderClient | None = None
        self._log = _LifecycleLogger(lifecycle_log)

    def __enter__(self) -> EmbedderClient:
        self._log.log("embedder_start", {"kind": self.cfg.kind, "model": self.cfg.model_name})
        self._client = EmbedderClient(self.cfg)
        # eager load so failures happen here, not later
        _ = self._client.dim
        return self._client

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._log.log("embedder_stopped", {"kind": self.cfg.kind, "model": self.cfg.model_name})
