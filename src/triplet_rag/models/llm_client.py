"""Unified LLM client.

We standardize on the OpenAI chat-completions API surface and use litellm under
the hood, which speaks OpenAI / Anthropic / vLLM / many others.

All calls go through a tenacity-backed retry decorator.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from ..config import LLMConfig
from ..settings import get_settings


class LLMError(Exception):
    pass


# Errors we retry on. Network / rate-limit / transient server errors.
class _Retriable(Exception):
    pass


def _retry_predicate(exc: BaseException) -> bool:
    """Return True if this exception should be retried."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    transient_markers = [
        "rate limit",
        "ratelimit",
        "timeout",
        "timed out",
        "connection",
        "service unavailable",
        "server error",
        "502",
        "503",
        "504",
        "529",
    ]
    return (
        "rate" in name
        or "timeout" in name
        or "connection" in name
        or any(m in msg for m in transient_markers)
    )


class LLMClient:
    """Wraps litellm.completion with retries and a unified interface."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        s = get_settings()
        self.settings = s

        # Resolve litellm-style model string + extra kwargs
        if cfg.kind == "openai":
            self._model_str = cfg.model_name  # e.g. "gpt-4o-mini"
            self._extra: dict[str, Any] = {"api_key": s.openai_api_key}
        elif cfg.kind == "anthropic":
            # litellm prefers "anthropic/claude-..."
            mn = cfg.model_name
            if not mn.startswith("anthropic/"):
                mn = f"anthropic/{mn}"
            self._model_str = mn
            self._extra = {"api_key": s.anthropic_api_key}
        elif cfg.kind in ("vllm", "local_hf"):
            # OpenAI-compatible local server
            mn = cfg.model_name
            if not mn.startswith("openai/"):
                mn = f"openai/{mn}"
            self._model_str = mn
            self._extra = {
                "api_base": s.vllm_base_url,
                "api_key": s.vllm_api_key,
            }
        else:
            raise ValueError(f"Unknown LLM kind: {cfg.kind}")

    # ----- Sync core call with tenacity retries -----

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Single completion. Returns the assistant message content."""
        from litellm import completion

        kwargs: dict[str, Any] = {
            "model": self._model_str,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
            "top_p": self.cfg.top_p,
            "timeout": self.settings.llm_request_timeout,
            **self._extra,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        # Tenacity retry, custom predicate
        @retry(
            stop=stop_after_attempt(self.settings.llm_max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self.settings.llm_retry_initial_wait,
                max=self.settings.llm_retry_max_wait,
            ),
            retry=lambda retry_state: (
                retry_state.outcome is not None
                and retry_state.outcome.failed
                and _retry_predicate(retry_state.outcome.exception())
            ),
            before_sleep=before_sleep_log(logger, "WARNING"),
            reraise=True,
        )
        def _call() -> str:
            t0 = time.time()
            resp = completion(**kwargs)
            content = resp["choices"][0]["message"]["content"] or ""
            elapsed = time.time() - t0
            logger.debug(f"chat[{self._model_str}] {elapsed*1000:.0f}ms")
            return content

        try:
            return _call()
        except Exception as e:
            raise LLMError(f"LLM call failed for {self._model_str}: {e}") from e

    # ----- Concurrent batch -----

    def chat_many(
        self,
        prompts: list[list[dict[str, str]]],
        *,
        concurrency: int | None = None,
        progress: str = "",
        **kwargs: Any,
    ) -> list[str]:
        """Run many chat calls concurrently using a thread pool."""
        from concurrent.futures import as_completed

        n = len(prompts)
        c = concurrency or self.settings.llm_concurrency
        if n == 0:
            return []
        results: list[str | None] = [None] * n
        with ThreadPoolExecutor(max_workers=c) as ex:
            futures = {ex.submit(self.chat, p, **kwargs): i for i, p in enumerate(prompts)}
            done = 0
            log_every = max(1, n // 20)
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    logger.error(f"chat_many[{i}] failed: {e}")
                    results[i] = ""
                done += 1
                if progress and done % log_every == 0:
                    logger.info(f"{progress}: {done}/{n}")
        return [r or "" for r in results]
