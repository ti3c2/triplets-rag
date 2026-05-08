"""Environment-driven settings, loaded once at startup.

Use `get_settings()` everywhere — it caches a single instance.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-level configuration, parsed from env or .env."""

    model_config = SettingsConfigDict(
        env_prefix="",  # we set explicit names below for clarity
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Provider credentials ---
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # --- Local OpenAI-compatible server (vLLM, llama.cpp, etc.) ---
    vllm_base_url: str = Field(default="http://localhost:8000/v1", alias="VLLM_BASE_URL")
    vllm_api_key: str = Field(default="EMPTY", alias="VLLM_API_KEY")

    # --- Local OpenAI-compatible embeddings server (TEI, infinity, jina, ...) ---
    embedder_base_url: str | None = Field(default=None, alias="EMBEDDER_BASE_URL")
    embedder_api_key: str = Field(default="EMPTY", alias="EMBEDDER_API_KEY")

    # --- Storage ---
    storage_dir: Path = Field(default=Path("./storage"), alias="TRIPLET_RAG_STORAGE_DIR")

    # --- Logging ---
    log_level: str = Field(default="INFO", alias="TRIPLET_RAG_LOG_LEVEL")

    # --- Retry behavior ---
    llm_max_retries: int = Field(default=6, alias="TRIPLET_RAG_LLM_MAX_RETRIES")
    llm_retry_initial_wait: float = Field(default=2.0, alias="TRIPLET_RAG_LLM_RETRY_INITIAL_WAIT")
    llm_retry_max_wait: float = Field(default=60.0, alias="TRIPLET_RAG_LLM_RETRY_MAX_WAIT")
    llm_request_timeout: float = Field(default=120.0, alias="TRIPLET_RAG_LLM_REQUEST_TIMEOUT")

    # --- Concurrency ---
    llm_concurrency: int = Field(default=8, alias="TRIPLET_RAG_LLM_CONCURRENCY")
    embed_batch_size: int = Field(default=64, alias="TRIPLET_RAG_EMBED_BATCH_SIZE")

    # --- vLLM lifecycle ---
    vllm_health_timeout_sec: int = Field(default=300, alias="TRIPLET_RAG_VLLM_HEALTH_TIMEOUT_SEC")
    vllm_gpu_memory_utilization: float = Field(
        default=0.9, alias="TRIPLET_RAG_VLLM_GPU_MEMORY_UTILIZATION"
    )

    # --- Convenience computed paths ---
    @property
    def raw_dir(self) -> Path:
        return self.storage_dir / "raw"

    @property
    def artifacts_dir(self) -> Path:
        return self.storage_dir / "artifacts"

    @property
    def indices_dir(self) -> Path:
        return self.storage_dir / "indices"

    @property
    def experiments_dir(self) -> Path:
        return self.storage_dir / "experiments"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.artifacts_dir, self.indices_dir, self.experiments_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
