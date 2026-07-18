"""vLLM subprocess wrapper for serving local HF models with an OpenAI-compatible API.

The subprocess is spawned by `start()`, killed by `stop()`, and a health check
polls the /health endpoint until ready. Errors are loud — we want to fail fast
when GPU memory is exhausted or a model fails to load.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time

import httpx
from loguru import logger

from ..config import LLMConfig
from ..settings import get_settings


class VLLMServer:
    def __init__(self, cfg: LLMConfig, port: int | None = None):
        if cfg.kind != "local_hf":
            raise ValueError("VLLMServer is only for kind='local_hf'")
        self.cfg = cfg
        self.settings = get_settings()
        self.port = port or cfg.vllm_port
        self.proc: subprocess.Popen | None = None
        self.base_url = f"http://localhost:{self.port}/v1"

    def _command(self) -> list[str]:
        gpu_util = (
            self.cfg.vllm_gpu_memory_utilization
            if self.cfg.vllm_gpu_memory_utilization is not None
            else self.settings.vllm_gpu_memory_utilization
        )
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.cfg.model_name,
            "--port",
            str(self.port),
            "--gpu-memory-utilization",
            str(gpu_util),
            "--dtype",
            self.cfg.vllm_dtype,
        ]
        if self.cfg.vllm_max_model_len:
            cmd += ["--max-model-len", str(self.cfg.vllm_max_model_len)]
        cmd += list(self.cfg.vllm_extra_args)
        return cmd

    def start(self) -> str:
        if self.proc is not None:
            raise RuntimeError("Server already started")
        cmd = self._command()
        logger.info(f"Starting vLLM: {shlex.join(cmd)}")
        env = os.environ.copy()
        env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            preexec_fn=os.setsid if os.name == "posix" else None,
        )
        self._wait_for_health()
        return self.base_url

    def _wait_for_health(self) -> None:
        deadline = time.time() + self.settings.vllm_health_timeout_sec
        url = f"http://localhost:{self.port}/health"
        last_err: Exception | None = None
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                # Process died; dump output
                out = self.proc.stdout.read() if self.proc.stdout else ""
                raise RuntimeError(
                    f"vLLM server exited early with code {self.proc.returncode}.\n"
                    f"Output:\n{out[-4000:]}"
                )
            try:
                r = httpx.get(url, timeout=5.0)
                if r.status_code == 200:
                    logger.info(f"vLLM ready at {self.base_url}")
                    return
            except Exception as e:
                last_err = e
            time.sleep(2.0)
        raise TimeoutError(
            f"vLLM did not become healthy within {self.settings.vllm_health_timeout_sec}s. "
            f"Last error: {last_err}"
        )

    def stop(self) -> None:
        if self.proc is None:
            return
        logger.info(f"Stopping vLLM (pid={self.proc.pid})")
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            else:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("vLLM did not stop on SIGTERM; sending SIGKILL")
                if os.name == "posix":
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                else:
                    self.proc.kill()
                self.proc.wait(timeout=10)
        except Exception as e:
            logger.error(f"Error stopping vLLM: {e}")
        finally:
            self.proc = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
