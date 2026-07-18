#!/usr/bin/env bash
# SQuAD-1k pilot using a Qwen2.5-7B-Instruct vLLM server you started yourself.
#
# Prereqs:
#   1. uv installed (https://docs.astral.sh/uv/)
#   2. A vLLM OpenAI-compatible LLM server running for Qwen/Qwen2.5-7B-Instruct
#      at http://localhost:7113. Example:
#        vllm serve Qwen/Qwen2.5-7B-Instruct \
#          --port 7113 --dtype auto --max-model-len 8192
#   3. An OpenAI-compatible embeddings server running for
#      jinaai/jina-embeddings-v3 at http://localhost:3300.
#   4. .env contains:
#        VLLM_BASE_URL=http://localhost:7113/v1
#        EMBEDDER_BASE_URL=http://localhost:3300/v1
#      (no OPENAI_API_KEY needed — both LLM and embedder use local servers)
#
# What it does:
#   - vanilla RAG baseline
#   - QuOTE-style (chunks+questions index)
#   - triplet RAG (q2q retrieval, the proposed approach)
# Then prints a comparison report.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "ERROR: .env not found. Copy .env.example to .env and set VLLM_BASE_URL=http://localhost:7113/v1"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found. See https://docs.astral.sh/uv/"
    exit 1
fi

# Quick connectivity checks
BASE_URL="${VLLM_BASE_URL:-http://localhost:7113/v1}"
if ! curl -sf "${BASE_URL%/v1}/health" >/dev/null 2>&1 \
   && ! curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
    echo "WARN: cannot reach vLLM (LLM) at ${BASE_URL}. Make sure the server is up."
fi

EMB_URL="${EMBEDDER_BASE_URL:-http://localhost:3300/v1}"
if ! curl -sf "${EMB_URL%/v1}/health" >/dev/null 2>&1 \
   && ! curl -sf "${EMB_URL}/models" >/dev/null 2>&1; then
    echo "WARN: cannot reach embeddings server at ${EMB_URL}. Make sure it's up."
fi

uv sync

# Smoke test on the bundled fixture corpus first (~10s, still hits your vLLM + jina).
uv run triplet-rag run --config experiment/fixture_smoke \
    -o generator.kind=vllm -o generator.model_name=Qwen/Qwen2.5-7B-Instruct \
    -o student.kind=vllm   -o student.model_name=Qwen/Qwen2.5-7B-Instruct \
    -o embedder.kind=openai_compatible \
    -o embedder.model_name=jinaai/jina-embeddings-v3 \
    -o embedder.dim=1024 \
    -o chunking.chunk_size=8192 -o chunking.chunk_overlap=0

# Vanilla baseline
uv run triplet-rag run --config experiment/squad_qwen7b_vanilla

# QuOTE-style
uv run triplet-rag run --config experiment/squad_qwen7b_quote

# Triplet-RAG (the proposed approach)
uv run triplet-rag run --config experiment/squad_qwen7b_triplet

echo
echo "=========================================="
echo " Comparison report (filtered to qwen7b runs)"
echo "=========================================="
uv run triplet-rag report --filter experiment_name=squad_qwen7b_*

echo
echo "Per-experiment artefacts live under: storage/experiments/<experiment_id>/"
echo "  - metrics/aggregate.json    -> headline metrics + bootstrap CIs"
echo "  - metrics/per_query.parquet -> long-format per-query scores"
echo "  - predictions.parquet       -> student answers + retrieved demo ids"
echo "  - config.yaml.json          -> fully-resolved frozen config"
echo "  - logs/run.log, logs/model_lifecycle.log"
echo
echo "List runs:    uv run triplet-rag list --filter dataset=squad"
echo "Inspect one:  uv run triplet-rag inspect <experiment_id>"
