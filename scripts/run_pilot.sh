#!/usr/bin/env bash
# Day-1 pilot: vanilla baseline vs triplet-RAG on SQuAD-1k.
#
# Prereqs:
#   - uv installed (https://docs.astral.sh/uv/)
#   - .env populated with OPENAI_API_KEY
#
# This runs on remote OpenAI APIs only — no GPU needed.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "ERROR: .env not found. Copy .env.example to .env and fill in OPENAI_API_KEY."
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found. See https://docs.astral.sh/uv/"
    exit 1
fi

# Sync deps
uv sync

# Smoke first to verify the pipeline is wired up end-to-end (uses fixture corpus, ~10s).
uv run triplet-rag run --config experiment/fixture_smoke

# Vanilla baseline
uv run triplet-rag run --config experiment/squad_vanilla_pilot

# Triplet-RAG (q2q retrieval)
uv run triplet-rag run --config experiment/squad_triplet_pilot

# QuOTE (chunks_and_questions index, vanilla inference) for context
uv run triplet-rag run --config experiment/squad_quote_pilot

# Show the comparison
uv run triplet-rag report --filter dataset=squad
