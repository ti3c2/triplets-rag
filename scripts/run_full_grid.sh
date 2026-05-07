#!/usr/bin/env bash
# Run the full headline grid: vanilla / QuOTE / triplet-q2q / triplet-chunkmed / qa_demo
# on SQuAD-1k. Each grid point is one experiment; results land in storage/experiments/.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "ERROR: .env not found."
    exit 1
fi

uv sync

uv run triplet-rag grid configs/grids/squad_pilot_grid.yaml

uv run triplet-rag report --filter dataset=squad
