#!/usr/bin/env bash
# Quick smoke check: run pytest with offline fixture/stub.

set -euo pipefail

cd "$(dirname "$0")/.."

uv sync --extra dev
uv run pytest tests/unit -v
uv run pytest tests/integration -v
