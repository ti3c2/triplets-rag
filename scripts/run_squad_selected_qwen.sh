#!/usr/bin/env bash
# Local CSV squad_selected experiment matrix:
#   - vanilla RAG vs triplet RAG
#   - Qwen2.5-7B-Instruct vs Qwen2.5-3B-Instruct student
#   - Qwen2.5-32B-Instruct teacher for triplet generation
#
# Expected external services:
#   QWEN32_BASE_URL     teacher endpoint, default http://localhost:7112/v1
#   QWEN7B_BASE_URL     7B student endpoint, default http://localhost:7113/v1
#   QWEN3B_BASE_URL     3B student endpoint, default http://localhost:7115/v1
#   EMBEDDER_BASE_URL   embeddings endpoint, e.g. http://localhost:3300/v1
#   TRIPLET_RAG_EMBED_BATCH_SIZE  optional embedding batch override, e.g. 4

set -euo pipefail

cd "$(dirname "$0")/.."

QWEN32_BASE_URL="${QWEN32_BASE_URL:-http://localhost:7112/v1}"
QWEN7B_BASE_URL="${QWEN7B_BASE_URL:-http://localhost:7113/v1}"
QWEN3B_BASE_URL="${QWEN3B_BASE_URL:-http://localhost:7115/v1}"
EMBEDDER_BASE_URL="${EMBEDDER_BASE_URL:-http://localhost:3300/v1}"

if [ ! -f storage/raw/squad_selected/squad_selected.csv ]; then
    echo "ERROR: expected CSV at storage/raw/squad_selected/squad_selected.csv"
    exit 1
fi

uv sync

run_one() {
    local config="$1"
    local student_url="$2"
    uv run triplet-rag run --config "$config" \
        -o generator.base_url="$QWEN32_BASE_URL" \
        -o embedder.base_url="$EMBEDDER_BASE_URL" \
        -o student.base_url="$student_url"
}

run_one experiment/squad_selected_qwen7b_vanilla "$QWEN7B_BASE_URL"
run_one experiment/squad_selected_qwen7b_triplet "$QWEN7B_BASE_URL"
run_one experiment/squad_selected_qwen3b_vanilla "$QWEN3B_BASE_URL"
run_one experiment/squad_selected_qwen3b_triplet "$QWEN3B_BASE_URL"

echo
echo "=============================================="
echo " Comparison report: squad_selected Qwen matrix"
echo "=============================================="
uv run triplet-rag report --filter experiment_name=squad_selected_qwen \
    --metrics em,f1,rouge_l

echo
echo "RAGAS judge pass example:"
echo "  uv run triplet-rag eval-ragas <experiment_id> \\"
echo "    -j vllm:Qwen/Qwen2.5-32B-Instruct-AWQ \\"
echo "    --base-url \${QWEN32_AWQ_BASE_URL:-http://localhost:7114/v1} \\"
echo "    -m faithfulness,nv_accuracy,nv_response_groundedness,nv_context_relevance,factual_correctness,rouge_score,bleu_score,non_llm_string_similarity,string_present,exact_match"
