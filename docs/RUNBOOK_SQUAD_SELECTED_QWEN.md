# Runbook: `squad_selected.csv` with Qwen teacher/students

This runbook uses the local CSV at:

```text
storage/raw/squad_selected/squad_selected.csv
```

The loader expects columns:

```text
id,title,text,query,answer
```

It writes normalized files to `storage/raw/squad_selected/{corpus,queries,qrels}.parquet`.

## Services

Start or route these OpenAI-compatible endpoints:

```bash
vllm serve Qwen/Qwen2.5-32B-Instruct \
  --port 7112 --dtype auto --max-model-len 8192

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 7113 --dtype auto --max-model-len 8192

vllm serve Qwen/Qwen2.5-3B-Instruct \
  --port 7115 --dtype auto --max-model-len 8192

vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
  --port 7114 --dtype auto --max-model-len 8192
```

Set endpoint variables:

```bash
export QWEN32_BASE_URL=http://localhost:7112/v1
export QWEN7B_BASE_URL=http://localhost:7113/v1
export QWEN3B_BASE_URL=http://localhost:7115/v1
export QWEN32_AWQ_BASE_URL=http://localhost:7114/v1
export EMBEDDER_BASE_URL=http://localhost:3300/v1
export EMBEDDER_API_KEY=EMPTY
export VLLM_API_KEY=EMPTY
# Lower this if the embeddings endpoint returns 503/OOM on long batches.
export TRIPLET_RAG_EMBED_BATCH_SIZE=4
# Optional: throttle only teacher question generation. If unset, the code uses
# TRIPLET_RAG_LLM_CONCURRENCY for all LLM batch calls.
export TRIPLET_RAG_QUESTION_GEN_CONCURRENCY=8
```

## Run the initial matrix

```bash
bash scripts/run_squad_selected_qwen.sh
```

This runs:

```text
squad_selected_qwen7b_vanilla
squad_selected_qwen7b_triplet
squad_selected_qwen3b_vanilla
squad_selected_qwen3b_triplet
```

The teacher is always `Qwen/Qwen2.5-32B-Instruct`. The student is 7B or 3B.
The inline metrics are retrieval plus lexical metrics. Run RAGAS as a separate
pass so the same predictions can be judged by multiple future models.

## RAGAS pass

For each completed experiment id:

```bash
uv run triplet-rag eval-ragas <experiment_id> \
  -j vllm:Qwen/Qwen2.5-32B-Instruct-AWQ \
  --base-url "$QWEN32_AWQ_BASE_URL" \
  -m faithfulness,nv_accuracy,nv_response_groundedness,nv_context_relevance,factual_correctness,rouge_score,bleu_score,non_llm_string_similarity,string_present,exact_match \
  --max-workers 4 \
  --timeout 180
```

Context-dependent metrics are evaluated at the `@k` values from the experiment
retrieval metrics by default, so you will see names such as `faithfulness@5`
and `nv_context_relevance@10`.

## Useful overrides

Pilot on the first 100 rows without reading the whole CSV:

```bash
uv run triplet-rag run --config experiment/squad_selected_qwen7b_triplet \
  -o dataset.max_queries=100 \
  -o preprocessing.max_questions_total=500 \
  -o generator.base_url="$QWEN32_BASE_URL" \
  -o embedder.base_url="$EMBEDDER_BASE_URL" \
  -o embedder.batch_size=4 \
  -o student.base_url="$QWEN7B_BASE_URL"
```

Use the AWQ judge metrics inline instead of as a later pass:

```bash
uv run triplet-rag run --config experiment/squad_selected_qwen7b_vanilla \
  -o metrics=ragas_qwen_2_5_32b_awq \
  -o generator.base_url="$QWEN32_BASE_URL" \
  -o embedder.base_url="$EMBEDDER_BASE_URL" \
  -o student.base_url="$QWEN7B_BASE_URL" \
  -o metrics.judge_model.base_url="$QWEN32_AWQ_BASE_URL"
```
