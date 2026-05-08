# Runbook: SQuAD-1k with Qwen2.5-7B-Instruct on local vLLM

End-to-end recipe for running this repo against:

- a vLLM server for `Qwen/Qwen2.5-7B-Instruct` at `localhost:7113`,
  used for both *preprocessing* (teacher: synthesizes
  questions / answers / triplets from chunks) and *inference*
  (student: answers held-out SQuAD questions);
- an OpenAI-compatible embeddings server for
  `jinaai/jina-embeddings-v3` at `localhost:3300`.

Chunking is set to **whole-document** mode (8192-char windows, no
overlap) — SQuAD passages all fit, so each Wikipedia paragraph becomes a
single chunk. To go back to short chunks, swap
`chunking: whole_doc_8192` for `sliding_512_64` in the experiment
config.

---

## 0. One-time setup

```bash
cd /home/groot/Desktop/projects/work/multiplat/triplet-rag

# Install deps (uv >= 0.4)
uv sync

# Copy env template, then edit
cp .env.example .env
```

Edit `.env` and set **only** these (the rest can stay default):

```env
# LLM
VLLM_BASE_URL=http://localhost:7113/v1
VLLM_API_KEY=EMPTY

# Embeddings
EMBEDDER_BASE_URL=http://localhost:3300/v1
EMBEDDER_API_KEY=EMPTY

TRIPLET_RAG_STORAGE_DIR=./storage
# Optional but useful while a 7B is the bottleneck:
TRIPLET_RAG_LLM_CONCURRENCY=16
```

`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` are **not required** — every LLM
in this run is `kind: vllm` and the embedder is
`kind: openai_compatible`, so both go to your local servers.

## 1. Make sure both servers are up

```bash
# LLM
curl -s http://localhost:7113/v1/models | jq .
# Embeddings
curl -s http://localhost:3300/v1/models | jq .
```

If you ever need to (re-)start the LLM:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 7113 --dtype auto --max-model-len 8192
```

## 2. Run the pilot

```bash
bash scripts/run_qwen7b_squad1k.sh
```

That script does, in order:

1. **Fixture smoke test** (~10 s, tiny built-in corpus) — verifies the
   pipeline talks to your vLLM end-to-end before burning time on SQuAD.
2. **`squad_qwen7b_vanilla`** — vanilla dense-retrieval RAG baseline.
3. **`squad_qwen7b_quote`** — QuOTE-style index (`chunks_and_questions`)
   with vanilla inference.
4. **`squad_qwen7b_triplet`** — triplet-RAG (the proposed method),
   q2q retrieval, 2 triplets × 5 contexts each (matched 10-item budget).
5. Prints a comparison report across the three.

Hash-keyed artefact sharing (see `CLAUDE.md`) means the chunks,
synthesized questions, embeddings, and FAISS indices that are common to
multiple runs are computed exactly once and reused.

## 3. Run any single experiment manually

```bash
uv run triplet-rag run --config experiment/squad_qwen7b_triplet
```

Override anything from the CLI with repeatable `-o key=value`:

```bash
uv run triplet-rag run --config experiment/squad_qwen7b_triplet \
    -o budget.num_triplets=3 \
    -o budget.per_triplet_contexts=4 \
    -o budget.total_context_items=12 \
    -o retriever.triplet_retrieval_mode=chunk_mediated
```

Force re-run (ignores `_SUCCESS.json` markers):

```bash
uv run triplet-rag run --config experiment/squad_qwen7b_triplet --force
```

## 4. Where the results live

Storage root is `./storage/` (override with `TRIPLET_RAG_STORAGE_DIR`).

```
storage/
├── artifacts/<preprocessing_hash>/   # shared across experiments
│   ├── chunks.parquet
│   ├── questions.parquet
│   ├── triplets.parquet
│   └── *_embeddings.npy
├── indices/<preprocessing_hash>_<strategy>_<index_hash>/
│   ├── index.faiss
│   └── id_map.parquet
└── experiments/<YYYYMMDD_HHMMSS_<exphash12>_<slug>>/
    ├── config.yaml.json              # frozen resolved config
    ├── refs.json                     # which preprocessing/index it used
    ├── predictions.parquet           # per-query: answer + retrieved ids
    ├── metrics/
    │   ├── aggregate.json            # ★ headline metrics + 95 % CIs
    │   └── per_query.parquet         # long-format scores
    └── logs/
        ├── run.log
        └── model_lifecycle.log
```

The file you usually want is **`storage/experiments/<id>/metrics/aggregate.json`**.

## 5. Querying results from the CLI

```bash
# List runs
uv run triplet-rag list --filter dataset=squad

# Inspect one (prints config + aggregate metrics)
uv run triplet-rag inspect <experiment_id>

# Side-by-side comparison across the three pilot runs
uv run triplet-rag report --filter experiment_name=squad_qwen7b_*

# Specific metrics only
uv run triplet-rag report --filter experiment_name=squad_qwen7b_* \
    --metrics em,f1
```

## 6. Tests

There are no integration tests that hit a real vLLM — `tests/integration/`
stubs out `litellm` and uses a deterministic fake embedder, so it runs
offline:

```bash
uv run pytest tests/unit         -v   # hashing, chunking, prompts, metrics
uv run pytest tests/integration  -v   # full orchestrator, no network
# or:
bash scripts/run_tests.sh
```

The actual end-to-end exercise against your live Qwen server is the
fixture-smoke step in `scripts/run_qwen7b_squad1k.sh` — that's the
closest thing to "run the pipeline against a real model and confirm it
works."

## 7. Files this runbook touches

New configs:
- `configs/generator/qwen_2_5_7b_local.yaml`
- `configs/student/qwen_2_5_7b_local.yaml`
- `configs/embedder/jina_v3_local.yaml`
- `configs/chunking/whole_doc_8192.yaml`
- `configs/experiment/squad_qwen7b_vanilla.yaml`
- `configs/experiment/squad_qwen7b_quote.yaml`
- `configs/experiment/squad_qwen7b_triplet.yaml`
- `scripts/run_qwen7b_squad1k.sh`
- `docs/RUNBOOK_QWEN7B_SQUAD1K.md` (this file)

Source changes (to support an OpenAI-compatible embeddings server):
- `src/triplet_rag/config.py` — added `openai_compatible` to embedder kinds.
- `src/triplet_rag/settings.py` — added `EMBEDDER_BASE_URL` /
  `EMBEDDER_API_KEY`.
- `src/triplet_rag/models/embedder_client.py` — new
  `_embed_openai_compat()` that hits `EMBEDDER_BASE_URL` via litellm.
- `.env.example` — documented the two new env vars.

## 8. Troubleshooting

- **"cannot reach vLLM at ..."** — `VLLM_BASE_URL` in `.env` must end in
  `/v1`. Test with `curl $VLLM_BASE_URL/models`.
- **"cannot reach embeddings server at ..."** — same for
  `EMBEDDER_BASE_URL`. Verify with `curl $EMBEDDER_BASE_URL/models`.
- **"openai_compatible embedder requires EMBEDDER_BASE_URL ..."** —
  you set `kind: openai_compatible` but didn't set the env var.
- **OOM on the vLLM server** — drop `--max-model-len`, or pass
  `--gpu-memory-utilization 0.85` when launching `vllm serve`.
- **Rate / timeout errors** — bump `TRIPLET_RAG_LLM_REQUEST_TIMEOUT` and
  lower `TRIPLET_RAG_LLM_CONCURRENCY` in `.env`.
- **You edited a prompt template** — bump `inference.prompt_version` in
  the experiment config so the experiment hash changes; otherwise prior
  runs get silently invalidated (per `CLAUDE.md`).
- **Re-run from scratch** — `--force`, or delete the relevant
  `storage/experiments/<id>/` (and `storage/artifacts/<hash>/` /
  `storage/indices/<hash>/` if you also want to rebuild upstream
  artefacts).
