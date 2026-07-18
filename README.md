# Triplet-RAG

A research codebase for testing whether teacher-LLM-generated `(query, contexts, answer)` triplets retrieved as in-context demonstrations help smaller student models in retrieval-augmented generation.

## The hypothesis

Standard RAG retrieves raw chunks and asks the student to synthesize an answer. **Triplet-RAG** retrieves teacher-generated demonstrations: each demonstration is a question, the contexts retrieved for it, and the answer the *teacher* produced from those contexts. The smaller student gets to see how a more capable model handled similar questions.

At a matched context-token budget — say, 2 triplets × 5 contexts each (10 contexts total) versus vanilla RAG's 10 raw contexts — does the student do better with the demonstrations? That's the empirical question this repo is built to answer, with enough configurability to also answer adjacent ones.

## What's in here

```
configs/                         # Hydra config groups + composed experiments
src/triplet_rag/
  config.py                      # Pydantic schemas + hash-keyed paths
  data/                          # SQuAD / NQ / MultiHop-RAG / fixture loaders, chunking
  models/                        # LLM client (litellm), embedder, vLLM subprocess wrapper, ManagedLLM
  preprocess/                    # question generation, triplet building, faithfulness filtering
  index/                         # FAISS wrappers; 5 indexing strategies
  retrieve/                      # dense + triplet (q2q & chunk-mediated) retrieval
  infer/                         # 4 inference strategies, the runner
  evaluate/                      # ir_measures retrieval, lexical (EM/F1/ROUGE), RAGAS judge, bootstrap CIs
  experiment/                    # the orchestrator + experiment registry
  utils/                         # hashing, IO, logging, seeds
  prompts/                       # versioned Jinja2 templates
  cli.py                         # `triplet-rag` Typer entrypoint
storage/                         # all artifacts and experiments live here
tests/                           # unit + offline integration tests
scripts/                         # pilot runner + full grid + tests
PLAN.md                          # design rationale (read this first)
```

## Setup

This repo uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# 1) Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Clone and sync
cd triplet-rag
uv sync
# Or with GPU support (vLLM, faiss-gpu):
# uv sync --extra gpu

# 3) Configure credentials
cp .env.example .env
# Edit .env to add OPENAI_API_KEY (and ANTHROPIC_API_KEY if using Claude models)
```

## Quickstart

```bash
# Smoke test (no API keys needed if you've installed the dev extras and want to run the offline tests)
uv sync --extra dev
uv run pytest tests/unit -v
uv run pytest tests/integration -v

# Real smoke test on the synthetic fixture (uses OpenAI by default — costs cents)
uv run triplet-rag run --config experiment/fixture_smoke

# The day-1 pilot: vanilla RAG vs triplet-RAG on SQuAD-1k
bash scripts/run_pilot.sh

# A full grid of strategies
bash scripts/run_full_grid.sh
```

## Storage layout

The single most important design decision: **artifacts are shared across experiments**, keyed by a hash of the preprocessing config. Only inference outputs are per-experiment.

```
storage/
├── raw/<dataset>/                          # corpus, queries, qrels parquet
├── artifacts/<preprocessing_hash>/         # chunks, questions, triplets, embeddings
├── indices/<preproc_hash>_<strategy>_…/    # FAISS index + id_map
├── experiments/<experiment_id>/            # config, predictions, metrics, logs
└── registry.parquet                        # index of all experiments + headline metrics
```

This means swapping the student model (or the inference strategy) re-uses the same triplets — no recomputation. Swapping the chunking, the teacher, or the embedder produces a new preprocessing hash and rebuilds.

`<experiment_id>` looks like `20260504_143022_a3f9b2c1d8e4_squad_triplet_pilot` — timestamp + experiment hash + slug.

## Configuration

Configs are composed with Hydra from groups under `configs/`. The fully-resolved config is validated by Pydantic and frozen for the duration of the run.

The composition groups are:

| Group | What it controls | Examples |
|-------|------------------|----------|
| `dataset` | corpus + queries + qrels | `squad_dev_1k`, `nq_open_1k`, `multihop_rag`, `fixture` |
| `chunking` | how documents become chunks | `sliding_512_64`, `small_200_30` |
| `generator` | the *teacher* LLM (questions + answers) | `gpt4o_mini`, `gpt4o`, `claude_haiku`, `llama_3_8b` |
| `embedder` | the embedding model | `bge_small`, `minilm`, `openai_3_small` |
| `indexer` | which strategy + FAISS variant | `faiss_flat_chunks`, `faiss_flat_triplets`, `faiss_flat_quote`, `faiss_flat_questions` |
| `retriever` | top-k + dedup + triplet retrieval mode | `dense_top_k`, `triplet_top_2`, `triplet_chunk_mediated` |
| `student` | the inference LLM | `gpt4o_mini`, `claude_haiku`, `qwen_2_5_1_5b`, `llama_3_8b` |
| `inference_strategy` | how the prompt is assembled | `vanilla`, `triplet`, `qa_demo`, `retrieval_only` |
| `metrics` | what to compute | `lexical_only`, `full` (adds RAGAS judge) |

Override anything from the CLI:

```bash
uv run triplet-rag run \
  --config experiment/squad_triplet_pilot \
  -o student.model_name=gpt-4o \
  -o budget.num_triplets=3 \
  -o retriever.triplet_retrieval_mode=chunk_mediated
```

## CLI

```bash
triplet-rag run --config <name>      # run one experiment end-to-end
triplet-rag grid <grid.yaml>         # run a sweep (see configs/grids/squad_pilot_grid.yaml)
triplet-rag inspect <experiment_id>  # show config, status, headline metrics
triplet-rag list --filter dataset=squad
triplet-rag report --filter experiment_name=squad_*_pilot --metrics em,f1,faithfulness
```

## Indexing strategies

The five strategies covering the design space:

- **`chunks_only`** — vanilla RAG baseline. Embed and index raw chunks.
- **`questions_only`** — extreme QuOTE. Index only synthetic questions; retrieve them, dereference to chunks.
- **`chunks_and_questions`** — full QuOTE. Both in one index, deduped by chunk_id at query time.
- **`triplets`** — the proposed approach. Each indexed item maps to a `(question, contexts, teacher_answer)` triplet.
- **`qa_pairs`** — ablation: QA pairs without contexts, to test how much the contexts in demos matter.

For `triplets`, two retrieval modes:
- **`q2q`** — embed the test query, find nearest synthetic *questions*, surface their triplets.
- **`chunk_mediated`** — embed the test query, find nearest *chunks*, surface triplets that point to those chunks.

## Inference strategies

- **`retrieval_only`** — no LLM call; just records what was retrieved. For isolating retrieval quality.
- **`vanilla_rag`** — `[query, chunk_1..k]` → student.
- **`triplet_rag`** — `[query, demo_1..n, retrieved_chunks?]` → student. Each demo is `Q / contexts / A`.
- **`qa_demo_rag`** — like triplet_rag but demos drop their contexts; tests how much they mattered.

The `include_fresh_contexts` flag toggles whether the test query's own retrieved chunks are appended to the prompt alongside the demonstrations.

## Metrics — what we report and why

Three layers, each chosen to answer a different research question.

**Retrieval** (`ir_measures`) — `nDCG@{5,10,20}`, `Recall@{5,10,20}`, `MRR`, `Precision@{1,5}`. Standard IR metrics on the doc level. Predictions' chunk-ids are mapped back to doc-ids via the `chunks` table.

**Generation, lexical** — Exact Match, token-F1 (SQuAD-style normalization), ROUGE-L. Cheap and unambiguous on extractive datasets.

**Generation, LLM-as-judge** (RAGAS) — `faithfulness`, `answer_relevancy`, `answer_correctness`. Used **selectively** because lexical metrics already work well on SQuAD/NQ; we add the judge only where it's needed:
- *Faithfulness* — is the answer supported by the retrieved context? Lexical can't see this. Critical for our hypothesis: a student that copies demonstration answers verbatim might score well on EM but be unfaithful to the test query's actual evidence.
- *Free-form datasets* (MultiHop-RAG) — paraphrasing scores badly on EM; the judge catches it.
- *Demo leakage* — answer relevancy + a custom check whether the answer mirrors a demo answer rather than the test question.

The judge model is configured separately from teacher and student to avoid self-favoritism. Default: `gpt-4o`.

Standalone judge runs are pinned to RAGAS 0.3.2 and use its `evaluate()` executor,
matching the known high-throughput evaluation path. `--max-workers N` is passed
directly to `RunConfig.max_workers` (default `TRIPLET_RAG_LLM_CONCURRENCY`), and
batching is disabled. Use `--max-queries N` for a bounded evaluation sample.
`judge.json` records the effective worker and query counts.

**Aggregation** — every metric is computed per-query (`metrics/per_query.parquet`, long format) and aggregated with 1000-iteration bootstrap 95% confidence intervals (`metrics/aggregate.json`). Differences between methods are usually small, so CIs matter.

## Models and lifecycle

`ModelManager` is a context manager that spins up only what each phase needs and tears it down before the next phase, so multi-model experiments don't OOM the GPU.

```python
# Inside the orchestrator, conceptually:
with ManagedLLM(cfg.generator) as teacher:
    questions = generate_questions(chunks, teacher, ...)
    triplets  = build_triplets(questions, teacher, ...)
# teacher process is dead; GPU is free

with ManagedEmbedder(cfg.embedder) as embedder:
    embeddings = embedder.embed(...)

with ManagedLLM(cfg.student) as student:
    predictions = run_inference(student, ...)
```

For OpenAI/Anthropic this is a no-op (just a client). For `kind=local_hf`, it spawns a vLLM subprocess on a configurable port, polls `/health`, and SIGTERMs on exit (with a SIGKILL fallback). Every spin-up writes to `experiments/<id>/logs/model_lifecycle.log`.

All LLM calls go through `tenacity` exponential-backoff retries with a custom predicate that retries on rate-limit / timeout / transient server errors and gives up on auth/parameter errors.

## Reproducibility

- Every experiment writes its fully-resolved config to `experiments/<id>/config.yaml.json`.
- The `preprocessing_hash` and `index_hash` are recorded in `refs.json`, so you can find which artifacts were used.
- `prompt_version` is part of the experiment hash — editing a prompt template requires bumping its version, otherwise old experiments would silently become invalid.
- `seed` is set across `random`, `numpy`, and `torch`.
- All IO is filesystem-mediated; no Python globals between phases.

## Tests

Two layers:

- **Unit tests** (`tests/unit/`): hashing determinism, chunking boundaries, prompt rendering with strict undefined, lexical metric computation, config hash layering. Pure-Python, fast.
- **Integration tests** (`tests/integration/`): end-to-end on the fixture corpus with stubbed `litellm.completion` and a hash-based fake embedder. No network required. Verifies the full orchestrator, file layout, artifact sharing across experiments, and resumability.

```bash
uv sync --extra dev
uv run pytest tests/unit -v
uv run pytest tests/integration -v
```

## See also

- [`PLAN.md`](PLAN.md) for the full design rationale: research questions, every config axis, every metric justified, and the implementation phasing for a coding agent.

## Status

Research code. Optimized for clarity, configurability, and reproducibility — not throughput. The pilot at SQuAD-1k finishes in well under an hour on remote APIs.
