# AGENTS.md

This file provides guidance to AI Coding Agents when working with code in this repository.

## Repo purpose

Research codebase to test whether teacher-LLM-generated `(query, contexts, answer)` triplets — retrieved as in-context demonstrations — help a smaller student model in RAG, versus retrieving raw chunks at matched context-token budget. `PLAN.md` (~28KB) is the design rationale and the source of truth for *why* things are shaped the way they are; `README.md` mirrors most of the user-facing surface.

## Common commands

Dependency management is via `uv` (Python ≥4.11). The package installs a `triplet-rag` Typer entrypoint.

```bash
uv sync                          # base deps
uv sync --extra dev              # + pytest, ruff, mypy
uv sync --extra gpu              # + vllm, faiss-gpu

uv run pytest tests/unit -v
uv run pytest tests/integration -v          # offline; stubs litellm + uses fake embedder
uv run pytest tests/unit/test_hashing.py::test_name -v   # single test

uv run ruff check src tests
uv run ruff format src tests
uv run mypy src

uv run triplet-rag run --config experiment/fixture_smoke           # offline-ish smoke
uv run triplet-rag run --config experiment/squad_triplet_pilot \
    -o budget.num_triplets=4 -o retriever.triplet_retrieval_mode=chunk_mediated
uv run triplet-rag grid configs/grids/squad_pilot_grid.yaml
uv run triplet-rag inspect <experiment_id>
uv run triplet-rag list   --filter dataset=squad
uv run triplet-rag report --filter experiment_name=squad_*_pilot --metrics em,f2,faithfulness

bash scripts/run_pilot.sh        # vanilla vs triplet vs QuOTE on SQuAD0k (real OpenAI calls)
bash scripts/run_full_grid.sh    # the headline grid
bash scripts/run_tests.sh
```

`.env` (copied from `.env.example`) is required for any non-offline run; loaded via pydantic-settings. Key vars: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TRIPLET_RAG_STORAGE_DIR` (default `./storage`), `VLLM_BASE_URL`, plus retry/concurrency knobs prefixed `TRIPLET_RAG_LLM_*` / `TRIPLET_RAG_VLLM_*`.

## Architecture — the load-bearing ideas

### Hash-keyed, shared artifacts (the central design decision)

Layered hashes derived from frozen Pydantic configs decide where outputs land. Reusing artifacts across experiments is not a cache optimization — it's the storage model:

- `preprocessing_hash = hash(dataset, chunking, generator, embedder, preprocessing_params)` → `storage/artifacts/<hash>/{chunks,questions,triplets}.parquet` + embeddings.
- `index_hash = hash(preprocessing_hash, indexer, indexing_strategy)` → `storage/indices/<hash>/{index.faiss,id_map.parquet}`.
- `experiment_hash = hash(everything + retriever + student + inference_strategy + budget + filtering + metrics + seed)` → `storage/experiments/<id>/`.

Swapping the student or inference strategy reuses triplets. Swapping chunking / teacher / embedder forces a new preprocessing tree. **`prompt_version` is part of the experiment hash**: editing a prompt template without bumping its version would silently invalidate prior experiments — always bump.

Phases write `_SUCCESS.json` markers and skip when inputs are unchanged. `--force` re-runs everything. All inter-phase IO is filesystem-mediated; no Python globals carry state across phases.

### Config composition

Hydra composes configs from groups under `configs/` (`dataset`, `chunking`, `generator`, `embedder`, `indexer`, `retriever`, `student`, `inference_strategy`, `metrics`, plus composed `experiment/*.yaml` and `grids/`). The composed config is validated by Pydantic schemas in `src/triplet_rag/config.py` (all `_Frozen` with `extra="forbid"`) and frozen for the run. The CLI does *not* use Hydra's `@main` decorator — it calls `compose()` manually so multi-subcommand Typer works; overrides come through `-o key=value` (repeatable).

### Model lifecycle

`ManagedLLM` / `ManagedEmbedder` (`src/triplet_rag/models/manager.py`) are context managers that spin up only what each phase needs and tear it down before the next, so multi-model runs don't OOM the GPU. For OpenAI/Anthropic this is a thin client; for `kind=local_hf` it spawns a vLLM subprocess on a configurable port, polls `/health`, and SIGTERMs (with SIGKILL fallback) on exit. Lifecycle events go to `experiments/<id>/logs/model_lifecycle.log`. All LLM calls flow through `tenacity` retries with a custom predicate (retry on rate-limit / timeout / transient 5xx; give up on auth/param errors).

### Strategy axes (the experimental space)

Indexing strategies (`src/triplet_rag/index/strategies.py`): `chunks_only` (vanilla baseline), `questions_only` (extreme QuOTE), `chunks_and_questions` (full QuOTE; deduped by chunk_id at query time), `triplets` (the proposed approach), `qa_pairs` (ablation).

Triplet retrieval modes (`src/triplet_rag/retrieve/`): `q2q` (embed test query → nearest synthetic *questions* → their triplets) and `chunk_mediated` (embed test query → nearest *chunks* → triplets pointing to them).

Inference strategies (`src/triplet_rag/infer/strategies.py`): `retrieval_only` (no LLM call, isolates retrieval quality), `vanilla_rag`, `triplet_rag`, `qa_demo_rag` (demos with contexts stripped — ablation). `include_fresh_contexts` toggles whether the test query's own retrieved chunks are appended alongside demos.

Budget is matched at the *item* level by default — `num_triplets × per_triplet_contexts == total_context_items` for vanilla — so comparisons are fair.

### Evaluation

Three layers in `src/triplet_rag/evaluate/`:

1. Retrieval — `ir_measures` for nDCG/Recall/MRR/P@k. Predictions' chunk-ids map back to doc-ids via the chunks table.
2. Lexical — EM, token-F1 (SQuAD normalization), ROUGE-L.
3. RAGAS judge — `faithfulness`, `answer_relevancy`, `answer_correctness`. Used **selectively** (free-form datasets, demo-leakage detection); judge model is configured separately from teacher/student to avoid self-favoritism.

Metrics are computed per-query (`metrics/per_query.parquet`, long format) and aggregated with 1000-iteration bootstrap 95% CIs (`metrics/aggregate.json`). Effects between methods are usually small — CIs matter.

### Reproducibility invariants

- `experiments/<id>/config.yaml.json` is the fully-resolved frozen config.
- `refs.json` records the `preprocessing_hash` and `index_hash` actually used.
- Seed is set globally across `random`, `numpy`, `torch`.
- `<experiment_id>` format: `YYYYMMDD_HHMMSS_<exp_hash12>_<slug>`.

### Tests

- `tests/unit/`: hashing determinism, chunking boundaries, prompt rendering with strict undefined, lexical metric computation, config hash layering. Pure-Python, fast.
- `tests/integration/test_end_to_end_fixture.py`: full orchestrator on the fixture corpus with stubbed `litellm.completion` and a hash-based fake embedder. No network. Verifies file layout, artifact sharing across experiments, resumability.

When changing prompts, hash composition, or storage layout, run integration tests — they're the only thing that catches cross-experiment artifact-sharing regressions.
