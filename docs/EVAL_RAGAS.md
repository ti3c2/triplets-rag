# `triplet-rag eval-ragas` — RAGAS evaluation on completed experiments

Run RAGAS judge metrics on an experiment whose `predictions.jsonl` already
exists, without re-running inference. Designed so multiple judge models can
coexist on the same experiment for comparison.

## When to use this

- You ran an experiment with `metrics: lexical_only` and now want
  faithfulness / groundedness / nv_accuracy numbers.
- You want to compare judges (e.g. `gpt-4o` vs a self-hosted Qwen-32B) on the
  same predictions.
- You're calibrating the judge per `PLAN.md` §9.3 — running the same predictions
  through multiple judges to check inter-judge agreement.

This command does **not** rerun `phase_compute_metrics`. It writes RAGAS
results into a per-judge subfolder (`metrics/ragas/<tag>/`); the original
`metrics/per_query.parquet` and `metrics/aggregate.json` are left untouched.

## Output layout

```
storage/experiments/<experiment_id>/metrics/
├── per_query.parquet          # original (lexical + retrieval)
├── aggregate.json             # original
└── ragas/
    ├── runs.json              # index: tag -> {judge_model, base_url, metrics, ks, ran_at}
    ├── <judge_tag_1>/
    │   ├── per_query.parquet  # this judge's per-query scores
    │   ├── aggregate.json     # this judge's means + bootstrap CIs
    │   ├── judge.json         # full judge config + endpoint + ks + run-config + n_queries
    │   └── inputs/            # (only when --dump-inputs is passed)
    │       ├── context_free.jsonl
    │       ├── k5.jsonl
    │       └── k10.jsonl
    └── <judge_tag_2>/
        └── ...
```

`<judge_tag>` defaults to a sanitized version of the model name
(`gpt-4o`, `Qwen_Qwen2.5-32B-Instruct`, `claude-3-5-haiku-20241022`). Override
with `--judge-tag <name>` for hand-labeled variants.

## Available metrics

Pass any subset to `--metrics`, comma-separated.

| Metric                         | What it measures                                                                  | Use on                            |
|--------------------------------|-----------------------------------------------------------------------------------|-----------------------------------|
| `faithfulness`                 | Claim-decomposed entailment of answer in retrieved contexts. The "groundedness" metric. | Always                            |
| `nv_response_groundedness`     | Whole-answer groundedness (NV prompt, no claim decomposition).                    | Robustness check vs faithfulness  |
| `answer_relevancy`             | Cosine(question, paraphrases-from-answer). Embedding-based.                       | Free-form datasets only           |
| `answer_correctness`           | Mixed judge + embedding score vs gold. Noisy on extractive QA.                    | Free-form datasets                |
| `nv_accuracy`                  | NVIDIA AnswerAccuracy: averaged 0/2/4 judge prompts vs reference.                 | Free-form datasets                |
| `context_precision`            | Judge-rated precision of retrieved contexts.                                      | When qrels are weak/absent        |
| `context_recall`               | Judge-rated recall of retrieved contexts vs gold.                                 | When qrels are weak/absent        |
| `nv_context_relevance`         | NVIDIA context-relevance prompt.                                                   | Free-form / no qrels              |

For SQuAD: `faithfulness` and `nv_response_groundedness` are the load-bearing
ones. `answer_correctness` / `nv_accuracy` add cost without much signal beyond
EM/F1.

### Context-dependent vs context-free metrics

For per-k evaluation (below) we partition the metric set:

| Metric                       | Uses contexts? |
|------------------------------|----------------|
| `faithfulness`               | yes            |
| `context_precision`          | yes            |
| `context_recall`             | yes            |
| `nv_response_groundedness`   | yes            |
| `nv_context_relevance`       | yes            |
| `answer_relevancy`           | no             |
| `answer_correctness`         | no             |
| `nv_accuracy`                | no             |

Context-dependent metrics are replicated per retrieval-k; context-free metrics
run exactly once. An unknown metric is treated as context-free (single pass)
so it doesn't get pointlessly multiplied across k.

## Common invocations

### 1. Default — gpt-4o judge, full classic RAGAS triple

```bash
uv run triplet-rag eval-ragas <experiment_id_or_substring>
```

Equivalent to:
```bash
uv run triplet-rag eval-ragas <id> \
    --judge-model openai:gpt-4o \
    --metrics faithfulness,answer_relevancy,answer_correctness
```

Requires `OPENAI_API_KEY` in `.env`.

### 2. Self-hosted Qwen2.5-32B judge on `localhost:7114`

You serve the model yourself, e.g.:
```bash
vllm serve Qwen/Qwen2.5-32B-Instruct \
    --port 7114 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --served-model-name Qwen/Qwen2.5-32B-Instruct
# vLLM exposes an OpenAI-compatible API at http://localhost:7114/v1
```

Then run the eval:
```bash
uv run triplet-rag eval-ragas <id> \
    --judge-model vllm:Qwen/Qwen2.5-32B-Instruct \
    --base-url http://localhost:7114/v1 \
    --metrics faithfulness,nv_response_groundedness,nv_accuracy
```

`--api-key` is optional; it defaults to `VLLM_API_KEY` from `.env` (which
itself defaults to `EMPTY`, the value vLLM accepts when started without auth).

The endpoint is recorded in `metrics/ragas/Qwen_Qwen2.5-32B-Instruct/judge.json`
and in `metrics/ragas/runs.json`, so a future reader can tell which host produced
which numbers.

#### A note on `answer_relevancy`

It still uses **OpenAI embeddings** (`text-embedding-3-small`) regardless of
the judge LLM. If you want a fully offline run, drop `answer_relevancy` from
`--metrics` (it's a sanity-check metric on free-form data anyway, and noise on
SQuAD).

### 3. Compare two judges on the same experiment

```bash
uv run triplet-rag eval-ragas <id> \
    -j openai:gpt-4o \
    -m faithfulness,nv_response_groundedness

uv run triplet-rag eval-ragas <id> \
    -j vllm:Qwen/Qwen2.5-32B-Instruct \
    --base-url http://localhost:7114/v1 \
    -m faithfulness,nv_response_groundedness
```

After both runs:
```
storage/experiments/<id>/metrics/ragas/
├── gpt-4o/{aggregate.json,per_query.parquet,judge.json}
├── Qwen_Qwen2.5-32B-Instruct/{aggregate.json,per_query.parquet,judge.json}
└── runs.json
```

For a quick side-by-side, read both `aggregate.json` files in a notebook and
join on `query_id` from `per_query.parquet`.

### 4. Anthropic judge

```bash
uv run triplet-rag eval-ragas <id> \
    -j anthropic:claude-3-5-haiku-20241022 \
    -m faithfulness
```

Requires `ANTHROPIC_API_KEY` in `.env`. `--base-url` is ignored for Anthropic.

### 5. Force re-run with the same tag

Trying to write to an existing tag raises an error. Pass `--force` to
overwrite, or set a different `--judge-tag` to keep both:

```bash
uv run triplet-rag eval-ragas <id> -j openai:gpt-4o --force
uv run triplet-rag eval-ragas <id> -j openai:gpt-4o --judge-tag gpt4o_rerun_2
```

### 6. Per-k evaluation (auto-derived from the experiment)

Context-dependent RAGAS metrics depend on *which* retrieved chunks you feed
them. To line up with retrieval@k, the runner replicates those metrics for
each `k` and suffixes the output names: `faithfulness@5`, `faithfulness@10`,
`faithfulness@20`. Context-free metrics (`answer_relevancy`,
`answer_correctness`, `nv_accuracy`) run once.

By default, `ks` are auto-derived from the experiment's
`metrics.retrieval_metrics` (everything with an `@<k>` suffix — `nDCG@10`,
`Recall@5`, etc.). To override:

```bash
uv run triplet-rag eval-ragas <id> --ks 5,10,20
uv run triplet-rag eval-ragas <id> --ks ""        # force a single un-suffixed pass
```

The resulting `aggregate.json` mixes the two scopes:

```json
{
  "faithfulness@5":  {"mean": 0.81, "ci_low": ..., "ci_high": ..., "n": 1000},
  "faithfulness@10": {"mean": 0.83, ...},
  "faithfulness@20": {"mean": 0.84, ...},
  "answer_correctness": {"mean": 0.62, ...}
}
```

`judge.json` records which `ks` were used and which metric names fell into
each partition.

### 7. Tuning concurrency and timeout

The two knobs map directly to RAGAS' `RunConfig` (defaults: `max_workers=16`,
`timeout=180s`). Raise concurrency for OpenAI when you're not rate-limited;
*lower* it for a local vLLM that you don't want to swamp:

```bash
uv run triplet-rag eval-ragas <id> --concurrency 4 --timeout 600
```

If neither flag is passed, RAGAS' defaults are used and no `RunConfig` is
constructed.

### 8. Dumping the inputs RAGAS actually sees

`--dump-inputs` writes the per-row dict (question / answer / contexts /
ground_truth / query_id) sent to `ragas.evaluate` as JSONL. One file per
scope:

```
metrics/ragas/<tag>/inputs/
├── context_free.jsonl         # contexts: []  (these metrics ignore them)
├── k5.jsonl                   # contexts truncated to top-5
├── k10.jsonl
└── k20.jsonl
```

Useful for sanity-checking gold-answer alignment and verifying that the
truncation actually happened. If no per-k is in effect, you get a single
`context_dependent.jsonl` instead.

### 9. Printing judge prompts (debug mode)

`--debug` toggles `langchain_core.globals.set_debug(True)` before calling
`ragas.evaluate`, which dumps every chat-LLM invocation and its response.
We use `set_debug` rather than `set_verbose` because RAGAS bypasses the
LangChain Chain layer that `set_verbose` hooks into — `set_verbose(True)` is
silent for RAGAS.

```bash
uv run triplet-rag eval-ragas <id> --debug -m faithfulness 2>&1 | tee judge_prompts.log
```

Pair with `--dump-inputs` for full reproducibility of what each judge saw.

## Flags reference

| Flag              | Default                                         | Notes |
|-------------------|-------------------------------------------------|-------|
| `--judge-model`   | `openai:gpt-4o`                                 | `<kind>:<model_name>`; kind ∈ {openai, anthropic, vllm, local_hf}. |
| `--metrics`       | `faithfulness,answer_relevancy,answer_correctness` | Comma-separated. See table above. |
| `--judge-tag`     | sanitized `<model_name>`                         | Folder label. Sanitization replaces `/`, ` `, `@` with `_`. |
| `--base-url`      | `VLLM_BASE_URL` (only for vllm/local_hf kinds)  | Per-call override; doesn't mutate env. |
| `--api-key`       | provider env var                                | Per-call override. |
| `--temperature`   | `0.0`                                           |  |
| `--max-tokens`    | `1024`                                          |  |
| `--force` / `-f`  | `false`                                         | Overwrite an existing tag. |
| `--concurrency`   | RAGAS default (16)                              | `RunConfig.max_workers`. |
| `--timeout`       | RAGAS default (180s)                            | `RunConfig.timeout`, per-call. |
| `--debug`         | `false`                                         | `langchain_core.globals.set_debug(True)` — prints judge prompts. |
| `--dump-inputs`   | `false`                                         | Dump `ragas.evaluate` inputs to `inputs/<scope>.jsonl`. |
| `--ks`            | auto-derive from `metrics.retrieval_metrics`    | Comma-separated; `""` forces a single un-suffixed pass. |

All of these are CLI/runner-level knobs — none of them flow into
`MetricsConfig`, so flipping any of them on a finished experiment does **not**
invalidate its `experiment_hash`.

## Programmatic API

```python
from pathlib import Path
from triplet_rag.config import LLMConfig
from triplet_rag.evaluate import run_ragas_on_experiment

agg, out_dir = run_ragas_on_experiment(
    exp_dir=Path("storage/experiments/20260508_134237_..._squad_triplet_pilot"),
    judge_cfg=LLMConfig(kind="vllm", model_name="Qwen/Qwen2.5-32B-Instruct"),
    metric_names=["faithfulness", "nv_response_groundedness", "answer_correctness"],
    judge_base_url="http://localhost:7114/v1",
    concurrency=8,
    timeout=600,
    debug=False,
    dump_inputs=True,
    ks=[5, 10, 20],   # or None → auto-derive from experiment config
)
print(agg)
# {"faithfulness@5": 0.81, "faithfulness@10": 0.83, "faithfulness@20": 0.84,
#  "nv_response_groundedness@5": 0.78, ...,
#  "answer_correctness": 0.62}
```

The in-pipeline call from `phase_compute_metrics` —
`compute_ragas_metrics(predictions, cfg.metrics)` — is unaffected: it still
runs a single pass with the metrics named in `MetricsConfig.ragas_metrics`
and no per-k replication. Per-k lives in the offline rerunner only.
