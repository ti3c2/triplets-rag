# Triplet-RAG Research Repo — Implementation Plan

A research codebase to test whether retrieving teacher-generated `(query, contexts, answer)` triplets as in-context demonstrations helps a smaller student model versus retrieving raw contexts at matched token budget.

## 1. Research questions the repo must answer

The codebase is designed to produce evidence on five questions, and every design choice flows from one of them:

1. **Core hypothesis.** At matched context-token budget, does Triplet-RAG (retrieve N triplets, each with K contexts and a teacher answer) beat Vanilla-RAG (retrieve N×K raw contexts) for a small student model?
2. **Retrieval mechanism.** Is query-to-query triplet retrieval better, worse, or equivalent to chunk-mediated triplet retrieval (retrieve chunks first, surface triplets attached to those chunks)?
3. **Capacity transfer.** Does the gain from triplets grow as the gap between teacher size and student size grows? If gains are flat across student sizes, the method is just "better RAG," not capability transfer.
4. **Filtering sensitivity.** How much of the gain (if any) comes from raw teacher answers versus filtered/verified teacher answers? Bad teacher answers as demonstrations should poison the student.
5. **Failure modes.** Does the student copy demonstration answers verbatim instead of adapting? Is multi-hop (where one triplet's evidence ≠ test query's evidence) particularly bad?

Every metric, every config flag, every experiment script ties back to one of these.

## 2. Top-level repo layout

```
triplet-rag/
├── pyproject.toml
├── README.md
├── PLAN.md                      # this file
├── configs/                     # Hydra configs
│   ├── dataset/
│   ├── chunking/
│   ├── generator/               # teacher LLM configs
│   ├── embedder/
│   ├── indexer/
│   ├── retriever/
│   ├── student/                 # inference LLM configs
│   ├── inference_strategy/
│   ├── metrics/
│   └── experiment/              # composed end-to-end configs
├── src/triplet_rag/
│   ├── __init__.py
│   ├── cli.py                   # typer entrypoint
│   ├── config.py                # pydantic schemas, hashing
│   ├── data/
│   │   ├── loaders.py           # SQuAD, NQ, MultiHop-RAG
│   │   ├── chunking.py
│   │   └── schemas.py           # pydantic record types
│   ├── models/
│   │   ├── manager.py           # lifecycle: start/stop subprocess models
│   │   ├── llm_client.py        # unified interface: OpenAI/Anthropic/vLLM
│   │   ├── embedder_client.py
│   │   └── vllm_server.py       # subprocess wrapper
│   ├── preprocess/
│   │   ├── question_gen.py      # chunk -> questions
│   │   ├── answer_gen.py        # (query, contexts) -> answer
│   │   ├── triplet_builder.py   # orchestrates the pipeline
│   │   └── filters.py           # faithfulness/quality filters
│   ├── index/
│   │   ├── store.py             # FAISS wrapper + parquet metadata
│   │   ├── strategies.py        # chunks_only, questions_only, triplets, ...
│   │   └── builder.py
│   ├── retrieve/
│   │   ├── retriever.py         # base class
│   │   ├── dense.py
│   │   ├── triplet_q2q.py       # query-to-query triplet retrieval
│   │   └── triplet_chunk_mediated.py
│   ├── infer/
│   │   ├── prompts.py           # prompt templates
│   │   ├── strategies.py        # vanilla_rag, triplet_rag, qa_demo_rag, retrieval_only
│   │   └── runner.py
│   ├── evaluate/
│   │   ├── retrieval_metrics.py # ir_measures wrappers
│   │   ├── generation_metrics.py# EM, F1, ROUGE
│   │   ├── judge.py             # LLM-as-judge
│   │   └── aggregator.py
│   ├── experiment/
│   │   ├── runner.py            # end-to-end orchestrator
│   │   └── registry.py          # experiment metadata + paths
│   └── utils/
│       ├── hashing.py
│       ├── io.py                # parquet, jsonl, faiss helpers
│       ├── logging.py
│       └── seeds.py
├── tests/
│   ├── unit/
│   ├── integration/             # tiny-corpus end-to-end smoke tests
│   └── fixtures/
└── scripts/
    ├── run_pilot.sh             # the day-1 sanity check
    └── run_full_grid.sh
```

## 3. Storage layout (single source of truth)

The key insight: preprocessing artifacts are deterministic given config and should be **shared across experiments**. Only inference outputs are per-experiment.

```
storage/
├── raw/
│   └── <dataset_name>/
│       ├── corpus.parquet           # columns: doc_id, title, text
│       ├── queries.parquet          # columns: query_id, query, gold_answer, gold_doc_ids
│       └── qrels.parquet            # standard IR format: query_id, doc_id, rel
├── artifacts/
│   └── <preprocessing_hash>/
│       ├── manifest.json            # the exact config that produced this
│       ├── chunks.parquet           # chunk_id, doc_id, text, span
│       ├── questions.parquet        # question_id, chunk_id, text, generator_model
│       ├── triplets.parquet         # triplet_id, question_id, retrieved_chunk_ids, teacher_answer, faithfulness_score
│       └── embeddings/
│           ├── chunks_<embedder_hash>.npy
│           ├── questions_<embedder_hash>.npy
│           └── meta.json
├── indices/
│   └── <preprocessing_hash>_<embedder_hash>_<strategy>/
│       ├── index.faiss
│       └── id_map.parquet
└── experiments/
    └── <experiment_id>/             # e.g. exp_20260504_a3f9b2_squad-triplet-rag
        ├── config.yaml              # frozen, fully resolved
        ├── refs.json                # which artifacts/indices were used
        ├── predictions.jsonl        # one line per query: {query_id, retrieved_ids, prompt, prediction, latency_ms}
        ├── metrics/
        │   ├── per_query.parquet    # one row per query: query_id, EM, F1, judge_correctness, judge_faithfulness, ...
        │   └── aggregate.json       # means, stds, n
        ├── logs/
        │   ├── run.log
        │   └── model_lifecycle.log
        └── status.json              # PENDING | RUNNING | DONE | FAILED, timestamps
```

`<preprocessing_hash>` is `sha1(canonical_json(preprocessing_config))[:12]`. Same for `<embedder_hash>` and the experiment id (which incorporates the inference config).

A registry file `storage/registry.parquet` indexes all experiments with their config hashes, dataset, status, and headline metrics so you can query "show me all SQuAD experiments with student=Llama-3-8B" without scanning folders.

## 4. Configuration system

Use **Hydra + Pydantic**. Hydra for composition and CLI overrides, Pydantic for runtime validation and JSON schema generation. Configs are pure data — no behavior — and the orchestrator builds objects from them.

A full experiment config is composed from the following groups:

```yaml
# configs/experiment/squad_triplet_rag_pilot.yaml
defaults:
  - dataset: squad_dev_1k
  - chunking: sliding_512_64
  - generator: gpt4o_mini       # teacher for question + answer generation
  - embedder: bge_small
  - indexer: faiss_flat
  - retriever: dense_top_k
  - student: llama_3_8b_instruct
  - inference_strategy: triplet_rag
  - metrics: retrieval_and_generation_with_judge
  - _self_

experiment_name: squad_triplet_rag_pilot
seed: 42
budget:
  total_context_items: 10        # for fair comparison: 2 triplets * 5 contexts == 10 raw contexts
  per_triplet_contexts: 5
  num_triplets: 2

triplet_retrieval_mode: q2q       # q2q | chunk_mediated
filtering:
  enabled: true
  faithfulness_threshold: 0.7
  judge_model: gpt4o_mini

preprocessing:
  num_questions_per_chunk: 5
  question_prompt: complex        # basic | complex (QuOTE prompts)
  answer_prompt: rag_default
```

Pydantic schema mirrors this structure. Every config has a `__hash__` derived from canonical JSON serialization (sorted keys, no defaults baked in). Two configs that produce the same artifacts → same hash → same folder → no recomputation.

**Configurability matrix the repo must support:**

| Axis | Options |
|------|---------|
| Dataset | SQuAD, NQ, MultiHop-RAG (pluggable loader) |
| Chunking | sliding window, sentence-based, semantic; configurable size and overlap |
| Generator (teacher) | OpenAI, Anthropic, local via vLLM (any HF model) |
| Embedder | sentence-transformers, OpenAI embeddings, BGE, E5 |
| Indexer | FAISS Flat, FAISS HNSW, optional BM25 hybrid |
| Indexing strategy | `chunks_only`, `questions_only`, `chunks_and_questions` (QuOTE), `triplets`, `qa_pairs` |
| Retrieval strategy | dense top-k, hybrid, query-to-query for triplets, chunk-mediated for triplets |
| Student | OpenAI, Anthropic, local vLLM |
| Inference strategy | `retrieval_only`, `vanilla_rag`, `triplet_rag`, `qa_demo_rag` |
| Filtering | none, faithfulness threshold, NLI-based |
| Budget | matched by item count or by token count |

## 5. Pipeline phases

The orchestrator runs phases in sequence. Each phase is idempotent: it checks for its expected output artifact and skips if present (unless `--force`).

```
Phase 0: ingest_dataset       → storage/raw/<dataset>/
Phase 1: chunk_corpus         → artifacts/<hash>/chunks.parquet
Phase 2: generate_questions   → artifacts/<hash>/questions.parquet         [needs generator LLM up]
Phase 3: embed_chunks_and_questions
                              → artifacts/<hash>/embeddings/               [needs embedder up]
Phase 4: build_chunk_index    → indices/.../index.faiss
Phase 5: build_triplets       → artifacts/<hash>/triplets.parquet          [needs generator LLM up]
   5a. for each generated question, retrieve top-K chunks via Phase 4 index
   5b. prompt teacher with (question, retrieved_chunks) -> answer
   5c. optionally compute faithfulness score via judge
   5d. optionally filter by threshold
Phase 6: build_inference_index→ indices/.../<strategy>.faiss
   strategy-dependent: question embeddings for q2q, chunk embeddings for chunk-mediated
Phase 7: run_inference        → experiments/<id>/predictions.jsonl         [needs student LLM up]
Phase 8: compute_metrics      → experiments/<id>/metrics/
   8a. retrieval metrics from predictions (no model needed)
   8b. generation lexical metrics (no model needed)
   8c. LLM-as-judge metrics (needs judge LLM up)
Phase 9: register_experiment  → updates storage/registry.parquet
```

Resumability: each phase writes a `_SUCCESS` marker. The orchestrator skips phases whose marker exists and whose inputs haven't changed (input hashes are recorded in the marker).

## 6. Model lifecycle management

This is one of the trickiest parts and deserves explicit treatment. The repo must spin models up only when needed and tear them down before the next phase, otherwise GPU memory will be exhausted on multi-model experiments.

Implementation: a `ModelManager` context manager.

```python
class ModelManager:
    def __init__(self, config: ModelConfig): ...
    def __enter__(self) -> LLMClient:
        if config.kind == "vllm_local":
            self._proc = launch_vllm_server(config)  # subprocess.Popen
            wait_for_health(self._proc, timeout=300)
            return VLLMClient(base_url=...)
        elif config.kind == "openai":
            return OpenAIClient(api_key=...)
        elif config.kind == "anthropic":
            return AnthropicClient(api_key=...)

    def __exit__(self, *args):
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=30)
            torch.cuda.empty_cache()
```

The orchestrator wraps each phase that needs a model in the corresponding context:

```python
with ModelManager(cfg.generator) as teacher:
    run_phase_2_generate_questions(teacher, ...)
    run_phase_5_build_triplets(teacher, ...)
# teacher process is dead here, GPU is free

with ModelManager(cfg.embedder) as embedder:
    run_phase_3_embed(embedder, ...)

with ModelManager(cfg.student) as student:
    run_phase_7_run_inference(student, ...)
```

Note the optimization: phases 2 and 5 both need the teacher, so they share one lifecycle. The orchestrator computes a phase-to-model dependency graph and groups phases that share a model.

vLLM servers are launched with `subprocess.Popen` on a configurable port; health-check via `GET /health`. Use `LITELLM`'s OpenAI-compatible client to talk to vLLM, OpenAI, and Anthropic with one interface (vLLM exposes an OpenAI-compatible API).

GPU memory and concurrency are configurable per model. Default: one model on GPU at a time.

Logging: every spin-up and tear-down writes to `model_lifecycle.log` with PID, GPU memory before/after, and elapsed time, so when something OOMs you can see what was loaded.

## 7. Indexing strategies (concrete behavior)

Each strategy produces a FAISS index and a parquet `id_map` that lets retrieval recover the source content.

**`chunks_only`** — classical baseline. Embed each chunk, index it with `chunk_id`. Retrieval returns chunks.

**`questions_only`** — extreme QuOTE. Embed each generated question, index it with `(question_id, chunk_id)`. Retrieval returns questions, then dereferences to chunks. Useful as an ablation: how much of the gain is from the question itself being a better embedding key vs. the chunk content?

**`chunks_and_questions`** — full QuOTE. Two indices (or one with a type field), retrieval merges and deduplicates by chunk_id at query time.

**`triplets`** — the proposed approach. Has two sub-strategies controlled by `triplet_retrieval_mode`:
- `q2q`: index the synthetic *question* embeddings; each indexed item carries a triplet_id pointing to `(question, chunks, answer)`. At query time: embed test query → find nearest synthetic questions → return their triplets.
- `chunk_mediated`: index *chunk* embeddings; at query time embed test query → find nearest chunks → look up triplets attached to those chunks. If multiple triplets attach to one chunk, score by chunk relevance and rank within.

**`qa_pairs`** — index the concatenation `f"Q: {question} A: {teacher_answer}"`. Tests whether you need the contexts at all or whether QA pairs alone help.

The strategy is a string in the config; a registry maps strings to classes implementing `IndexStrategy.build()` and `IndexStrategy.retrieve()`.

## 8. Inference strategies

Each strategy takes the test query and the retrieved items and produces a prompt for the student. Templates live in `infer/prompts.py` as Jinja2 templates so they're easy to inspect and swap.

**`retrieval_only`** — no LLM call. Just records `retrieved_ids` for retrieval-metric evaluation. Used to isolate retrieval quality from generation quality.

**`vanilla_rag`** — `[query, chunk_1, ..., chunk_k]` → student. The classical baseline.

**`triplet_rag`** — `[query, demo_1, demo_2, ..., demo_n, retrieved_chunks?]` → student. Each demo is a formatted block:
```
Example {i}:
Question: {triplet.question}
Context:
{triplet.contexts joined}
Answer: {triplet.teacher_answer}
```
Whether to *also* include freshly retrieved contexts for the test query is a config flag (`include_fresh_contexts: bool`). Both modes are interesting: pure-demo mode tests whether demos alone carry enough information; demo+fresh tests whether they help on top of standard RAG.

**`qa_demo_rag`** — like `triplet_rag` but demos are just `(question, answer)` without contexts. Tests how much the contexts inside demos matter.

Prompts are versioned (`v1`, `v2`, ...) and the version is part of the experiment hash. Otherwise prompt drift will silently invalidate comparisons.

## 9. Metrics — what we measure and why

Three layers of metrics, chosen so each research question (§1) is covered.

### 9.1 Retrieval metrics (use `ir_measures`)

For datasets with qrels (SQuAD, NQ have gold passages; MultiHop-RAG has gold doc lists):

- **nDCG@{5,10,20}** — graded relevance, the IR standard
- **Recall@{5,10,20}** — did we find the gold doc(s)? Critical for multi-hop
- **MRR** — first-correct-rank, a clean single number
- **Precision@{1,5}** — for span-extractive datasets where top-1 matters

Implementation: format predictions as TREC-run, gold as qrels, call `ir_measures.calc_aggregate(...)` and `ir_measures.iter_calc(...)` for per-query results. Both go to `metrics/per_query.parquet` and `metrics/aggregate.json`.

For triplet retrieval there's a subtlety: what counts as a "retrieved doc" for the metric? Two scoring rules, both reported:
- **Strict**: the chunks attached to the retrieved triplets are the retrieved set. Compare against gold chunks.
- **Lenient**: the union of all chunks across retrieved triplets. Useful for understanding what evidence the student *could* have used.

### 9.2 Generation metrics (lexical)

- **Exact Match (EM)** — strict, useful for SQuAD/NQ
- **Token-F1** — partial credit for SQuAD/NQ
- **ROUGE-L** — for MultiHop-RAG (free-form answers)

Implementation: `evaluate` (HF library) or direct strings (the SQuAD official scorer is short).

### 9.3 LLM-as-judge metrics — justification and scope

**Justification**: lexical metrics fail in three places, and that's exactly where a judge earns its keep:
1. *Free-form answers* (MultiHop-RAG, any synthesis question) — paraphrases score badly on EM/F1 even when correct.
2. *Faithfulness* — "is the answer supported by the retrieved context?" can't be computed from lexical overlap. This is critical for our hypothesis: a student that copies demo answers verbatim might score well on EM but be unfaithful to the test query's actual evidence. Detecting this *is* a research finding.
3. *Demonstration leakage* — "did the student regurgitate the demo answer instead of answering the test query?" needs semantic comparison.

So the judge is justified, but **only for the metrics where lexical methods fail**. We don't use it for SQuAD EM where the gold span is unambiguous; that's just adding noise and cost.

**Judge metrics implemented** (each as a 0–1 score with rationale stored):
- **Correctness** vs gold — agreement with gold answer; used on MultiHop-RAG and as a sanity check on SQuAD/NQ where it should correlate strongly with F1 (validation that our judge prompt isn't broken).
- **Faithfulness** — answer supported by retrieved contexts. Critical and used everywhere.
- **Demo leakage** (only for triplet_rag) — answer copies a demo answer rather than answering the test query. Computed by passing student output and demo answers to the judge and asking for a copy-vs-adapt score.
- **Multi-hop coverage** (MultiHop-RAG only) — does the answer integrate evidence from all gold docs?

Judge model is **separate from the student and the teacher** to avoid self-favoritism. Default: `gpt-4o` (or Claude Sonnet) regardless of teacher/student. This is a config option.

Judge prompts use **structured output** (JSON schema) for deterministic parsing. Each judgment stores `score`, `rationale`, and `judge_model_version` so you can audit.

To control judge variance: prompt templates are zero-shot, deterministic decoding (`temperature=0`), and we run a small calibration set (~100 SQuAD examples) where we know F1 and check that judge correctness correlates >0.8 with F1. If not, the judge prompt is broken and we fix it before reporting any judge numbers.

### 9.4 Per-query and aggregate storage

Every metric is computed per-query and written to `metrics/per_query.parquet` with one row per `(query_id, metric_name, value)`. Aggregate stats (mean, std, n, 95% bootstrap CI) go to `metrics/aggregate.json`. Bootstrap CI matters because absolute differences between methods are often small (5 points on Top-1) and you want to know if they're significant.

### 9.5 Failure-mode-specific metrics

Tied directly to research questions:
- **RQ3 (capacity transfer)**: gain-vs-student-size curve, computed across experiments.
- **RQ5 (demo leakage)**: histogram of per-query demo-leakage scores.
- **Multi-hop specific**: per-hop recall (did we find chunk for hop 1? hop 2?).

## 10. Experiment runner — the CLI

```bash
# Smoke test: tiny corpus, two queries, verifies the pipeline runs
triplet-rag run --config experiment/smoke_test

# Pilot: 1k SQuAD queries, the day-1 sanity check
triplet-rag run --config experiment/squad_triplet_rag_pilot

# Override anything via CLI
triplet-rag run --config experiment/squad_triplet_rag_pilot \
  student=qwen_2_5_7b \
  budget.num_triplets=3 \
  triplet_retrieval_mode=chunk_mediated

# A full grid over students and strategies
triplet-rag grid --grid configs/grids/student_x_strategy.yaml

# Inspect an experiment
triplet-rag inspect <experiment_id>     # prints config, status, headline metrics
triplet-rag report --filter dataset=squad,student=llama_3_8b
                                         # produces a comparison table across experiments
```

`grid` mode generates a Cartesian product of configs, computes hashes, and dispatches them. Resumes are automatic: a grid where 8/12 experiments are already done will only run the remaining 4.

## 11. Test plan

Three test layers.

**Unit tests** for everything pure-functional: chunking boundaries, hash determinism, metric computation against fixed inputs, prompt template rendering, retrieval result formatting.

**Integration tests** with a tiny fixture corpus (10 documents, 20 queries, gold annotations) that runs the full pipeline end-to-end with a stubbed LLM (returns canned responses). Verifies orchestration, file layout, resumability. Runs in <2 minutes on CPU.

**Smoke test with real models** (manually triggered): 50 SQuAD queries, real teacher (gpt-4o-mini), real small student (Qwen-2.5-1.5B). Just checks "does this produce sane numbers and a clean experiment folder?" Run before any real experiment.

## 12. Implementation phases for the coding agent

Six phases, each with a deliverable that can be reviewed before moving on. Treat the boundaries as commit points.

**Phase A — Skeleton and config (1–2 days).**
Repo structure, `pyproject.toml`, Hydra+Pydantic config schemas for every group in §4, hashing utility, the `ModelManager` interface (with stub clients only), CLI skeleton with `run` doing nothing but printing the resolved config. Unit tests for hashing and config validation.

**Phase B — Data and preprocessing (2–3 days).**
Dataset loaders for SQuAD and NQ first; MultiHop-RAG can wait. Chunking. Question generation with a real LLM client (start with OpenAI API to avoid vLLM complexity early). Embedding via sentence-transformers. FAISS index builder. Phase 0–4 of §5 working end-to-end on SQuAD-1k. Integration test with stubbed LLM.

**Phase C — Triplets and retrieval strategies (2–3 days).**
`triplet_builder.py` (phase 5). Both retrieval strategies (q2q, chunk_mediated). All five indexing strategies from §7. Filtering module with a placeholder faithfulness scorer. Unit tests on retrieval determinism.

**Phase D — Inference and lexical metrics (2 days).**
Prompt templates for all four inference strategies. Inference runner using `ModelManager` for the student (OpenAI client first, vLLM second). Lexical metrics via `ir_measures` and standard SQuAD scorer. End-to-end pilot: SQuAD-1k with vanilla_rag and triplet_rag, both numbers in `metrics/aggregate.json`. **This is the first scientifically meaningful checkpoint** — you can already test the core hypothesis with these numbers, even before LLM-as-judge.

**Phase E — LLM-as-judge and full metric suite (1–2 days).**
Judge module with structured output, calibration script, all judge metrics from §9.3. Per-query parquet output, bootstrap CIs, registry parquet.

**Phase F — vLLM, MultiHop-RAG, and grids (2–3 days).**
vLLM subprocess wrapper, lifecycle integration, MultiHop-RAG loader and multi-hop-specific metrics, grid runner, `inspect` and `report` CLI subcommands.

After Phase F: ready to run the full experimental program against the research questions in §1.

## 13. Things the agent should not do

- Don't build a UI. The output is parquet/json; analysis happens in notebooks.
- Don't add a vector DB service (Qdrant, Weaviate). FAISS + parquet is enough at research scale and removes a moving part.
- Don't optimize for speed before correctness. The pilot is 1k queries; everything finishes in under an hour even unoptimized.
- Don't write your own LLM client classes from scratch — use `litellm` for the unified interface.
- Don't bake prompts into Python code as strings. Jinja2 templates in `prompts/` so they're inspectable and diffable.
- Don't share state between phases via Python globals or singletons. Everything goes through the filesystem; phases must be re-runnable from disk.

## 14. Open questions to resolve during Phase B–C, not before

These are intentionally deferred because the right answer depends on what the data looks like:

- **Question deduplication.** Generated questions can be near-duplicates. We can dedupe by embedding similarity (>0.95 cosine) at index time. Decide after looking at the actual generated questions.
- **Triplet count per chunk.** §1 says budget = 2 triplets * 5 chunks = 10. The "right" K and N to test is empirical; the grid script will sweep.
- **Token-budget vs. item-budget matching.** Items are simpler but tokens are fairer (a triplet's contexts may be longer than a raw chunk because they include the question and answer). Default to item-matched in the pilot, add token-matched as a follow-up experiment.
- **Whether the test query's retrieved contexts should also go into the triplet_rag prompt.** Make it a flag (`include_fresh_contexts`); test both and report.

These are noted in the README and in code comments at the relevant decision points so the agent doesn't silently pick one.

## 15. Day-1 deliverable for the human

After Phase D, the agent should produce a single command that runs the pilot:

```bash
triplet-rag run --config experiment/squad_triplet_rag_pilot
triplet-rag run --config experiment/squad_vanilla_rag_pilot
triplet-rag report --filter experiment_name=squad_*_pilot
```

…and outputs a markdown table comparing the two experiments on EM, F1, nDCG@10, and a 95% CI on the difference. That table — produced from real models on a real dataset — is what tells you whether the hypothesis survives contact with reality. Everything in Phases E and F just makes the answer richer; the core question is answerable at the end of Phase D.
