# Experiment configs — what each one tests

Index of `configs/experiment/*.yaml` and the research question (RQ) each
addresses. RQs are numbered per `PLAN.md` §1.

## Added 2026-05-08 (cover the unaddressed RQs)

| File | Research question | Notes |
|------|-------------------|-------|
| `squad_triplet_chunk_mediated_pilot.yaml` | RQ2 (q2q vs chunk-mediated) | Same `index_hash` as `squad_triplet_pilot`; only the retriever differs, so the FAISS index is reused. |
| `squad_qa_demo_pilot.yaml` | RQ4 (do contexts inside demos matter?) | `per_triplet_contexts: 0`, `include_fresh_contexts: true` (10 fresh chunks for fair budget). |
| `squad_retrieval_only.yaml` | Isolate retrieval quality | No LLM call; only retrieval metrics meaningful. |
| `squad_questions_only_pilot.yaml` | Extreme QuOTE ablation | Index synthetic questions only. |
| `squad_capacity_gap_vanilla.yaml` | RQ3 (capacity transfer) — baseline | Teacher gpt-4o, student Qwen-2.5-1.5B-Instruct via local vLLM. |
| `squad_capacity_gap_triplet.yaml` | RQ3 — triplet arm | Same teacher/student; preprocessing artifacts shared with the vanilla arm. |

### Open items

- `qa_demo_pilot` budget sanity-check on first run: `inference_strategy/qa_demo.yaml`
  defaults `include_fresh_contexts: true`, so the prompt has 10 fresh
  chunks plus 2 demo (Q,A) pairs. For a strict no-context demo run,
  override with `-o inference.include_fresh_contexts=false`.
- For the capacity-gap pair, vLLM port `8001` is set in `student/qwen_2_5_1_5b.yaml`.
  Make sure nothing else is bound there.

## Pre-existing pilots (for reference)

| File | Role |
|------|------|
| `fixture_smoke.yaml`, `fixture_smoke_triplet.yaml` | Offline pipeline smoke tests on the fixture corpus. |
| `squad_vanilla_pilot.yaml` | RQ1 vanilla baseline at matched 10-context budget. |
| `squad_triplet_pilot.yaml` | RQ1 triplet arm (q2q retrieval). |
| `squad_quote_pilot.yaml` | QuOTE comparator (chunks_and_questions index, vanilla inference). |
| `squad_triplet_full.yaml` | Triplet + RAGAS judge + filtering enabled. |
| `squad_qwen7b_vanilla.yaml`, `squad_qwen7b_triplet.yaml`, `squad_qwen7b_quote.yaml` | Same trio with local Qwen-7B teacher+student (see `RUNBOOK_QWEN7B_SQUAD1K.md`). |
