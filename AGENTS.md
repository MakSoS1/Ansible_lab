# AGENTS.md — E-CUP 2026 Matching current handoff

Mandatory entry point for agents working on `ecup-matching-2026`.

## Current verified state — 2026-08-15 — v14

- Competition: ODS E-CUP 2026 Ozon product matching; metric is unweighted Macro Average Precision over 20 categories.
- Historical canonical split: `365,654` human rows; `285,210` development rows; `80,444` sealed-gold rows; 5 component-disjoint folds; split SHA-256 `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- The historical row map was recovered from strict OOF evidence and pinned by SHA-256 `00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f`; current-data dev↔sealed and train↔held item overlap are both `0`.
- Sealed gold remains unopened: `gold_metric_opened=false`, `gold_rows_scored=0`.
- Best measured Public-LB anchor remains **v12 = `0.3798116204`**.
- v13 B is now a measured negative external anchor: fold0 `0.7086611385531062`, Public LB `0.3783781653`, so local ordering inverted relative to v12.
- Current submission-ready candidate is **v14 / `v14-v12-category-gated-residual`**.
- v14 parent is the exact measured v12 one-CrossEncoder model; new work uses a human-only category-gated six-feature lexical residual.
- `matches_llm.parquet` has `11,187,780` rows but exact human-pair overlap `0`; v14 admits **zero** LLM-labelled rows.
- v14 corrected cross-fit probe run `31882322590` passed the frozen promotion rule:
  - v12 fold0 `0.7059297810308699`;
  - v14 diagnostic fold0 `0.7065769713851786`;
  - delta `+0.0006471903543086022`;
  - cross-fit side deltas `+0.000437006267165585` / `+0.000734831086673049`;
  - cross-fit mean `+0.000585918676919317`;
  - six admitted categories;
  - 20/20 full-fold categories non-negative vs v12.
- Exact final file: `ecup-v14-v12-category-gated-residual-submission.zip`.
- Exact final size: `663770301` bytes.
- Exact final SHA-256: `fcaace1a7f0e663b7c9b0b29ca78a768241c3b417b8f4d4a342f52874a29615e`.
- Packaging/runtime run: `31882572941`.
- Binding organizer-shaped Check on **exact final bytes**: `28.810029840000425 s / 60 s`, return code 0, valid output, `910` unique scores, PASS.
- Runtime still contains exactly one v12 ruBERT safetensors checkpoint; the added residual is lightweight and category-gated.
- v14 Public LB is **not measured yet**. Do not claim `>0.5` or even `>v12` externally until ODS returns the score.

## What changed from v13

1. v13 proved that local near-neighbour ordering can invert externally; fold0/Validation-v3 alone no longer promotes a candidate.
2. The historical split is now represented by an explicit recovered row map, not by silently recomputing a new manifest in a changed environment.
3. The historical LLM weak stream was audited before reuse and rejected because no controlled human overlap exists.
4. A2 item-centric / MaxSim research fixed split and endpoint-direction bugs but was not completed through a multi-hour strict Transformer cycle; it is unfinished research, not a quality rejection.
5. Fast v12 residual v1 was rejected by cross-fit.
6. Category-gated v2 found an evaluator bug; after fixing only the evaluator and leaving thresholds unchanged, corrected v2 passed the frozen gate.
7. Final packaging reuses the exact v12 neural parent and adds corrections only in categories with opposite-half item-disjoint evidence.

## Mandatory reading order

1. `ecup_matching/experiments/CURRENT.json`
2. `docs/agent-memory/PROJECT_STATE.md`
3. `docs/agent-memory/EXPERIMENT_INDEX.md`
4. `docs/agent-memory/DECISIONS.md`
5. `ecup_matching/experiments/v14/PLAN.md`
6. `ecup_matching/experiments/v14/RESULTS.md`
7. `ecup_matching/experiments/v14/SAFE_METRICS.json`
8. `docs/agent-memory/SECURITY.md`
9. `docs/agent-memory/ITERATION_PROTOCOL.md`
10. `ecup_matching/SOLUTION_RESEARCH.md`
11. `ecup_matching/BASELINE_CONTRACT.md`

Historical v1–v13 files remain evidence but do not redefine v14 state.

## Non-negotiable invariants

- Never replace the historical split/row map merely to improve a metric.
- Never inspect/use sealed-gold labels during research unless a one-shot rule was frozen beforehand.
- Public leaderboard scores are experiment-level external anchors only; never convert them into row labels.
- Private data, models, OOF predictions, submission ZIPs, persistent Memora DBs and credentials never enter public Git.
- Public source never owns the home RTX runner; GPU work goes through private `MakSoS1/gpu-dispatch`.
- Infrastructure failures are not model-quality failures.
- A local quality gain is not a Public-LB claim.
- Final artifact identity is filename + exact byte count + SHA-256.
- Runtime acceptance is based on the organizer-shaped supplied-item Check on the exact final ZIP; full-item stress is a distinct diagnostic.

## Persistent memory protocol

Public source-backed files are canonical; hardened Memora provides semantic retrieval/history. The supported profile remains pinned local SQLite/TF-IDF as documented in `docs/agent-memory/SECURITY.md`.

After every meaningful KEEP/REJECT/FAIL result, update PLAN/RESULTS/SAFE_METRICS/CURRENT/index/state/decisions as applicable. Only from GREEN repository state run:

```bash
python -m pytest ecup_matching/tests -q
python scripts/memory_policy.py
python scripts/memory_ingest.py
python scripts/memory_checkpoint.py --iteration v14
```

The branch workflow `.github/workflows/ecup-memora-memory.yml` performs the same fail-closed sequence and verifies the private-HF checkpoint.

## Immediate next action

Submit **exactly** `ecup-v14-v12-category-gated-residual-submission.zip` with SHA-256 `fcaace1a7f0e663b7c9b0b29ca78a768241c3b417b8f4d4a342f52874a29615e` to ODS and record its measured Public LB. Until then, v12 `0.3798116204` remains the best observed external anchor.
