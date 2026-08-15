# AGENTS.md — E-CUP 2026 Matching current handoff

Mandatory entry point for agents working on `ecup-matching-2026`.

## Current verified state — 2026-08-15 — v14 new architecture in progress

- Competition: ODS E-CUP 2026 Ozon product matching; metric is unweighted Macro Average Precision over 20 categories.
- Historical canonical split: `365,654` human rows; `285,210` development rows; `80,444` sealed-gold rows; 5 component-disjoint folds; split SHA-256 `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Historical row map SHA-256: `00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f`; dev↔sealed and every train↔held item overlap are `0`.
- Sealed gold remains unopened: `gold_metric_opened=false`, `gold_rows_scored=0`.
- Best measured Public-LB anchor remains **v12 = `0.3798116204`**.
- v13 B is a measured negative external anchor: fold0 `0.7086611385531062`, Public LB `0.3783781653`, below v12 despite the better local score.
- The historical `v14-v12-category-gated-residual` ZIP is a **superseded technical fallback/reference only**. It is not the current requested v14 architecture and must not be submitted as the active v14 while new-architecture research is running.
- `matches_llm.parquet` has `11,187,780` rows but exact human-pair overlap `0`; current v14 architecture screens admit **zero LLM-labelled rows**.

## New architecture direction

The active inference hypothesis is:

```text
unique item -> shared encoder -> reusable token/item representation -> learned compact slots
            -> tiny pair-conditioned bidirectional cross-attention -> symmetric score
```

The expensive Transformer runs independently per unique item and can be cached. Pair-specific reasoning is restored only after compression; this is not the old concatenated pair CrossEncoder.

Measured controls:

- A0 item-centric LateInteraction: `0.5486140975180157` fold0 — REJECTED.
- A1 + human hard-negative repeats: `0.5422162762826607` — REJECTED.
- A2 component closure: cancelled without quality metric after A0/A1 made sampler-only continuation non-credible.
- A3 LateInteraction + category MoE/ranking: `0.3222800376478955` — REJECTED.

Active / queued screens in private `gpu-dispatch` branch `ecup-v14-active`:

1. **A5** — ruBERT cached 12-slot compressed cross-attention + category expert/ranking. Run `31891601603` is the current long fold0 job.
2. **A8** — pinned Granite-97M multilingual retrieval encoder + same compressed cross block. ModernBERT implicit compilation is disabled with `reference_compile=false`.
3. **A5c** — causal control: same compressed cross-attention but no category expert residual and ranking weight `0`.
4. **A12** — Granite compressed cross + fold-train-only typed product `features_v2` fusion.
5. **A6** — LLM-free retrieval-domain distillation; weak parquet contributes only `id1,id2`, never the legacy `target`; soft targets come from fold-safe human-trained teachers.
6. **A10** — multilingual-E5-base compressed-cross reserve with exact-model preflight.

## Frozen fold0 promotion rule

This rule was fixed before the A5 result:

- fold0 `<0.64`: architectural REJECT;
- `0.64 <= fold0 < 0.68`: research-only, no strict OOF;
- fold0 `>=0.68`: credible strict-OOF region; compare screened candidates before promotion;
- v12 fold0 `0.7059297810308699` is the reference, not a Public-LB calibration.

Never claim a Public-LB improvement from local scores.

## Completion contract for a promoted new architecture

A candidate is not submission-ready after fold0. Required sequence:

1. exact five-fold component-disjoint OOF over all `285210` development rows;
2. exact coverage once, no duplicates/missing rows, zero train/held overlap, zero sealed-gold scoring;
3. full-development production refit;
4. one-checkpoint offline ZIP, no raw competition data and no network;
5. exact organizer-shaped supplied-item Check `<60 s` on the final ZIP bytes with valid `id1,id2,predict` and continuous finite scores;
6. private Hugging Face upload and download-back exact size/SHA verification;
7. canonical docs/Memora updated and latest hardened memory workflow GREEN.

Private dispatcher has separate final-only workflows so a research checkpoint cannot be packaged without strict selection evidence:

- `ecup-v14-final-production.yml`
- `ecup-v14-final-package.yml`
- `ecup-v14-final-upload-hf.yml`

## Legacy residual fallback identity

Retain only as fallback/reference:

- `ecup-v14-v12-category-gated-residual-submission.zip`
- bytes `663770301`
- SHA-256 `fcaace1a7f0e663b7c9b0b29ca78a768241c3b417b8f4d4a342f52874a29615e`
- organizer-shaped Check `28.810029840000425 s / 60 s` PASS.

Do not call it the active v14 final. Durable decision D051 supersedes the old final interpretation of D050.

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

Historical v1–v13 and residual-v14 files remain evidence but do not redefine current v14 state.

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
- Current architecture screens use no legacy LLM target labels.

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

Finish A5 run `31891601603`, apply the frozen fold0 promotion rule, then continue A8 → A5c → A12 → A6 → A10 only as required. Do not start strict OOF or package a new architecture below the promotion threshold. v12 `0.3798116204` remains the best measured external result until a new exact archive receives a platform score.
