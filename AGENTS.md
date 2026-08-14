# AGENTS.md — E-CUP 2026 Matching current handoff

Mandatory entry point for agents working on `ecup-matching-2026`.

## Current verified state — 2026-08-15

- Competition: ODS E-CUP 2026 Ozon product matching; metric is unweighted Macro Average Precision over 20 categories.
- Canonical human split remains immutable: `365,654` human rows; `285,210` development rows; `80,444` sealed-gold rows; 5 component-disjoint folds; cross-split item overlap `0`; split SHA-256 `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Sealed gold remains unopened: `gold_metric_opened=false`, `gold_rows_scored=0`.
- Best observed Public-LB anchor is **v12 = `0.3798116204`**. v7 is `0.3655833314`.
- Current next Public-LB candidate is **v13 B / groupweak**, not yet a verified leaderboard keeper.
- v13 B probe fold0 Macro AP: `0.7086611385531062`.
- Frozen Validation-v3 for v13 B: p05 `0.5690974845`, mean `0.6869505675`.
- Submission file: `ecup-v13-groupweak-v7runtime-submission.zip`; size `663760087` bytes; SHA-256 `f4b7aad36c8d293a3939d9fb2ce7f91cff1bd8381c870015b2f16ea65a17badb`.
- Binding organizer-shaped 1,000-row supplied-item Check: **PASS**, `26.135347286995966 s / 60 s`, valid output, return code 0, 881 unique scores.
- A deliberately stricter diagnostic that scans the full canonical 4.1-GB `items.parquet` timed out at `60.004995396971935 s`; it is not the binding closed-test subset contract. Preserve this distinction in every runtime claim.
- Private-HF round trip is verified: upload workflow `31843423348` downloaded the exact candidate back and revalidated bytes/SHA (`canonical.zip: OK`, `V13_CANDIDATE_HF_ROUNDTRIP_VERIFIED`).
- Runtime shape stays one `ai-forever/ruBert-base` pair CrossEncoder, one tokenizer, one checkpoint; no structured/TF-IDF/graph inference branches.
- v13 B is explicitly `strict_five_fold_confirmed=false` and `strict_final_keeper_claimed=false`. It is the next external calibration/leaderboard candidate.

## Source provenance warning

The v13 package was built from exact public source commit `4e83294eb5f6c31c720f7cbb0220f0f4d0ee3cb1`, production run `31828844182`, packaging run `31829720888` and private-HF run `31843423348`.

The current `ecup-matching-2026` branch tip was later used for the one-time HF bridge and is not ancestry-equivalent to that packaging source. Do **not** infer that branch-tip source equals the already packaged candidate. For reproducibility of v13 runtime/package behavior, use the exact source SHA above.

## What the research established

1. Local human OOF/fold0 is not calibrated to Public LB. v7 `0.70238 -> 0.36558`; v12 `0.70593 -> 0.37981`.
2. The v7→v12 Public-LB gain is `+0.0142282890`; comparable local delta is only `+0.0035495184`.
3. Human prevalence (`~0.25677`) and weak mean target (`~0.24356`) are similar enough that prevalence alone cannot explain the local→LB gap. Candidate degree/hardness is a more important shift.
4. Historical v11 can fail already at Check from fixed/full-item startup cost: exact forensic run `31789001358` hit `60.033 s` before valid output.
5. Target-free graph post-processing is cheap (~`1.29 s / 275k`) but rejected on quality stability: no predeclared graph variant improved v7, v8 and v12 together.
6. The retained v7/v12 weak curriculum destroyed original retrieval-list topology by row sampling/canonical pair orientation and removed ambiguous `0.30–0.70` weak rows.
7. v13 B fixes only topology/orientation by preserving complete retrieval-anchor groups. This produced fold0 `0.7086611386`, +`0.0027313575` over v12 while keeping the exact same inference architecture.
8. C2/ListNet was rejected in controlled equal-exposure testing; complexity must not be retained merely because it is more sophisticated.
9. Heavy v10/v11 structured/TF-IDF runtime is closed. Useful structured/teacher complexity belongs offline at training/distillation time.

## Mandatory reading order

1. `ecup_matching/experiments/CURRENT.json`
2. `docs/agent-memory/PROJECT_STATE.md`
3. `docs/agent-memory/EXPERIMENT_INDEX.md`
4. `docs/agent-memory/DECISIONS.md`
5. `ecup_matching/experiments/v13/PLAN.md`
6. `ecup_matching/experiments/v13/RESULTS.md`
7. `ecup_matching/experiments/v13/SAFE_METRICS.json`
8. `docs/agent-memory/SECURITY.md`
9. `docs/agent-memory/ITERATION_PROTOCOL.md`
10. `ecup_matching/SOLUTION_RESEARCH.md`
11. `ecup_matching/BASELINE_CONTRACT.md`

Historical v1–v12 files remain evidence, but they do not redefine the current external anchor/candidate state.

## Non-negotiable invariants

- Never change split SHA `aae58f...eb55b` to improve a metric.
- Never inspect/use sealed-gold labels during research unless a one-shot rule was frozen beforehand.
- Public leaderboard scores are experiment-level external anchors only; never convert them to row labels.
- Private data, models, OOF predictions, submission ZIPs, persistent Memora DBs and credentials never enter public Git.
- Public source never owns the home RTX runner; GPU work goes through private `MakSoS1/gpu-dispatch`.
- A model-quality claim requires completed evidence. Infrastructure failures are not model scores.
- A runtime claim must state which item-universe contract was tested. Supplied-item subset Check and full-item diagnostic are different tests.
- Final artifact identity is filename + exact byte count + SHA-256, not a friendly version label.

## Persistent memory protocol

Public source-backed files are canonical; hardened Memora provides semantic retrieval/history. The only supported profile remains the pinned local SQLite/TF-IDF configuration documented in `docs/agent-memory/SECURITY.md`.

After every meaningful KEEP/REJECT/FAIL result, update PLAN/RESULTS/SAFE_METRICS/CURRENT/index/state/decisions as applicable. Only from GREEN repository state run:

```bash
python -m pytest ecup_matching/tests -q
python scripts/memory_policy.py
python scripts/memory_ingest.py
python scripts/memory_checkpoint.py --iteration v13
```

The branch workflow `.github/workflows/ecup-memora-memory.yml` performs the same fail-closed sequence and verifies the private-HF checkpoint.

## Immediate next action

Submit the exact v13 B ZIP whose SHA is recorded above to the platform and record the measured Public LB as a new external anchor. Do not call v13 the final keeper before that platform result (and any separately required strict selection evidence) exists.
