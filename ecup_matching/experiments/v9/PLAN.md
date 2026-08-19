# v9 — leaderboard-adapted runtime-safe submission

## Goal

Replace the timed-out v8 path with a candidate selected jointly for transfer robustness and measured runtime. The owner-reported v7 leaderboard score near `0.36` is retained only as an external anchor; it is never used as a training label or fitted calibration target. The desired leaderboard region near `0.5` is a target, not a claimed result.

## Frozen validation contract

- development rows: `285210`;
- sealed gold rows: `80444`;
- five component-disjoint outer folds;
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- official metric: unweighted Macro Average Precision over exactly 20 official categories;
- sealed gold remains unopened and unscored.

## Root-cause correction

The v8 gate70 run completed `run.py` in roughly `731.22 s`, but the true outside-container wall was `820.784 s`. The previous workflow nevertheless wrote `full_runtime_gate_passed=true` because it checked only the `timeout` exit code and output validity. v9 therefore treats the **outer wall** as authoritative.

Runtime completion contract:

- exact candidate bytes;
- organizer image;
- exact `275000`-pair fixture;
- RTX 2060 SUPER runner;
- watchdog: `720 s`;
- acceptance: `wall_seconds <= 700 s`, exit code zero, valid ordered finite output;
- runtime has veto power over quality selection.

## Predeclared candidate family

Teacher coverages were already frozen before v9 research: `25%`, `40%`, `55%`, `70%`, `85%`, `95%`.

Primary v9 Pareto comparison uses gate25 and gate40 because they materially reduce the expensive pair-teacher phase while retaining the established structured, sparse, explicit, contrastive, typed and category/HGB signals. FP16 CUDA inference, structured worker cap 8 and the corrected import-closure runtime are retained.

Target-free graph postprocessing is permitted only with fold-local validation and the frozen config:

- reciprocal-best bonus `0.0`;
- reciprocal-top3 bonus `0.0`;
- endpoint-rank weight `0.02`;
- ambiguity penalty `0.01`.

## Validation v2

Report three axes separately:

1. `strict_oof`: original honest component-disjoint Macro AP;
2. `target_stress`: deterministic prevalence-shift stress using development labels plus target-free retrieval-distribution statistics;
3. `leaderboard_anchor`: owner-reported v7 `~0.36`, stored only as external sanity evidence and excluded from fitting.

Selection rule: choose the candidate that is Pareto-superior on strict OOF and target stress, then require the exact outer-wall runtime gate. If gate40 fails `<=700 s`, automatically fall back to the already predeclared gate25; never relax the runtime threshold post-result.

## Completion

v9 is complete only after package integrity, exact RTX outer-wall gate, canonical documentation, full repository tests, `scripts/memory_policy.py`, hardened Memora ingest/checkpoint and immutable package SHA are all verified. The actual leaderboard score remains unknown until the platform successfully scores the archive.
