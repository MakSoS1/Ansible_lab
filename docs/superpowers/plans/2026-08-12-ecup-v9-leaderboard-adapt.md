# E-CUP v9 leaderboard-adapt implementation plan

## Goal
Build a new submission after closing v8 as a runtime failure. Optimize for leaderboard transfer and hard runtime safety, not for the largest original-development OOF number.

Known external anchor supplied by the owner: v7 is currently the best scored submission at approximately 0.36 on the leaderboard. This is an external observation, not a validation label and must never be used to fit a model.

## Frozen safety contract
- Keep the existing 285,210-row development universe and 5 component-disjoint outer folds.
- Keep split SHA256 `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Official metric remains unweighted Macro Average Precision over exactly 20 official categories.
- Sealed gold stays unopened and unscored.
- Preserve old strict OOF numbers; v9 adds diagnostics instead of rewriting history.

## Root cause to fix first
The exact v8 gate70 GPU run `31632357768` recorded `run.py` around 731.22 s but outer wall around 820.784 s while the workflow still wrote `full_runtime_gate_passed=true` under a 780 s budget. Therefore the old hard gate is invalid as a completion criterion.

v9 runtime completion policy:
1. Measure outer wall from before organizer-container startup through container exit/output validation.
2. `pass = exit_code == 0 AND output_valid AND wall_seconds <= 700` on the exact 275k fixture.
3. Treat 700 s as the acceptance ceiling to leave >=80 s private-platform margin; a separate watchdog kills at 720 s.
4. Never derive pass only from process exit status or inner `run.py` telemetry.

## Architecture search
Use the already predeclared teacher coverages `(0.25, 0.40, 0.55, 0.70, 0.85, 0.95)`; do not invent a new coverage after reading outcomes.

Candidate family:
- corrected complete runtime closure from v8;
- structured multiprocessing capped at 8 workers;
- CUDA FP16/autocast for neural inference;
- contrastive signal retained;
- selective pair-teacher only on target-free within-category disagreement gate;
- cheap graph postprocess retained only if it remains positive under leakage-free cross-fitting;
- no v7 cross-encoder distillation as the primary target, because the observed leaderboard score says its high local fold diagnostic does not transfer reliably.

Primary Pareto candidates are gate25 and gate40. Existing immutable OOF evidence before v9:
- gate25 strict fusion Macro AP: `0.5947115591000889`, teacher fraction `0.25002279022474666`;
- gate40 strict fusion Macro AP: `0.595505427416499`, teacher fraction `0.400024543318958`.
These are not final v9 scores; graph/target-stress diagnostics are still required.

## Validation v2
Report three clearly separated views:
1. `strict_oof`: unchanged official component-disjoint Macro AP.
2. `target_stress`: deterministic prevalence/density-shift stress on OOF using only development labels plus target-free retrieval/test-like distribution statistics. This is diagnostic, not the official metric.
3. `leaderboard_anchor`: store the owner's observed v7 ~=0.36 only as an external sanity anchor. Never fit or tune weights directly to that single point and never claim a predicted leaderboard score as measured truth.

Selection rule: choose the fastest candidate that is Pareto-competitive under strict OOF and target-stress, then require the exact GPU runtime gate. Runtime safety has veto power.

## Implementation sequence (TDD)
1. Add tests for the outer-wall pass function: 699 s passes, 700 s passes, 700.001 s fails, non-zero exit fails, invalid output fails.
2. Add v9 validation/Pareto code and tests ensuring sealed-gold fields stay false/zero and the external leaderboard anchor cannot enter fitting inputs.
3. Evaluate immutable gate25/gate40 OOF with the same leakage-free graph and target-stress protocol used for higher gates.
4. Select candidate using the frozen rule and record results in `ecup_matching/experiments/v9/`.
5. Production-refit only the selected candidate on all development rows.
6. Build ZIP from import closure, verify CLI/output ordering/finite float scores/CRC/SHA.
7. Copy exact ZIP bytes to private `gpu-dispatch`, run organizer image on exact 275k fixture with corrected outer hard gate on RTX 2060 SUPER.
8. If >700 s, fall back automatically to the next cheaper predeclared candidate (gate25, then no-teacher) without weakening the gate.
9. Publish only a package that passes the exact wall gate. Persist package SHA, size, phase timings, validation metrics and decision in Memora and repository handoff files.

## Completion definition
v9 is complete only when all of the following are true:
- strict validation provenance recorded;
- target-stress diagnostic recorded;
- production refit reproducible;
- exact package integrity checks pass;
- exact 275k organizer-image outer wall <=700 s on `ecup-rtx2060`;
- sealed gold remains unopened;
- Memora ingest + checkpoint succeed;
- final submission archive is published with immutable SHA256 and a retrievable path.
