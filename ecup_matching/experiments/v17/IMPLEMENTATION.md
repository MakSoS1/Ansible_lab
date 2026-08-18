# v17 implementation — weak exposure to exact submission

Created: 2026-08-18.

## Immutable training source

The scaled experiment and any production refit use exact source commit:

`f08e7828a9bd9c5fd08c1228987b56810b44ce7b`

The control remains run `32169719512`, created from the preregistered v17 weak-scale driver before any scaled result was known.

## Scaled request

The private execution request pins:

- candidate: `v17-weakscale-x6p8`
- weak presample rows: `3,000,000`
- weak final rows: `1,500,000`
- weak epochs: `1.0`
- weak holdout fraction: `0.05`
- human epochs: `1.0`
- seed: `2026`
- frozen split SHA: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`

The 3M presample keeps the historical 2:1 presample/final ratio while changing the actual training exposure requested by `PLAN.md`.

## Promotion gate

`ecup_matching.ml.v17_promotion.evaluate_promotion` implements the rule already fixed in `PLAN.md`.

The weak axis is the **final model**, `weak_holdout_after_human_phase.macro_average_precision`.
The human axis is `fold0_macro_average_precision`.

Promotion requires both:

1. `scaled_weak - control_weak > 0.005` (strict);
2. `scaled_human - control_human >= -0.005`.

Boundary tests explicitly cover the strict `0.005` case and floating-point tolerance. No result-dependent threshold edit is allowed.

## Production

Only a promoted candidate may enter production. Production uses the already established `v12_production_entry.py --mode plain-v7` wrapper to patch the manifest builder to the frozen split loader, then runs the existing `run_v7_production` path with the scaled weak knobs.

The output remains one `ai-forever/ruBert-base` pair CrossEncoder checkpoint. Exposure changes training only; no extra inference model, graph, residual, or service is introduced.

A `v17-training-policy.json` binds the production checkpoint to the exact source, scaled request, promotion evidence, and unopened-gold assertion.

## Packaging and Check

The private `v17_build_final.py` reuses the static, previously accepted v7 submission packaging source (`4e83294eb5f6c31c720f7cbb0220f0f4d0ee3cb1`) while adding v17 provenance checks.

It refuses to package unless:

- the production metrics match the frozen split and full-development refit contract;
- exactly one `.safetensors` checkpoint exists;
- `v17-training-policy.json` matches the immutable request;
- both preregistered promotion gates passed;
- sealed gold remains unopened.

The built archive is audited for safe ZIP members and required metadata, then executed in `odsai/ecup26-matching-baseline:1.0` on an organizer-shaped 1000-pair fixture with a 60-second wall limit. Output ids/order, finiteness, and non-degenerate score count are validated.

Expected final filename:

`ecup-v17-weakscale-x6p8-v7runtime-submission.zip`

The exact archive SHA-256 is generated only after packaging; no hash is predicted in advance.

## Private executor

Execution is isolated on `MakSoS1/gpu-dispatch:ecup-v17-autopilot`.
The request commit is `e4b115f779823aa9b051fa912f52ba319689ab3d`.
The workflow shares the `ecup-isolated-gpu` concurrency group with the control, so the scaled job cannot consume the single GPU before control run `32169719512` releases it.

The final ZIP is retained on persistent runner storage. Actions artifact export is best-effort because repository artifact quota may lag deletions; artifact failure does not delete the verified runner copy. GitHub Releases are not used.
