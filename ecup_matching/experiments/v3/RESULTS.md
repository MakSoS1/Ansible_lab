# E-CUP Matching — Iteration v3 Results

Date: 2026-08-10
Status: in progress

## Baseline

- v2b Macro AP: `0.5010008994958702`
- fixed validation rows: 73,131
- validation train/item overlap: 0
- v2 organizer benchmark: 334 s / 275k pairs / 537,300 items

## GPU backend

Selected backend: home RTX 2060 SUPER through the private isolated
`MakSoS1/gpu-dispatch` self-hosted runner. Hardened WSL/container CUDA check is
verified. End-to-end training smoke and the retained v3 run are recorded below
only after their metric gates pass. Lightning is not retried as primary because
the authenticated account previously returned HTTP 403 for Studio creation and
exposed no reusable Studio.

## Prepared data

Pending.

## Neural stage 1

Pending.

## Model-mined hard-negative stage 2

Pending.

## Blend / gating

Pending.

## Selected fixed-validation result

Pending. v3 will not be marked completed unless the selected candidate strictly exceeds v2b.

## Per-category AP

Pending for all 20 categories.

## Submission package

Pending.

## Organizer offline runtime

Pending. Completion gate: <=585 s on the 275k benchmark slice.

## Failures / diagnostics

None recorded yet.
