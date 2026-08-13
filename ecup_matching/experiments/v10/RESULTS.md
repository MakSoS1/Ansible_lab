# E-CUP Matching — v10 results

Status: **COMPLETED / KEEPER**

Date: 2026-08-13

## Selected production candidate

`no_teacher + frozen target-free graph`

The pair cross-encoder teacher was removed from inference and from the submission archive. CPU structured scoring runs concurrently with the GPU contrastive branch; the missing teacher signal is the frozen v6 no-teacher rank surrogate.

## Frozen quality evidence

| Metric | Value |
|---|---:|
| strict raw OOF Macro AP | `0.5931387077244183` |
| strict graph OOF Macro AP | `0.5950413762943735` |
| graph delta | `+0.0019026685699552` |
| graph-positive immutable folds | `5/5` |
| target-stress graph mean | `0.44961526826354` |
| target-stress std | `0.0016793717` |
| target-stress p05 | `0.44725891257` |
| target-stress p50 | `0.44972458938` |
| target-stress p95 | `0.45217563833` |

These are local validation axes. `0.44961526826354` is the closest current leaderboard-oriented diagnostic, but it is not a leaderboard score or calibrated promise. Actual v10 leaderboard score remains unknown until the competition platform scores this exact archive.

For comparison, the locally stronger `no_contrastive + graph` candidate reached `0.5978943607354008` graph OOF and `0.45353679907723865` stress, but retained the pair-scaled teacher and was rejected for runtime architecture risk.

## Exact keeper archive

Build run `31689478925`.

- filename: `ecup-v10-no-teacher-graph-0.5950413763-submission.zip`;
- bytes: `480249520`;
- SHA-256: `6cebc276f45fc52247db054eb83d2a8110b25d4407cc34b0d5b148a4773c321d`;
- release tag: `ecup-v10-faststack-9de2bc83f878`;
- source SHA: `9de2bc83f878c87703c3290670f042bfdbb70dfc`;
- teacher checkpoint packaged: `false`;
- CPU/GPU overlap: `true`;
- contrastive-only text cache: `true`.

## Runtime proof

All measurements below use the exact archive SHA and `odsai/ecup26-matching-baseline:1.0` on NVIDIA GeForce RTX 2060 SUPER.

### Public-size

Run `MakSoS1/gpu-dispatch#31689794784`:

- rows: `115000`;
- outer inference wall: `173.842174445 s`;
- return code: `0`;
- internal production acceptance: `330 s`;
- nominal organizer budget retained by the project: `360 s`.

The workflow itself is red only because that exploratory run still enforced an intentionally over-strict `<120 s` tuning target. That `<120 s` target is rejected as a keeper criterion; it was never an organizer rule.

### Private-size keeper gate

Run `MakSoS1/gpu-dispatch#31692817075`, artifact `9178292328`:

- rows: `275000`;
- outer inference wall: **`391.608035937 s`**;
- internal production acceptance: `700 s`;
- headroom: **`308.391964063 s`**;
- return code: `0`;
- exact columns/order: valid;
- finite nonconstant scores: valid;
- unique scores: `271964`;
- score min: `-0.0118716200922457`;
- score max: `1.02`;
- teacher checkpoint packaged: `false`;
- result: **PASS**.

Inner phase timing on the same private run:

- input/model load: `16.369 s`;
- contrastive-only text cache: `70.942 s`;
- structured branch: `297.304 s`;
- text+contrastive branch: `345.838 s`;
- critical overlapped feature path: `345.838 s`;
- meta: `1.121 s`;
- graph: `0.390 s`;
- write: `0.409 s`;
- `run.py` total: `364.64 s`.

## Runtime tuning evidence

Corrected 60k sweep run `31690779588` showed batch/worker changes preserve predictions within `rtol=1e-6, atol=1e-7`. The best measured tuning point was contrastive batch `512`, structured workers `8`, but the frozen keeper intentionally keeps the already runtime-safe packaged defaults rather than changing immutable bytes after the keeper gate.

Larger structured chunks (`20k/25k`) were slower and rejected. Earlier tuning runs that failed because the runner lacked `unzip` or `/submission` was absent from `sys.path` are infrastructure failures only and provide no model evidence.

## Hugging Face publication

Run `31693414226`: **SUCCESS**.

Private dataset: `Maksim123321/e-cup-2026-matching-private`.

Verified remote paths:

- `submissions/v10/final/ecup-v10-no-teacher-graph-0.5950413763-submission.zip`;
- `submissions/v10/final/V10_KEEPER.json`.

The publication workflow redownloaded the immutable GitHub Release asset, rechecked exact bytes and SHA-256, verified the no-teacher archive contract, uploaded the ZIP and manifest, and relisted both paths after upload.

## Validation integrity

- human labels: `365654`;
- development rows: `285210`;
- sealed gold rows: `80444`;
- five immutable component-disjoint folds;
- cross-split item overlap: `0`;
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- sealed gold opened: `false`;
- sealed gold rows scored: `0`.

## External objective

The owner-reported v7 leaderboard result is approximately `0.36`. v10 was designed to materially improve transfer while removing the runtime stage that caused platform timeouts. The requested region near `0.5` is still a target, not a claim. Only the competition platform can establish whether this exact v10 archive beats v7 and how close it gets to `0.5`.
