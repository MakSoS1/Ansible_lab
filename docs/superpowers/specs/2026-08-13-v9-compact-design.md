# E-CUP v9 Compact Submission Design

## Goal

Produce a new v9 keeper that preserves the validated gate40 + graph inference behavior as far as possible, removes avoidable packaging/runtime overhead, measures organizer-like end-to-end time from ZIP extraction through `submit.csv`, and publishes only after the compact archive passes the RTX 2060 SUPER gate.

## Evidence motivating the change

- Current v9 keeper: 1,251,659,961 bytes, SHA-256 `925456cde1e47c50dc0141ce64bed5ef00d9f574152f285869ebea2db6935782`.
- Current v8 and v9 are both approximately 1.25 GB and both received platform `Container did not finish in time`.
- The working v7 package was materially smaller because it packaged one cross-encoder plus its minimal runtime closure.
- The previous v9 hard gate started the inference timer only after the archive had already been downloaded and extracted. It therefore did not measure packaging/setup overhead that may be included in the organizer wall-clock budget.

## Architecture

### 1. Audit before mutation

Download the exact immutable v9 keeper and inspect every ZIP member. Record compressed and uncompressed bytes, top contributors, duplicates by SHA-256, and which files are referenced by the packaged runtime. This audit is evidence-only and does not alter the keeper.

### 2. Compact without silent model changes

Build `v9-compact` from the exact v9 keeper. First remove only files proven unreachable by `run.py` and the runtime import closure, plus duplicate/non-runtime metadata where safe. Do not remove any model directory or runtime module merely because it is large.

If the archive cannot reach the target range through dead-file elimination alone, test model-weight compaction as a separate candidate. Any FP16/serialization change must be validated against the original keeper on the same fixture before it can become the final compact keeper.

### 3. Prediction equivalence gate

For a fixed organizer-like fixture, run original v9 and compact v9 and compare:

- row count and pair order exactly;
- finite/non-constant scores;
- exact equality when only dead files were removed;
- if weights were transformed, report max absolute score delta, Spearman/rank agreement, and Macro AP delta on an available held-out validation fixture before accepting.

The preferred final package is exact-prediction-equivalent. A numerically changed package is acceptable only if it is needed to meet runtime/size constraints and validation does not regress materially.

### 4. End-to-end runtime gate

The timer starts immediately before safe ZIP extraction and ends only after a validated `submit.csv` exists. Use the exact organizer image `odsai/ecup26-matching-baseline:1.0`, `--network none`, GPU enabled, read-only submission mount, and the existing 115k/public plus 275k/private fixtures. Record extraction time, inference time, total end-to-end wall time, output validity, GPU identity, and archive SHA.

The compact keeper must beat the prior acceptance envelopes with positive headroom. Runtime has veto power.

### 5. Publishing and memory

Publish the exact final compact ZIP and a manifest to `Maksim123321/e-cup-2026-matching-private/submissions/v9/compact/` using the existing private `HF_TOKEN` GitHub secret. The manifest records archive size/SHA, validation metrics, equivalence evidence, end-to-end runtime evidence, and keeps leaderboard score `null` until the platform produces a measured score.

Update canonical experiment state and Memora so the 1.25 GB v9 is marked `platform-timeout/rejected-for-submission` while retaining its validation evidence, and the compact keeper becomes the submission candidate only after all gates pass.

## Success criteria

- Final archive is materially smaller than 1.25 GB, with a target of <= 700 MiB where achievable without unacceptable validation loss.
- ZIP integrity and path-safety checks pass.
- Output schema/order/finite-score checks pass.
- Original-vs-compact equivalence is measured and documented.
- Full end-to-end 115k and 275k RTX gates pass.
- Exact final bytes are uploaded to private HF under `submissions/v9/compact/` and verified present.
- Canonical state and Memora contain the platform timeout diagnosis and compact keeper evidence.
