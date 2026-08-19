# E-CUP v5 Finalization and ≥0.6000 Validation Design

## Goal

Finish the already-trained six-signal v5 submission, add observable progress telemetry to long stages, and improve the strict leakage-safe validation score from 0.5975445721 to at least 0.6000 without weakening the production model for runtime limits.

## Constraints

- Keep the six retained signals: weak, sparse, explicit, contrastive, teacher, typed_explicit.
- Do not use sealed gold labels for model selection or training.
- Preserve component/item-group leakage protections already used by v5 validation.
- Organizer runtime is H100 80 GB / 20 CPU / 200 GB RAM; do not remove quality-bearing components merely to optimize local runtime.
- Final deliverable must pass the exact organizer image with networking disabled.
- Long stages must expose progress and resource telemetry so bottlenecks can be optimized later without changing model quality.

## Architecture

### 1. Recover and freeze current v5

Normalize GitHub Actions artifact layouts before packaging. The macOS structured artifact stores the model under `out/model_v5_structured.joblib` and the legacy runtime under `legacy/legacy_ecup`; neural artifacts are rooted differently. A small resolver will discover and validate required files rather than assuming a common root layout. The packager then runs the full test suite, builds the six-signal ZIP, and performs an offline organizer-image smoke test.

### 2. Progress telemetry

Introduce a lightweight progress helper usable by training and feature-building stages. It reports phase name, completed/total work, percent, elapsed time, rolling throughput, ETA, RSS/peak RSS, and CUDA memory/utilization when available. Workflows will also emit GitHub step summaries and preserve a machine-readable timing JSON artifact.

The first instrumentation targets the known long stages: structured feature/model construction, contrastive training, teacher training, validation inference, and packaging/smoke.

### 3. Reach honest validation ≥0.6000

Start with the least expensive and least capacity-increasing option: a second-level grouped meta-blend over the six already out-of-fold base signals. Meta-fitting must use group-disjoint folds so the row being scored is never used to choose its fusion weights. Candidate families are deliberately low capacity:

1. global non-negative simplex weights over percentile-ranked signals;
2. globally regularized weights plus category deviations strongly shrunk toward the global weights;
3. logistic/rank stacker with group cross-fitting and strong L2 regularization.

Select the simplest candidate whose aggregate strict OOF Macro AP is highest. The acceptance gate is ≥0.6000 on fully out-of-fold meta predictions, not an in-sample refit score.

If all low-capacity meta-blends remain below 0.6000, use the restored `ecup-rtx2060` runner for one neural improvement at a time while keeping the same sealed split. Priority: improve the weakest/diversifying neural signal rather than retraining everything blindly. Each candidate must produce OOF predictions and be accepted only if the strict aggregate improves.

### 4. Production refit and final freeze

After an honest ≥0.6000 recipe is selected, refit only the fusion/meta parameters on all development OOF rows (and, if a neural candidate was required, train its production counterpart on the allowed development data). Keep the strict cross-fitted score as the headline validation metric; any full-fit score is marked in-sample.

Build a final ZIP with immutable provenance: source commit, split SHA, component artifact IDs/digests, final validation metric, and ZIP SHA-256.

## Failure handling

- Missing/relocated artifact files fail with an explicit inventory rather than a bare `test -f` exit.
- Long jobs checkpoint progress often enough to distinguish slow computation from a dead runner.
- GPU tasks validate CUDA and available VRAM before training and automatically choose a safe batch/accumulation configuration without changing effective training examples or objective.
- Any candidate that leaks gold rows, has non-finite predictions, changes row order, or cannot reproduce its OOF metric is rejected.

## Verification

- Unit tests for artifact resolution, progress math/serialization, and grouped meta-blend leakage guards.
- Full existing `ecup_matching/tests` suite.
- Recomputed strict OOF Macro AP with saved per-category metrics and predictions.
- Exact `odsai/ecup26-matching-baseline:1.0` offline smoke test.
- Final ZIP integrity, size, required-file inventory, and SHA-256 checks.
