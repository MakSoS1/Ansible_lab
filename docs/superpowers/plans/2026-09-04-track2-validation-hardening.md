# Track 2 Validation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Track 2 validation competition-relevant: reproducible real-OPM surrogate fidelity, exact organizer-CHDD NPV labels, ranking/regret gates, tail metrics, and an explicit separation between the current 4D baseline policy domain and the full competition action space.

**Architecture:** Preserve the completed 32-scenario frozen OPM DoE as a low-dimensional regression benchmark. Add a deterministic small-data surrogate/evaluator that uses only dates from 2007-01-01, selects hyperparameters without holdout access, refits on train+validation, and evaluates holdout once. Build NPV labels from well-level OPM telemetry through the vendored organizer economics implementation; report density-conversion parity separately. Do not call the 4D benchmark a full-action-space validation.

**Tech Stack:** Python 3.12, NumPy, SciPy, pandas, OPM Flow 2026.04, pytest, Ruff, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-04-aios-track2-v2-design.md`

## Global Constraints

- Do not read or modify any `ecup*` branch.
- History before `2007-01-01` must not influence Track 2 surrogate metrics or controls.
- Holdout membership is frozen before target generation and must never be used for model/hyperparameter selection.
- Final candidate ranking is based on OPM-verified NPV, not surrogate reward.
- `WLPR <= 500 m3/day` is a hard constraint.
- The organizer economics implementation `7.0.2-negative-row-filter` is the source of truth for CHDD/NPV.
- The current 4D global producer/injector scaling family is a low-dimensional benchmark, not evidence of full competition action-space coverage.

---

### Task 1: Honest real-OPM validation metrics

**Files:**
- Create: `src/aios_track2/real_validation.py`
- Create: `tests/test_real_validation.py`

**Interfaces:**
- Produces `align_common_post_start`, `dynamic_delta_report`, `ranking_report`, and `RbfKernelRidge`.

- [ ] Write failing tests proving pre-2007 history is excluded, delta metrics expose aggregate and tail quality separately, ranking reports regret, and kernel regression is deterministic.
- [ ] Run `pytest tests/test_real_validation.py -q` and confirm RED because the production module does not exist.
- [ ] Implement the minimal production module.
- [ ] Run `pytest tests/test_real_validation.py -q` and confirm GREEN.

### Task 2: OPM well telemetry to organizer CHDD

**Files:**
- Create: `src/aios_track2/model_z_economics.py`
- Create: `tests/test_model_z_economics.py`

**Interfaces:**
- Produces `load_model_z_density_map`, `summary_npz_to_chdd_rows`, and `scenario_chdd`.
- Consumes `opm_rows_to_chdd` and `economics_official.compute_calculation`.

- [ ] Write failing tests for PVTNUM expansion, completion-region density assignment, last-report-per-month deduplication, and economic start date 2007-01-01.
- [ ] Run the focused tests and confirm RED.
- [ ] Implement the converter without global-density fallback.
- [ ] Run focused tests and confirm GREEN.

### Task 3: Frozen 32-scenario evaluator

**Files:**
- Create: `scripts/evaluate_real_doe.py`
- Create: `.github/workflows/aios-track2-real-validation.yml`

**Interfaces:**
- Reads all 32 `summary.npz` artifacts from real OPM run `33874390567` and the final Model Z archive.
- Writes a machine-readable report with validation-selection evidence, final holdout metrics, NPV metrics, density parity diagnostics, and gate status.

- [ ] Fit candidate kernels only on the 20 training scenarios.
- [ ] Select hyperparameters using training cross-validation plus the 4 validation scenarios only.
- [ ] Refit the selected model on train+validation.
- [ ] Evaluate the 8 holdout scenarios once.
- [ ] Report per-channel delta-R2, mean/min aggregate channel R2, P10 and worst `scenario x channel`, NPV MAE/max error, Spearman, pairwise accuracy, top-3 recall, and simple regret.
- [ ] Fail closed if any holdout data is touched by model selection code.

### Task 4: Quality gate hardening

**Files:**
- Modify: `src/aios_track2/quality_gate.py`
- Modify: `tests/test_quality_gate.py`

**Interfaces:**
- Extends the gate with tail robustness and NPV regret/error diagnostics while preserving existing fields.

- [ ] Add regression tests first.
- [ ] Require aggregate dynamic R2 >= 0.95, NRMSE <= 0.05, NPV Spearman >= 0.95, pairwise accuracy >= 0.95, top-k recall >= 0.90, physics violations = 0, and explicitly report tail/worst metrics.
- [ ] Do not silently convert a failed tail diagnostic into a passing average.

### Task 5: Competition-domain coverage

**Files:**
- Create: `docs/analysis/track-2-validation-audit.md`
- Modify: `README.md`

**Interfaces:**
- Documents exactly what has and has not been validated.

- [ ] State that the 32-run benchmark spans only four smooth global control variables.
- [ ] State that individual-well control, open/close, cyclic operation, injection redistribution, and producer-to-injector conversion need a richer challenge DoE before claiming full-domain generalization.
- [ ] Add the official evaluation dimensions: NPV, baseline/reference comparison, constraints, perturbation robustness, adaptability, inter-well coordination, and Track 2 surrogate speed/quality.

### Task 6: Verification and PR packaging

**Files:**
- Modify: `src/aios_track2/ecl_summary.py` only for the existing Ruff import failure.
- Update PR #8 description after fresh verification.

- [ ] Run full `pytest -q` and `ruff check src tests scripts` on Ubuntu and macOS CI.
- [ ] Run the frozen real-validation workflow and inspect its report artifact.
- [ ] Do not claim the submission is complete unless both CI and real validation are green with fresh evidence.
