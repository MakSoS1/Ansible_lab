# AIOS Track 2 v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver five comparable reservoir-control strategies, an uncertainty-aware active-learning loop, exact OPM/economics validation, and an auditable final Model Z schedule.

**Architecture:** A deterministic contract layer wraps Model Z, OPM, schedule semantics and economics. Five surrogate/search branches share scenario data, constraints and promotion gates. Graph-temporal CEM is the default candidate; MAPPO is a challenger and cannot bypass OPM.

**Tech Stack:** Python 3.12, NumPy, SciPy, pandas, PyTorch, Pydantic, OPM Flow 2026, pytest, Ruff, FastAPI, GitHub Actions, Hugging Face Hub.

**Spec:** `docs/superpowers/specs/2026-09-04-aios-track2-v2-design.md`

## Global Constraints

- Do not read or depend on `ecup*` branches.
- Economic control horizon starts at `2007-01-01`; keep the 1991 workbook as a separate regression diagnostic.
- Reject `WLPR > 500 m3/day`.
- Split train/validation/test by complete `scenario_id` only.
- Record every stochastic seed.
- Final NPV is never sourced directly from surrogate/RL reward.
- Final schedule requires OPM validation and exact economics parity.
- HF data is immutable under a commit/run scoped prefix.
- OPM runs on Linux; M1 is used only where benchmark evidence supports it.

---

## Task 1: Core contracts and TDD safety net

**Files:** `src/aios_track2/{config,schedule,dataset,economics,deck,opm}.py`, `tests/test_core_contracts.py`, `tests/test_opm_deck.py`.

- [x] Write failing tests for config, scenario leakage, `WLPR`, role transitions, economics row exclusion and OPM process isolation.
- [x] Implement minimal typed contracts.
- [x] Add semantic regression for `WCONPROD` LRAT item placement.
- [x] Add safe existing-schedule perturbation for `WCONPROD`/`WCONINJE`.
- [ ] Validate current economics implementation byte-for-byte/numerically against the corrected supplied calculator; add property-tax, pump-size and any date/event rules found in fixtures before calling it exact.

Verification:

```bash
pytest -q tests/test_core_contracts.py tests/test_eclipse_schedule.py tests/test_opm_deck.py
```

## Task 2: DoE, physics diagnostics and scenario storage

**Files:** `src/aios_track2/{doe,physics,active_learning,hfstore}.py`, `configs/doe/pilot.yaml`.

- [x] Implement reproducible smooth Sobol trajectories.
- [x] Implement compensation, water-breakthrough and lagged-connectivity diagnostics.
- [x] Implement value+uncertainty+novelty acquisition.
- [x] Implement immutable HF run prefixes.
- [ ] Inspect the real Model Z schedule and map normalized control dimensions to actual producers/injectors.
- [ ] Generate 32 pilot OPM scenarios and persist raw + normalized outputs to `MakSoS1/aios-track2-runs`.
- [ ] Expand to 128–256 only after pilot QC.

Verification:

```bash
pytest -q tests/test_doe_metrics.py tests/test_physics_active_learning.py
python scripts/validate_model_z.py aios-track2/materials/41_Model_Z_final_OPM.zip
```

## Task 3: Five surrogate/control branches

**Files:** `src/aios_track2/surrogates/*`, `optimization.py`, `marl.py`, `strategy_runner.py`, `strategies.py`.

- [x] Implement linear surrogate.
- [x] Implement GRU surrogate.
- [x] Implement TCN surrogate.
- [x] Implement dependency-light graph-temporal surrogate.
- [x] Implement CEM.
- [x] Implement diagonal CMA-ES.
- [x] Implement shared-graph actor + centralized critic MAPPO/CTDE.
- [x] Make all five strategies executable through one black-box objective protocol.
- [ ] Fit real checkpoints on OPM pilot data and use identical target normalization/action bounds.
- [ ] Build 3–5 seed ensemble for the graph primary.

Verification:

```bash
pytest -q tests/test_surrogates.py tests/test_optimizers_strategies.py tests/test_mappo_agents.py tests/test_strategy_execution.py
python scripts/smoke_bakeoff.py
```

## Task 4: Multi-metric validation and bake-off

**Files:** `src/aios_track2/{metrics,validation,bakeoff}.py`.

- [x] Implement MAE/RMSE/NRMSE/sMAPE.
- [x] Implement Spearman/Kendall/top-k/pairwise rank metrics.
- [x] Implement interval coverage and horizon rollout error.
- [x] Implement hard-gate-first final ranking.
- [ ] Add per-target and field-level metrics from real OPM holdout scenarios.
- [ ] Calibrate 90% intervals on validation scenarios only.
- [ ] Add OOD slices and perturbation robustness.
- [ ] Run all five across the same OPM-confirmation budget and ≥3 seeds where stochastic.

## Task 5: GitHub Actions compute split

**Files:** `.github/workflows/aios-track2-*.yml`, `src/aios_track2/benchmark.py`.

- [x] Add Ubuntu + `macos-15` CI matrix.
- [x] Add M1/MPS compute benchmark.
- [x] Keep OPM installation/inspection on Ubuntu.
- [x] Add manual immutable OPM baseline workflow.
- [ ] Use CI artifacts to choose M1 vs Linux for real surrogate training based on measured throughput, not assumption.

## Task 6: Real Model Z baseline and economics parity

- [ ] Install OPM 2026 package in Ubuntu Actions.
- [ ] Inspect real archive, root deck, include tree and Python OPM bindings.
- [ ] Assert `DIMENS 91 102 59` and 109 wells.
- [ ] Run unchanged baseline once and hash all output artifacts.
- [ ] Parse monthly well results with OPM Python bindings or ESMRY/summary reader.
- [ ] Reproduce corrected CHDD workbook for 2007 start within a documented numerical tolerance.
- [ ] Preserve supplied 1991 calculation as a non-contract regression fixture.

## Task 7: Pilot active-learning cycle

- [ ] Generate 32 valid smooth controls around the existing schedule.
- [ ] Run OPM in bounded parallelism; retry only infrastructure failures, never failed physics as success.
- [ ] Build scenario Parquet and QC report.
- [ ] Train five branches / graph ensemble.
- [ ] Evaluate holdout and quality gate.
- [ ] Search candidate pool, select exploitation+exploration batch, run OPM.
- [ ] Retrain and quantify true NPV gain per additional OPM run.
- [ ] Persist manifests/checkpoints/metrics to HF.

## Task 8: Final optimizer tournament

- [ ] Freeze dataset revision and evaluation protocol.
- [ ] Give CEM/CMA/MAPPO equal surrogate budget.
- [ ] OPM-check top candidates from every branch, not only the apparent surrogate winner.
- [ ] Run sensitivity perturbations around finalists.
- [ ] Choose winner lexicographically: hard gates → OPM NPV → robustness → OPM calls.
- [ ] Re-run exact winning schedule from a clean directory.
- [ ] Verify reported NPV equals the deterministic economics result derived from that clean OPM run.

## Task 9: Delivery/UI

- [x] Add FastAPI-compatible health/strategy endpoints and minimal audit UI shell.
- [ ] Add graph map, time series, uncertainty, bake-off, economics and agent audit panels from real run artifacts.
- [ ] Add final schedule download endpoint.
- [ ] Add Docker/repro command after OPM runtime packaging is fixed.
- [ ] Produce final run manifest with commit, Actions run, HF revision, OPM hashes and NPV.

## Completion verification

Before release, run fresh:

```bash
pytest -q
ruff check src tests scripts
python -m compileall -q src scripts tests
python scripts/validate_model_z.py aios-track2/materials/41_Model_Z_final_OPM.zip
python scripts/smoke_bakeoff.py
```

Then require successful GitHub Actions runs for Linux CI, M1 CI, Model Z inspection and final OPM validation. A green smoke test is not evidence of a high competition NPV; only the clean OPM finalist run closes the task.
