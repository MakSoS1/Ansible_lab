# AIOS Track 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Track 2 system that learns an uncertainty-aware reservoir surrogate, searches constrained well-control schedules, validates finalists with OPM Flow, and emits an auditable `wells_schedule.inc` plus contract NPV.

**Architecture:** A deterministic simulation/economics core produces scenario-level Parquet data. Sequence and graph-temporal surrogates share one typed prediction interface; CEM/CMA-ES and MARL challengers share one constraint gate and one OPM promotion gate. Role-based agents orchestrate these components, while the LLM explanation layer remains read-only with respect to numerical controls and NPV.

**Tech Stack:** Python 3.12, Pydantic 2, NumPy, pandas, PyArrow, PyTorch 2, PyTorch Geometric, Optuna/CMA, Gymnasium, Stable-Baselines3 or Ray RLlib for the challenger, OPM Flow, FastAPI, React/Vite, Docker, pytest, Ruff, mypy, GitHub Actions, Hugging Face Hub.

**Spec:** `docs/superpowers/specs/2026-09-04-aios-track2-design.md`

## Global Constraints

- Track 2 economic start is `2007-01-01`; preserve a separate 1991 regression fixture for the supplied Model Z workbook discrepancy.
- Reject schedules that can produce `WLPR > 500 m3/day` before promotion.
- Exclude rows with a negative `WLPT_Diff`, `WOMT_Diff`, or `WWIT_Diff` from the economic calculation exactly as required by the corrected methodology.
- Split ML data by complete `scenario_id`; never split rows from one scenario across train, validation, or test.
- Every stochastic entry point accepts an explicit integer seed and records it in the run manifest.
- The final reported NPV must come from an OPM-validated schedule and the deterministic economics module, never directly from the surrogate.
- LLM components cannot write schedule values, bypass constraints, or modify computed NPV.
- Experiment artifacts are immutable under `runs/<git_sha>-<github_run_id>/` in the private Hugging Face Dataset.
- Heavy training starts only from a GitHub Actions workflow; local commands are limited to tests and smoke-sized fixtures.

---

## File Map

```text
pyproject.toml                         packaging, tools and pinned dependency groups
configs/base.yaml                     shared paths, seed and economic defaults
configs/doe/pilot.yaml                32-scenario Sobol pilot
configs/models/{linear,tcn,graph}.yaml model-specific hyperparameters
configs/optimizers/{cem,cma,mappo}.yaml optimizer-specific settings
src/aios_track2/config.py             validated configuration loader
src/aios_track2/deck.py               Model Z metadata and static well graph
src/aios_track2/schedule.py           typed controls, projection and INC writer
src/aios_track2/opm.py                isolated OPM Flow process adapter
src/aios_track2/economics.py          corrected deterministic NPV implementation
src/aios_track2/dataset.py            scenario schema, splits and HF layout
src/aios_track2/doe.py                bounded Sobol trajectory generation
src/aios_track2/surrogates/base.py     common prediction protocol
src/aios_track2/surrogates/linear.py   non-neural reference baseline
src/aios_track2/surrogates/tcn.py      temporal neural baseline
src/aios_track2/surrogates/graph.py    graph-temporal ensemble member
src/aios_track2/optimization.py        CEM/CMA-ES and OPM promotion
src/aios_track2/agents.py              role-based orchestration state machine
src/aios_track2/api.py                 FastAPI endpoints and audit stream
ui/                                   React dashboard
tests/                                 unit, contract, integration and smoke tests
.github/workflows/aios-*.yml           CI, DoE, training, optimization and validation
```

## Task 1: Reproducible Python Project and Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `configs/base.yaml`
- Create: `src/aios_track2/__init__.py`
- Create: `src/aios_track2/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path: Path) -> AppConfig`
- Produces: `AppConfig.seed: int`, `AppConfig.economic_start: date`, `AppConfig.paths: PathsConfig`

- [ ] **Step 1: Write the configuration contract test**

```python
from datetime import date
from pathlib import Path

from aios_track2.config import load_config


def test_base_config_has_track2_contract() -> None:
    cfg = load_config(Path("configs/base.yaml"))
    assert cfg.seed == 42
    assert cfg.economic_start == date(2007, 1, 1)
    assert cfg.max_wlpr_m3_day == 500.0
    assert cfg.hf_dataset_id == "MakSoS1/aios-track2-runs"
```

- [ ] **Step 2: Run the isolated test and confirm import failure**

Run: `python -m pytest tests/test_config.py -q`

Expected: FAIL because `aios_track2.config` does not exist.

- [ ] **Step 3: Implement strict Pydantic configuration**

```python
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deck: Path
    work_dir: Path


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int
    economic_start: date
    max_wlpr_m3_day: float
    hf_dataset_id: str
    paths: PathsConfig


def load_config(path: Path) -> AppConfig:
    return AppConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
```

- [ ] **Step 4: Add exact base values**

```yaml
seed: 42
economic_start: 2007-01-01
max_wlpr_m3_day: 500.0
hf_dataset_id: MakSoS1/aios-track2-runs
paths:
  deck: aios-track2/work/model-z/BASE.DATA
  work_dir: runs/local
```

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/test_config.py -q`

Expected: `1 passed`.

Commit: `git commit -m "build: scaffold AIOS track 2 project"`

## Task 2: Model Z Parser and Static Well Graph

**Files:**
- Create: `src/aios_track2/deck.py`
- Test: `tests/test_deck.py`
- Test fixture: `tests/fixtures/minimal.DATA`

**Interfaces:**
- Produces: `parse_deck(path: Path) -> DeckMetadata`
- Produces: `build_well_graph(metadata: DeckMetadata, radius_m: float) -> WellGraph`
- `DeckMetadata.dimensions: tuple[int, int, int]`
- `DeckMetadata.wells: tuple[Well, ...]`

- [ ] **Step 1: Add a minimal Eclipse deck fixture**

```text
RUNSPEC
DIMENS
 3 2 1 /
SCHEDULE
WELSPECS
 'P1' 'G' 1 1 1* 'OIL' /
 'I1' 'G' 3 2 1* 'WATER' /
/
END
```

- [ ] **Step 2: Write parser and graph tests**

```python
from pathlib import Path

from aios_track2.deck import build_well_graph, parse_deck


def test_parse_dimensions_and_wells() -> None:
    metadata = parse_deck(Path("tests/fixtures/minimal.DATA"))
    assert metadata.dimensions == (3, 2, 1)
    assert [well.name for well in metadata.wells] == ["P1", "I1"]


def test_graph_is_symmetric() -> None:
    graph = build_well_graph(parse_deck(Path("tests/fixtures/minimal.DATA")), radius_m=10_000)
    assert set(graph.edges) == {("P1", "I1"), ("I1", "P1")}
```

- [ ] **Step 3: Confirm the tests fail before implementation**

Run: `python -m pytest tests/test_deck.py -q`

Expected: FAIL because `parse_deck` is undefined.

- [ ] **Step 4: Implement DIMENS/WELSPECS parsing and deterministic edge ordering**

Use frozen dataclasses for `Well`, `DeckMetadata`, and `WellGraph`. Resolve nested `INCLUDE` paths relative to the including file, reject include cycles, and sort wells by name before constructing edges. Compute initial edges from `(i, j)` grid distance; add completion-overlap and lag-correlation weights in Task 6 without changing edge identity.

- [ ] **Step 5: Verify the fixture and real Model Z metadata**

Run: `python -m pytest tests/test_deck.py -q`

Expected: `2 passed`.

Run: `python -m aios_track2.deck aios-track2/work/model-z/BASE.DATA --summary`

Expected JSON fields: `"dimensions": [91, 102, 59]` and `"well_count": 109`.

- [ ] **Step 6: Commit**

Commit: `git commit -m "feat: parse Model Z and build well graph"`

## Task 3: Typed Schedule and Hard Constraint Gate

**Files:**
- Create: `src/aios_track2/schedule.py`
- Test: `tests/test_schedule.py`
- Test fixture: `tests/fixtures/expected_schedule.inc`

**Interfaces:**
- Produces: `project_schedule(schedule: Schedule, constraints: ConstraintSet) -> ProjectionResult`
- Produces: `write_schedule_inc(schedule: Schedule, path: Path) -> str`
- `ProjectionResult.schedule: Schedule`
- `ProjectionResult.violations: tuple[Violation, ...]`

- [ ] **Step 1: Write rejection and deterministic rendering tests**

```python
from datetime import date

from aios_track2.schedule import Control, Schedule, project_schedule, write_schedule_text


def test_wlpr_above_limit_is_rejected() -> None:
    schedule = Schedule(controls=(Control(date=date(2007, 1, 1), well="P1", wlpr=501.0),))
    result = project_schedule(schedule)
    assert result.accepted is False
    assert result.violations[0].code == "WLPR_LIMIT"


def test_schedule_text_is_stable() -> None:
    schedule = Schedule(controls=(Control(date=date(2007, 1, 1), well="P1", status="OPEN", wlpr=250.0),))
    assert write_schedule_text(schedule) == "DATES\n  1 JAN 2007 /\n/\nWCONPROD\n  'P1' 'OPEN' 'LRAT' 1* 250.000 /\n/\n"
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/test_schedule.py -q`

Expected: FAIL because the schedule module does not exist.

- [ ] **Step 3: Implement immutable controls and projection**

Define `Control` with date, well, status, role, liquid rate, injection rate, BHP/THP and conversion flag. Sort by `(date, well)`. Reject unknown wells, duplicated controls, reverse injector-to-producer conversion, negative rates, pressure bounds, infrastructure overflow and `WLPR > 500`. Return every violation in stable order.

- [ ] **Step 4: Implement the INC writer and parser round trip**

The writer must emit `DATES`, `WCONPROD`, `WCONINJE`, `WELOPEN` and conversion records using fixed three-decimal formatting. Add `parse_schedule_inc(text: str) -> Schedule`; assert `parse_schedule_inc(write_schedule_text(schedule)) == schedule` for producer, injector, close/open and conversion cases.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/test_schedule.py -q`

Expected: all schedule tests pass.

Commit: `git commit -m "feat: add constrained schedule contract"`

## Task 4: Isolated OPM Flow Adapter

**Files:**
- Create: `src/aios_track2/opm.py`
- Test: `tests/test_opm.py`
- Test fixture: `tests/fixtures/fake_flow.py`

**Interfaces:**
- Produces: `run_flow(request: FlowRequest) -> FlowResult`
- `FlowResult` includes `status`, `runtime_seconds`, `stdout_sha256`, `output_files`, and parsed monthly data path.

- [ ] **Step 1: Create a fake executable that records arguments and writes deterministic output**

```python
from pathlib import Path
import sys

Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
Path(sys.argv[2], "CASE.UNSMRY").write_bytes(b"fixture")
print("OPM FLOW FIXTURE OK")
```

- [ ] **Step 2: Write tests for success, timeout and non-zero exit**

```python
from pathlib import Path

from aios_track2.opm import FlowRequest, run_flow


def test_flow_result_is_hashed(tmp_path: Path) -> None:
    result = run_flow(FlowRequest(deck=Path("tests/fixtures/minimal.DATA"), output_dir=tmp_path, timeout_seconds=10))
    assert result.status == "success"
    assert len(result.stdout_sha256) == 64
```

- [ ] **Step 3: Run tests and confirm failure**

Run: `python -m pytest tests/test_opm.py -q`

Expected: FAIL because `FlowRequest` is undefined.

- [ ] **Step 4: Implement subprocess isolation**

Invoke `flow` with an explicit argument list, never a shell string. Copy the immutable deck into a scenario-specific directory, set a timeout, capture stdout/stderr, hash logs and all produced files, and return `timeout` or `failed` without converting either to success.

- [ ] **Step 5: Add an optional real-OPM smoke marker**

Run: `python -m pytest tests/test_opm.py -q`

Expected: unit tests pass; `pytest -m opm` runs only when `flow --version` succeeds.

- [ ] **Step 6: Commit**

Commit: `git commit -m "feat: isolate and audit OPM Flow runs"`

## Task 5: Corrected NPV Engine

**Files:**
- Create: `src/aios_track2/economics.py`
- Copy fixture from supplied archive to: `tests/fixtures/economics/example_input.xlsx`
- Create: `tests/fixtures/economics/model_z_expected.json`
- Test: `tests/test_economics.py`

**Interfaces:**
- Produces: `calculate_npv(monthly: DataFrame, config: EconomicsConfig) -> NpvResult`
- `NpvResult.npv_mrub: Decimal`
- `NpvResult.annual: DataFrame`
- `NpvResult.excluded_rows: tuple[RowReference, ...]`

- [ ] **Step 1: Write exact tests for row exclusion and discount start**

```python
from datetime import date
from decimal import Decimal

import pandas as pd

from aios_track2.economics import EconomicsConfig, calculate_npv


def test_negative_diff_row_is_fully_excluded() -> None:
    monthly = pd.DataFrame([
        {"DATA": "2007-01-01", "well": "P1", "WLPT_Diff": -1.0, "WOMT_Diff": 2.0, "WWIT_Diff": 0.0, "WLPR": 10.0, "WWIR": 0.0},
        {"DATA": "2007-02-01", "well": "P1", "WLPT_Diff": 3.0, "WOMT_Diff": 2.0, "WWIT_Diff": 0.0, "WLPR": 10.0, "WWIR": 0.0},
    ])
    result = calculate_npv(monthly, EconomicsConfig.default_track2())
    assert len(result.excluded_rows) == 1
    assert result.annual.loc[2007, "oil_t"] == Decimal("2.0")


def test_track2_discount_factor_is_one_in_2007() -> None:
    assert EconomicsConfig.default_track2().discount_factor(date(2007, 12, 31)) == Decimal("1")
```

- [ ] **Step 2: Confirm tests fail**

Run: `python -m pytest tests/test_economics.py -q`

Expected: FAIL because the economics module does not exist.

- [ ] **Step 3: Port the corrected calculator with Decimal arithmetic**

Use the supplied constants verbatim, aggregate monthly volumes by calendar year, calculate pump/start/stop/conversion events from ordered well states, apply 25% profit tax, 2.2% property tax and 10% discounting. Keep unrounded internal values and round only serialized output.

- [ ] **Step 4: Add 1991 and 2007 Model Z regression fixtures**

Store both expected totals with source workbook hash and a `method` field. Assert that the CLI requires `--economic-start` and that the default Track 2 command selects 2007.

- [ ] **Step 5: Verify supplied and new tests**

Run: `python -m pytest tests/test_economics.py -q`

Expected: every corrected-method test passes and both baseline variants remain distinguishable.

- [ ] **Step 6: Commit**

Commit: `git commit -m "feat: implement corrected track 2 NPV"`

## Task 6: Scenario Schema, DoE and Private HF Layout

**Files:**
- Create: `src/aios_track2/dataset.py`
- Create: `src/aios_track2/doe.py`
- Create: `configs/doe/pilot.yaml`
- Test: `tests/test_dataset.py`
- Test: `tests/test_doe.py`

**Interfaces:**
- Produces: `generate_scenarios(config: DoeConfig, graph: WellGraph) -> tuple[Schedule, ...]`
- Produces: `write_scenario(result: ScenarioResult, root: Path) -> ScenarioManifest`
- Produces: `split_scenarios(ids: Sequence[str], seed: int) -> SplitAssignment`

- [ ] **Step 1: Write leakage and determinism tests**

```python
from aios_track2.dataset import split_scenarios


def test_scenario_split_has_no_overlap() -> None:
    split = split_scenarios([f"s{i:03d}" for i in range(20)], seed=42)
    assert set(split.train).isdisjoint(split.validation)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.validation).isdisjoint(split.test)
    assert split == split_scenarios([f"s{i:03d}" for i in range(20)], seed=42)
```

- [ ] **Step 2: Confirm tests fail**

Run: `python -m pytest tests/test_dataset.py tests/test_doe.py -q`

Expected: FAIL because dataset and DoE modules do not exist.

- [ ] **Step 3: Implement the immutable manifest schema**

Require `scenario_id`, seed, deck/schedule hashes, simulator version, status, runtime, NPV, violations and GitHub run URL. Write dynamic rows to `scenarios/<scenario_id>/monthly.parquet`; write one atomic `manifest.json` only after every expected file hash has been computed.

- [ ] **Step 4: Implement bounded Sobol trajectory generation**

Generate 32 pilot schedules. Use graph clusters, quarterly knots, cubic interpolation and a maximum 20% change per quarter. Apply `project_schedule` before returning a scenario; deterministically resample rejected Sobol points.

- [ ] **Step 5: Verify exact repeatability**

Run: `python -m pytest tests/test_dataset.py tests/test_doe.py -q`

Expected: identical schedule hashes for two seed-42 runs and no split overlap.

- [ ] **Step 6: Commit**

Commit: `git commit -m "feat: generate auditable OPM scenarios"`

## Task 7: Shared Surrogate Protocol and Temporal Baselines

**Files:**
- Create: `src/aios_track2/surrogates/base.py`
- Create: `src/aios_track2/surrogates/linear.py`
- Create: `src/aios_track2/surrogates/tcn.py`
- Create: `configs/models/linear.yaml`
- Create: `configs/models/tcn.yaml`
- Test: `tests/test_surrogates.py`

**Interfaces:**
- Produces protocol: `fit(train: ScenarioBatch, validation: ScenarioBatch) -> TrainingReport`
- Produces protocol: `predict(batch: ScenarioBatch) -> Prediction(mean, variance)`
- Produces: `evaluate_surrogate(model, test) -> SurrogateMetrics`

- [ ] **Step 1: Write protocol shape and uncertainty tests**

```python
import numpy as np

from aios_track2.surrogates.linear import LinearSurrogate


def test_prediction_contains_finite_mean_and_variance(scenario_batch) -> None:
    model = LinearSurrogate(seed=42).fit(scenario_batch, scenario_batch)
    prediction = model.predict(scenario_batch)
    assert prediction.mean.shape == scenario_batch.targets.shape
    assert prediction.variance.shape == scenario_batch.targets.shape
    assert np.isfinite(prediction.mean).all()
    assert (prediction.variance >= 0).all()
```

- [ ] **Step 2: Confirm tests fail**

Run: `python -m pytest tests/test_surrogates.py -q`

Expected: FAIL because the surrogate package does not exist.

- [ ] **Step 3: Implement persistence and ridge references**

Fit per-target standardized ridge regressions with bootstrap variance. Serialize feature names, scalers, seed, dataset revision and metrics next to the weights.

- [ ] **Step 4: Implement the causal TCN**

Use dilations `(1, 2, 4, 8)`, kernel size 3, 128 hidden channels, residual blocks, dropout 0.1 and Huber loss. Predict deltas for `WOPR`, `WLPR`, `WWIR`, `BHP`, `THP`, and `WCT`; reconstruct levels only in the evaluator.

- [ ] **Step 5: Verify metric serialization and leakage guards**

Run: `python -m pytest tests/test_surrogates.py -q`

Expected: protocol tests pass; evaluator rejects batches containing a scenario ID from training.

- [ ] **Step 6: Commit**

Commit: `git commit -m "feat: add temporal surrogate baselines"`

## Task 8: Graph-Temporal Ensemble and Uncertainty Gate

**Files:**
- Create: `src/aios_track2/surrogates/graph.py`
- Create: `configs/models/graph.yaml`
- Test: `tests/test_graph_surrogate.py`

**Interfaces:**
- Produces: `GraphTemporalSurrogate(SurrogateProtocol)`
- Produces: `DeepEnsemble.predict(batch) -> Prediction`
- Produces: `is_ood(prediction: Prediction, threshold: float) -> ndarray[bool]`

- [ ] **Step 1: Write permutation, variance and OOD tests**

```python
def test_well_permutation_preserves_field_total(graph_batch, trained_graph_model) -> None:
    original = trained_graph_model.predict(graph_batch).mean.sum(axis=2)
    permuted = trained_graph_model.predict(graph_batch.permute_wells(seed=7)).mean.sum(axis=2)
    assert_allclose(original, permuted, rtol=1e-5, atol=1e-6)


def test_ensemble_variance_increases_outside_training_domain(ensemble, in_domain, out_of_domain) -> None:
    assert ensemble.predict(out_of_domain).variance.mean() > ensemble.predict(in_domain).variance.mean()
```

- [ ] **Step 2: Confirm tests fail**

Run: `python -m pytest tests/test_graph_surrogate.py -q`

Expected: FAIL because `GraphTemporalSurrogate` is undefined.

- [ ] **Step 3: Implement temporal encoding and graph message passing**

Encode each well with a shared causal TCN, apply two GraphSAGE layers with 128 hidden units and edge weights from distance/completion/lag features, then decode per-well deltas and field aggregates. Mask unavailable targets and add non-negativity plus field-balance penalties to the loss.

- [ ] **Step 4: Implement a five-seed ensemble**

Train seeds `[11, 23, 42, 71, 101]`. The prediction mean is the member mean; epistemic variance is member variance. Store each checkpoint independently and an ensemble manifest containing every SHA-256.

- [ ] **Step 5: Verify the graph model**

Run: `python -m pytest tests/test_graph_surrogate.py -q`

Expected: graph invariance, variance and OOD tests pass.

- [ ] **Step 6: Commit**

Commit: `git commit -m "feat: add uncertainty-aware graph surrogate"`

## Task 9: CEM/CMA-ES Search and Active Learning

**Files:**
- Create: `src/aios_track2/optimization.py`
- Create: `configs/optimizers/cem.yaml`
- Create: `configs/optimizers/cma.yaml`
- Test: `tests/test_optimization.py`

**Interfaces:**
- Produces: `optimize(request: OptimizationRequest) -> OptimizationResult`
- Produces: `promote_candidates(result, budget: int) -> tuple[Candidate, ...]`
- Produces: `active_learning_batch(candidates, exploit: int, explore: int) -> tuple[Candidate, ...]`

- [ ] **Step 1: Write constraint and risk-score tests**

```python
def test_ood_candidate_cannot_win_without_promotion(toy_optimizer) -> None:
    result = toy_optimizer.run()
    assert result.best.accepted is True
    assert result.best.opm_validated is True


def test_risk_adjusted_score_penalizes_uncertainty() -> None:
    assert risk_score(mean_npv=100.0, std_npv=20.0, penalty=1.0) == 80.0
```

- [ ] **Step 2: Confirm tests fail**

Run: `python -m pytest tests/test_optimization.py -q`

Expected: FAIL because the optimization module does not exist.

- [ ] **Step 3: Implement CEM and CMA-ES over the same latent controls**

Encode cluster-level quarterly spline coefficients, status logits and irreversible conversion times. Every decoded candidate passes `project_schedule`. Score accepted candidates as `mean_npv - lambda * std_npv`; assign negative infinity to rejected candidates.

- [ ] **Step 4: Implement OPM promotion and active-learning selection**

Promote the top eight risk-adjusted candidates plus four high-uncertainty diverse candidates per round. Deduplicate by schedule hash. After OPM, append results to the dataset and rerank by deterministic NPV.

- [ ] **Step 5: Verify on a deterministic quadratic surrogate fixture**

Run: `python -m pytest tests/test_optimization.py -q`

Expected: both optimizers converge inside the allowed interval; no unvalidated candidate is reported as winner.

- [ ] **Step 6: Commit**

Commit: `git commit -m "feat: optimize schedules with OPM promotion"`

## Task 10: MARL Challenger with Shared Safety Boundary

**Files:**
- Create: `src/aios_track2/marl.py`
- Create: `configs/optimizers/mappo.yaml`
- Test: `tests/test_marl.py`

**Interfaces:**
- Produces Gymnasium environment: `ReservoirEnv`
- Produces: `evaluate_policy(policy, seeds: Sequence[int]) -> PolicyReport`
- Consumes the same `project_schedule`, surrogate protocol and OPM promotion interface as Task 9.

- [ ] **Step 1: Write reward and safety tests**

```python
def test_reward_uses_npv_delta_and_constraint_cost(env) -> None:
    _, reward, _, _, info = env.step(env.safe_action())
    assert reward == info["npv_delta"] - info["constraint_cost"] - info["uncertainty_cost"]


def test_invalid_action_is_projected_before_surrogate_call(env, spy_surrogate) -> None:
    env.step(env.action_with_wlpr(700.0))
    assert spy_surrogate.last_batch.controls.max_wlpr <= 500.0
```

- [ ] **Step 2: Confirm tests fail**

Run: `python -m pytest tests/test_marl.py -q`

Expected: FAIL because `ReservoirEnv` does not exist.

- [ ] **Step 3: Implement CTDE by graph cluster**

Each cluster is one policy actor with local observations; the centralized critic sees field aggregates, every cluster state and proposed action. Reward equals field NPV delta minus constraint and uncertainty costs. Do not use one actor per well.

- [ ] **Step 4: Define the promotion rule**

Evaluate seeds `[11, 23, 42, 71, 101]`. MARL becomes a finalist only when median OPM-NPV exceeds the best CEM/CMA result, the lower bootstrap confidence bound is positive, and all OPM runs have zero hard violations.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/test_marl.py -q`

Expected: environment tests pass and the unsafe fixture cannot reach the surrogate unprojected.

Commit: `git commit -m "feat: add CTDE MARL challenger"`

## Task 11: Role-Based Agent Orchestration and Audit Trail

**Files:**
- Create: `src/aios_track2/agents.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Produces: `run_pipeline(request: PipelineRequest) -> PipelineResult`
- Produces append-only events: `AgentEvent(actor, action, input_hashes, output_hashes, timestamp)`

- [ ] **Step 1: Write authority-boundary tests**

```python
def test_explanation_agent_cannot_modify_schedule(orchestrator) -> None:
    result = orchestrator.run_fixture()
    assert result.schedule_sha256 == result.audit.schedule_before_explanation_sha256


def test_economics_is_only_npv_writer(orchestrator) -> None:
    result = orchestrator.run_fixture()
    assert result.audit.writers_for("npv_mrub") == {"EconomicsAgent"}
```

- [ ] **Step 2: Confirm tests fail**

Run: `python -m pytest tests/test_agents.py -q`

Expected: FAIL because orchestration does not exist.

- [ ] **Step 3: Implement the deterministic state machine**

Use states `INGESTED`, `DIAGNOSED`, `PLANNED`, `PROJECTED`, `PREDICTED`, `PROMOTED`, `SIMULATED`, `VALUED`, `EXPLAINED`, `PACKAGED`. Reject illegal transitions and hash every input/output artifact at transition time.

- [ ] **Step 4: Add the read-only explanation adapter**

Pass only the audit record, aggregate metrics and already computed schedule summary to the LLM. Parse its response as prose; never expose mutating tool handles inside this role.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/test_agents.py -q`

Expected: authority and transition tests pass.

Commit: `git commit -m "feat: orchestrate auditable AIOS agents"`

## Task 12: API, Dashboard and Final Package

**Files:**
- Create: `src/aios_track2/api.py`
- Create: `ui/package.json`
- Create: `ui/src/App.tsx`
- Create: `tests/test_api.py`
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Modify: `aios-track2/README.md`

**Interfaces:**
- `POST /api/runs` returns run ID and accepted config hash.
- `GET /api/runs/{run_id}` returns current state, metrics and audit events.
- `GET /api/runs/{run_id}/schedule` downloads the validated INC file.

- [ ] **Step 1: Write API tests**

```python
def test_schedule_endpoint_refuses_unvalidated_run(client) -> None:
    response = client.get("/api/runs/fixture-unvalidated/schedule")
    assert response.status_code == 409


def test_validated_schedule_is_downloadable(client, validated_run) -> None:
    response = client.get(f"/api/runs/{validated_run.id}/schedule")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
```

- [ ] **Step 2: Confirm tests fail**

Run: `python -m pytest tests/test_api.py -q`

Expected: FAIL because the API does not exist.

- [ ] **Step 3: Implement FastAPI endpoints and event streaming**

Return immutable run summaries. Stream append-only `AgentEvent` objects using server-sent events. Allow a schedule download only from state `PACKAGED` with an OPM validation hash.

- [ ] **Step 4: Implement the React dashboard**

Display field KPI cards, a well graph, time-series comparison of surrogate and OPM, uncertainty bands, constraint events, NPV breakdown and chronological agent trace. Disable the download button until the API returns `PACKAGED`.

- [ ] **Step 5: Add container health checks and a smoke test**

Run: `docker compose build`

Expected: API and UI images build successfully.

Run: `docker compose up -d && curl --fail http://127.0.0.1:8000/health`

Expected JSON: `{"status":"ok"}`.

- [ ] **Step 6: Commit**

Commit: `git commit -m "feat: add AIOS dashboard and reproducible package"`

## Task 13: GitHub Actions and HF Experiment Publication

**Files:**
- Create: `.github/workflows/aios-ci.yml`
- Create: `.github/workflows/aios-generate-doe.yml`
- Create: `.github/workflows/aios-train-surrogate.yml`
- Create: `.github/workflows/aios-optimize.yml`
- Create: `.github/workflows/aios-validate-candidate.yml`
- Create: `.github/workflows/aios-publish-results.yml`
- Test: `tests/test_workflows.py`

**Interfaces:**
- Every workflow accepts `config_path`, `seed` and `dataset_revision` where applicable.
- Every producing workflow uploads `manifest.json` to `MakSoS1/aios-track2-runs/runs/<git_sha>-<github_run_id>/`.

- [ ] **Step 1: Write static workflow policy tests**

```python
def test_training_is_never_triggered_by_pull_request(workflow_yaml) -> None:
    training = workflow_yaml(".github/workflows/aios-train-surrogate.yml")
    assert set(training["on"]) == {"workflow_dispatch"}


def test_hf_token_is_referenced_as_secret(workflow_text) -> None:
    text = workflow_text(".github/workflows/aios-publish-results.yml")
    assert "secrets.HF_TOKEN" in text
    assert "hf_" not in text
```

- [ ] **Step 2: Confirm tests fail**

Run: `python -m pytest tests/test_workflows.py -q`

Expected: FAIL because workflows do not exist.

- [ ] **Step 3: Implement CI and manual production workflows**

Pin third-party actions by full commit SHA. CI runs Ruff, mypy and pytest. DoE uses a bounded matrix with `fail-fast: false`. Training and optimization use only `workflow_dispatch`; Lightning credentials and `HF_TOKEN` are secret environment variables.

- [ ] **Step 4: Implement immutable HF publication**

Before upload, reject an existing run path. Upload configuration, source commit, dependency lock, metrics, checkpoint hashes, schedule, OPM outputs and NPV breakdown. Re-download the manifest and run `hf cache verify` before marking the Action successful.

- [ ] **Step 5: Verify workflow syntax and policy**

Run: `python -m pytest tests/test_workflows.py -q`

Expected: all workflow policy tests pass.

Run: `actionlint .github/workflows/aios-*.yml`

Expected: no diagnostics.

- [ ] **Step 6: Commit**

Commit: `git commit -m "ci: orchestrate AIOS experiments through Actions"`

## Task 14: End-to-End Reproducibility Gate

**Files:**
- Create: `scripts/run_smoke.sh`
- Create: `tests/test_end_to_end.py`
- Create: `docs/reproducibility.md`
- Modify: `aios-track2/README.md`

**Interfaces:**
- Produces final directory containing `wells_schedule.inc`, `npv.json`, `audit.jsonl`, `manifest.json`, `run_config.yaml` and hashes of all OPM outputs.

- [ ] **Step 1: Write the end-to-end fixture assertion**

```python
def test_smoke_pipeline_is_reproducible(run_smoke_twice) -> None:
    first, second = run_smoke_twice(seed=42)
    assert first.schedule_sha256 == second.schedule_sha256
    assert first.npv_mrub == second.npv_mrub
    assert first.manifest_without_timestamps == second.manifest_without_timestamps
```

- [ ] **Step 2: Confirm the test fails before the smoke script exists**

Run: `python -m pytest tests/test_end_to_end.py -q`

Expected: FAIL because `scripts/run_smoke.sh` is missing.

- [ ] **Step 3: Implement the smoke command**

The script runs the minimal deck through ingest, projection, fake/real OPM selection, economics, explanation and packaging. It exits non-zero on a missing hash, unvalidated schedule, unexpected dataset revision or NPV mismatch.

- [ ] **Step 4: Run the complete verification suite**

Run: `ruff check src tests && mypy src && python -m pytest -q`

Expected: zero Ruff errors, zero mypy errors and zero pytest failures.

Run: `bash scripts/run_smoke.sh --seed 42`

Expected: two identical schedule/NPV hashes are reported and the package manifest verifies.

- [ ] **Step 5: Build the final container**

Run: `docker compose build --pull`

Expected: all images build from a clean cache without undeclared local files.

- [ ] **Step 6: Commit**

Commit: `git commit -m "test: enforce end-to-end reproducibility"`

## Plan Self-Review

- Spec coverage: storage, data generation, OPM, economics, constraints, surrogate alternatives, optimizer alternatives, MARL challenger, role agents, UI, Actions, HF publication and reproducibility each map to a concrete task.
- Type consistency: `Schedule`, `ProjectionResult`, `FlowRequest`, `FlowResult`, `ScenarioManifest`, `Prediction`, `OptimizationResult` and `AgentEvent` are defined before downstream use.
- Safety boundary: every route to surrogate, OPM and final packaging passes through `project_schedule`; only `EconomicsAgent` writes NPV.
- Publication boundary: no step reports a surrogate-only candidate as final, and all experiment paths are immutable.
