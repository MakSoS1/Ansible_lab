from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import qmc

from aios_track2.challenge_doe import (
    CHALLENGE_DIMENSIONS,
    CHALLENGE_GROUPS,
    CHALLENGE_NODE_DATES,
    FROZEN_CHALLENGE_SHA256,
    frozen_challenge_doe,
)
from aios_track2.challenge_evaluation import physical_inventory_gate
from aios_track2.optimization import diagonal_cma_es, risk_aware_cem
from aios_track2.real_validation import align_common_post_start, dynamic_delta_report, ranking_report
from aios_track2.small_data import AdditiveGroupKernelRidge, QuadraticRidge, StationaryKernelRidge, project_temporal_policy
from aios_track2.strategy_runner import run_blackbox_strategy

CHANNELS = ("FOPT", "FWPT", "FWIT", "FOPR", "FWPR", "FWIR", "FLPR", "FPR")
START_DATE = "2007-01-01"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    factory: Callable[[], Any]


def _model_specs() -> list[ModelSpec]:
    specs = [
        ModelSpec(f"quadratic-ridge-{ridge:g}", lambda ridge=ridge: QuadraticRidge(ridge=ridge))
        for ridge in (1e-6, 1e-4, 1e-2)
    ]
    for kind in ("rbf", "matern52"):
        for length in (1.0, 2.0, 4.0, 8.0):
            for ridge in (1e-5, 1e-3):
                specs.append(
                    ModelSpec(
                        f"{kind}-ls{length:g}-r{ridge:g}",
                        lambda kind=kind, length=length, ridge=ridge: StationaryKernelRidge(
                            kind=kind,
                            length_scale=length,
                            ridge=ridge,
                        ),
                    )
                )
    for length in (0.75, 1.5, 3.0, 6.0):
        for global_weight in (0.0, 0.15, 0.35):
            specs.append(
                ModelSpec(
                    f"additive-m52-ls{length:g}-g{global_weight:g}",
                    lambda length=length, global_weight=global_weight: AdditiveGroupKernelRidge(
                        group_size=len(CHALLENGE_NODE_DATES),
                        length_scale=length,
                        ridge=1e-4,
                        global_weight=global_weight,
                    ),
                )
            )
    return specs


def _load_run(run_dir: Path) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    manifest = json.loads((run_dir / "scenario-manifest.json").read_text(encoding="utf-8"))
    economics = json.loads((run_dir / "economics.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "summary.npz") as summary:
        dates = summary["dates"].astype(str)
    return manifest, dates, economics


def _field_tensor(runs_dir: Path, common_dates: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    for scenario_id in range(64):
        with np.load(runs_dir / str(scenario_id) / "summary.npz") as summary:
            dates = summary["dates"].astype(str)
            index_by_date = {value: index for index, value in enumerate(dates)}
            missing = [value for value in common_dates if value not in index_by_date]
            if missing:
                raise ValueError(f"scenario {scenario_id} is missing common dates: {missing[:5]}")
            indices = np.asarray([index_by_date[value] for value in common_dates], dtype=int)
            rows.append(np.stack([summary[f"field_{channel}"][indices] for channel in CHANNELS], axis=-1))
    return np.stack(rows, axis=0)


def _split_ids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = frozen_challenge_doe()
    train = np.asarray([row.scenario_id for row in rows if row.split == "train"], dtype=int)
    validation = np.asarray([row.scenario_id for row in rows if row.split == "validation"], dtype=int)
    holdout = np.asarray([row.scenario_id for row in rows if row.split == "holdout"], dtype=int)
    if (len(train), len(validation), len(holdout)) != (40, 8, 16):
        raise RuntimeError("unexpected frozen challenge split")
    return train, validation, holdout


def _dynamic_selection_key(report: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(report["min_aggregate_channel_r2"]),
        float(report["mean_aggregate_channel_r2"]),
        -float(report["max_aggregate_channel_nrmse"]),
        float(report["p10_scenario_channel_r2"]),
    )


def _ranking_selection_key(report: dict[str, float]) -> tuple[float, ...]:
    spearman = report["spearman"] if np.isfinite(report["spearman"]) else -1.0
    return (
        float(spearman),
        float(report["pairwise_accuracy"]),
        float(report["top_k_recall"]),
        -float(report["mae"]),
        -float(report["max_abs_error"]),
    )


def _predict_ensemble(models: list[Any], x: np.ndarray, baseline_npv: float) -> np.ndarray:
    predictions = [np.asarray(model.predict(x), dtype=float).reshape(-1) + baseline_npv for model in models]
    return np.stack(predictions, axis=0)


def _unit_to_policy(unit: np.ndarray) -> np.ndarray:
    raw = 0.8 + 0.4 * np.asarray(unit, dtype=float)
    return project_temporal_policy(
        raw,
        groups=CHALLENGE_GROUPS,
        nodes=len(CHALLENGE_NODE_DATES),
        lower=0.8,
        upper=1.2,
        max_delta=0.12,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    design = frozen_challenge_doe()
    x = np.stack([row.flat_vector() for row in design], axis=0)
    manifests: dict[int, dict[str, Any]] = {}
    date_arrays: list[np.ndarray] = []
    npv = np.zeros(64, dtype=float)
    for scenario_id in range(64):
        manifest, dates, economics = _load_run(args.runs_dir / str(scenario_id))
        if manifest.get("design_sha256") != FROZEN_CHALLENGE_SHA256:
            raise ValueError(f"scenario {scenario_id} has wrong challenge design SHA")
        manifests[scenario_id] = manifest
        date_arrays.append(dates)
        npv[scenario_id] = float(economics["summary"]["totalChddM"])

    physical = physical_inventory_gate(manifests, expected_ids=set(range(64)))
    if not physical["passed"]:
        raise ValueError(f"physical inventory gate failed: {physical}")
    reference_parity = manifests[0].get("reference_parity")
    if not isinstance(reference_parity, dict) or not reference_parity.get("passed"):
        raise ValueError("baseline organizer-reference parity was not proven")

    common_dates = align_common_post_start(date_arrays, start_date=START_DATE)
    if len(common_dates) < 200:
        raise ValueError(f"too few common post-2007 report dates: {len(common_dates)}")
    y = _field_tensor(args.runs_dir, common_dates)
    baseline = y[0]
    dynamic_delta = y - baseline[None, :, :]
    npv_delta = npv - npv[0]
    train, validation, holdout = _split_ids()
    specs = _model_specs()

    dynamic_candidates: list[dict[str, Any]] = []
    npv_candidates: list[dict[str, Any]] = []
    for spec in specs:
        dynamic_model = spec.factory().fit(x[train], dynamic_delta[train])
        val_prediction = dynamic_model.predict(x[validation]) + baseline[None, :, :]
        full_prediction = y.copy()
        full_prediction[validation] = val_prediction
        dynamic_report = dynamic_delta_report(
            y,
            full_prediction,
            baseline=baseline,
            scenario_ids=validation,
            channels=CHANNELS,
        )
        dynamic_candidates.append({"name": spec.name, "report": dynamic_report})

        npv_model = spec.factory().fit(x[train], npv_delta[train])
        val_npv_prediction = npv_model.predict(x[validation]).reshape(-1) + npv[0]
        npv_report = ranking_report(npv[validation], val_npv_prediction, top_k=3)
        npv_candidates.append({"name": spec.name, "report": npv_report})

    dynamic_candidates.sort(key=lambda row: _dynamic_selection_key(row["report"]), reverse=True)
    npv_candidates.sort(key=lambda row: _ranking_selection_key(row["report"]), reverse=True)
    dynamic_winner_name = dynamic_candidates[0]["name"]
    npv_winner_name = npv_candidates[0]["name"]
    spec_by_name = {spec.name: spec for spec in specs}

    fit_ids = np.concatenate([train, validation])
    dynamic_winner = spec_by_name[dynamic_winner_name].factory().fit(x[fit_ids], dynamic_delta[fit_ids])
    holdout_dynamic = dynamic_winner.predict(x[holdout]) + baseline[None, :, :]
    full_dynamic = y.copy()
    full_dynamic[holdout] = holdout_dynamic
    holdout_dynamic_report = dynamic_delta_report(
        y,
        full_dynamic,
        baseline=baseline,
        scenario_ids=holdout,
        channels=CHANNELS,
    )

    npv_winner = spec_by_name[npv_winner_name].factory().fit(x[fit_ids], npv_delta[fit_ids])
    holdout_npv_prediction = npv_winner.predict(x[holdout]).reshape(-1) + npv[0]
    holdout_npv_report = ranking_report(npv[holdout], holdout_npv_prediction, top_k=3)

    failures: list[str] = []
    if holdout_dynamic_report["min_aggregate_channel_r2"] < 0.95:
        failures.append("DYNAMIC_R2_LT_095")
    if holdout_dynamic_report["max_aggregate_channel_nrmse"] > 0.05:
        failures.append("DYNAMIC_NRMSE_GT_005")
    if not np.isfinite(holdout_npv_report["spearman"]) or holdout_npv_report["spearman"] < 0.95:
        failures.append("NPV_SPEARMAN_LT_095")
    if holdout_npv_report["pairwise_accuracy"] < 0.95:
        failures.append("NPV_PAIRWISE_LT_095")
    if holdout_npv_report["top_k_recall"] < 0.90:
        failures.append("NPV_TOP3_RECALL_LT_090")

    evaluation = {
        "design_sha256": FROZEN_CHALLENGE_SHA256,
        "split": {"train": train.tolist(), "validation": validation.tolist(), "holdout": holdout.tolist()},
        "common_dates": {"count": len(common_dates), "start": str(common_dates[0]), "end": str(common_dates[-1])},
        "physical_gate": physical,
        "reference_parity": reference_parity,
        "dynamic_selection": {
            "winner": dynamic_winner_name,
            "validation": dynamic_candidates,
            "holdout": holdout_dynamic_report,
        },
        "npv_selection": {
            "winner": npv_winner_name,
            "validation": npv_candidates,
            "holdout": holdout_npv_report,
        },
        "passed": not failures,
        "failures": failures,
        "baseline_npv_mrub": float(npv[0]),
        "observed_best_npv_mrub": float(np.max(npv)),
        "observed_best_scenario": int(np.argmax(npv)),
    }
    (args.output_dir / "challenge-evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if failures:
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    # Only after the untouched holdout gate is computed do we refit on all 64
    # real OPM scenarios for final candidate search.
    top_npv_names = [row["name"] for row in npv_candidates[:5]]
    ensemble = [spec_by_name[name].factory().fit(x, npv_delta) for name in top_npv_names]
    npv_range = max(float(np.ptp(npv)), 1.0)
    normalized_train = (x - 1.0) / 0.2

    def objective(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        policy = _unit_to_policy(unit)
        predictions = _predict_ensemble(ensemble, policy, float(npv[0]))
        mean = np.mean(predictions, axis=0)
        disagreement = np.std(predictions, axis=0)
        normalized = (policy - 1.0) / 0.2
        distances = np.sqrt(np.sum((normalized[:, None, :] - normalized_train[None, :, :]) ** 2, axis=-1))
        novelty = np.min(distances, axis=1) / np.sqrt(CHALLENGE_DIMENSIONS)
        uncertainty = disagreement + 0.02 * npv_range * novelty
        valid = np.all((policy >= 0.8 - 1e-12) & (policy <= 1.2 + 1e-12), axis=1)
        return mean, uncertainty, valid

    cem = risk_aware_cem(
        objective,
        dim=CHALLENGE_DIMENSIONS,
        seed=9201,
        population=512,
        iterations=16,
        risk_beta=0.35,
    )
    cma = diagonal_cma_es(
        objective,
        dim=CHALLENGE_DIMENSIONS,
        seed=9202,
        population=384,
        iterations=20,
        risk_beta=0.25,
    )
    mappo = run_blackbox_strategy("graph_mappo", objective, dim=CHALLENGE_DIMENSIONS, seed=9203, budget=4096)
    sobol_sampler = qmc.Sobol(d=CHALLENGE_DIMENSIONS, scramble=True, seed=9204)
    sobol_unit = sobol_sampler.random_base2(m=13)
    sobol_value, sobol_unc, sobol_valid = objective(sobol_unit)
    sobol_score = np.where(sobol_valid, sobol_value - 0.30 * sobol_unc, -np.inf)
    sobol_index = int(np.argmax(sobol_score))

    def finalist(name: str, unit_vector: np.ndarray, evaluations: int) -> dict[str, Any]:
        policy = _unit_to_policy(np.asarray(unit_vector, dtype=float).reshape(1, -1))[0]
        predictions = _predict_ensemble(ensemble, policy[None, :], float(npv[0]))[:, 0]
        return {
            "name": name,
            "vector": [float(value) for value in policy],
            "predicted_npv_mrub": float(np.mean(predictions)),
            "predicted_uncertainty_mrub": float(np.std(predictions)),
            "surrogate_evaluations": int(evaluations),
        }

    best_observed_id = int(np.argmax(npv))
    rows = [
        {
            "name": "baseline",
            "vector": [float(value) for value in x[0]],
            "predicted_npv_mrub": float(npv[0]),
            "predicted_uncertainty_mrub": 0.0,
            "surrogate_evaluations": 64,
            "source_scenario_id": 0,
        },
        {
            "name": "best_observed",
            "vector": [float(value) for value in x[best_observed_id]],
            "predicted_npv_mrub": float(npv[best_observed_id]),
            "predicted_uncertainty_mrub": 0.0,
            "surrogate_evaluations": 64,
            "source_scenario_id": best_observed_id,
        },
        finalist("cem", cem.x, cem.evaluations),
        finalist("cma_es", cma.x, cma.evaluations),
        finalist("mappo", mappo.x, mappo.evaluations),
        finalist("sobol", sobol_unit[sobol_index], len(sobol_unit)),
    ]
    finalists = {
        "design_sha256": FROZEN_CHALLENGE_SHA256,
        "npv_ensemble_models": top_npv_names,
        "finalists": rows,
    }
    (args.output_dir / "finalists.json").write_text(json.dumps(finalists, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"evaluation": evaluation, "finalists": finalists}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
