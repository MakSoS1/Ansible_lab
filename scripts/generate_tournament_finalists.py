from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

from aios_track2.challenge_doe import CHALLENGE_DIMENSIONS, FROZEN_CHALLENGE_SHA256, frozen_challenge_doe
from aios_track2.challenge_evaluation import real_opm_tournament_gate
from aios_track2.optimization import diagonal_cma_es, risk_aware_cem
from aios_track2.strategy_runner import run_blackbox_strategy
from evaluate_challenge_doe import _model_specs, _predict_ensemble, _unit_to_policy


def _load_npvs(runs_dir: Path) -> np.ndarray:
    values = np.zeros(64, dtype=float)
    for scenario_id in range(64):
        economics_path = runs_dir / str(scenario_id) / "economics.json"
        economics = json.loads(economics_path.read_text(encoding="utf-8"))
        values[scenario_id] = float(economics["summary"]["totalChddM"])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if evaluation.get("design_sha256") != FROZEN_CHALLENGE_SHA256:
        raise ValueError("evaluation design SHA does not match frozen challenge design")

    authorization = real_opm_tournament_gate(evaluation)
    (args.output_dir / "tournament-authorization.json").write_text(
        json.dumps(authorization, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not authorization["passed"]:
        print(json.dumps(authorization, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    design = frozen_challenge_doe()
    x = np.stack([row.flat_vector() for row in design], axis=0)
    npv = _load_npvs(args.runs_dir)
    npv_delta = npv - npv[0]

    specs = _model_specs()
    spec_by_name = {spec.name: spec for spec in specs}
    validation_rows = evaluation["npv_selection"]["validation"]
    top_npv_names = [row["name"] for row in validation_rows[:5]]
    unknown = [name for name in top_npv_names if name not in spec_by_name]
    if unknown:
        raise ValueError(f"evaluation references unknown NPV models: {unknown}")

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
        "surrogate_holdout_passed": bool(evaluation.get("passed")),
        "surrogate_holdout_failures": list(evaluation.get("failures", [])),
        "tournament_authorization": authorization,
        "finalists": rows,
    }
    (args.output_dir / "finalists.json").write_text(
        json.dumps(finalists, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(finalists, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
