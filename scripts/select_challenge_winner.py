from __future__ import annotations

import argparse
import json
from pathlib import Path

from aios_track2.final_selection import VerifiedCandidate, choose_verified_winner


def _load_result(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalists", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--robustness-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    finalist_payload = json.loads(args.finalists.read_text(encoding="utf-8"))
    finalist_names = [row["name"] for row in finalist_payload["finalists"]]
    rows: list[VerifiedCandidate] = []
    raw_results: dict[str, dict] = {}
    robustness: dict[str, list[dict]] = {}
    for name in finalist_names:
        result = _load_result(args.results_dir / name / "candidate-result.json")
        raw_results[name] = result
        perturbations: list[dict] = []
        for seed in (1, 2):
            perturb_path = args.robustness_dir / f"{name}__perturb_{seed}" / "candidate-result.json"
            perturbations.append(_load_result(perturb_path))
        robustness[name] = perturbations
        rows.append(
            VerifiedCandidate(
                name=name,
                vector=tuple(float(value) for value in result["vector"]),
                opm_npv_mrub=float(result["npv_mrub"]),
                max_wlpr=float(result["compact_summary"]["max_wlpr"]),
                status=str(result["status"]),
                schedule_sha256=str(result["schedule_sha256"]),
                robustness_npvs_mrub=tuple(float(item["npv_mrub"]) for item in perturbations),
                opm_calls=1 + len(perturbations),
            )
        )

    winner = choose_verified_winner(rows)
    ranked = sorted(
        rows,
        key=lambda row: (-row.opm_npv_mrub, -row.robustness_floor_mrub, -row.robustness_mean_mrub, row.name),
    )
    payload = {
        "winner": winner.as_dict(),
        "ranking": [row.as_dict() for row in ranked],
        "candidate_results": raw_results,
        "robustness_results": robustness,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
