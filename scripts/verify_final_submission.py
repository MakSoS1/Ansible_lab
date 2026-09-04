from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from aios_track2.final_selection import VerifiedCandidate, verify_clean_rerun
from aios_track2.hfstore import RunManifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--github-run-id", required=True)
    parser.add_argument("--dataset-id", default="MakSoS1/aios-track2-runs")
    args = parser.parse_args()

    winner_payload = json.loads(args.winner.read_text(encoding="utf-8"))
    row = winner_payload["winner"]
    winner = VerifiedCandidate(
        name=row["name"],
        vector=tuple(float(value) for value in row["vector"]),
        opm_npv_mrub=float(row["opm_npv_mrub"]),
        max_wlpr=float(row["max_wlpr"]),
        status=str(row["status"]),
        schedule_sha256=str(row["schedule_sha256"]),
        robustness_npvs_mrub=tuple(float(value) for value in row.get("robustness_npvs_mrub", [])),
        opm_calls=int(row.get("opm_calls", 1)),
    )
    clean = json.loads((args.clean_dir / "candidate-result.json").read_text(encoding="utf-8"))
    verification = verify_clean_rerun(
        winner,
        clean_status=str(clean["status"]),
        clean_schedule_sha256=str(clean["schedule_sha256"]),
        clean_npv_mrub=float(clean["npv_mrub"]),
        clean_max_wlpr=float(clean["compact_summary"]["max_wlpr"]),
        npv_abs_tolerance_mrub=1e-6,
    )
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if not evaluation.get("passed"):
        verification.setdefault("failures", []).append("CHALLENGE_HOLDOUT_GATE_FAILED")
        verification["passed"] = False
    if not evaluation.get("reference_parity", {}).get("passed"):
        verification.setdefault("failures", []).append("REFERENCE_PARITY_FAILED")
        verification["passed"] = False
    if not verification["passed"]:
        raise ValueError(f"final clean-rerun verification failed: {verification['failures']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.clean_dir / "wells_schedule.inc", args.output_dir / "wells_schedule.inc")
    shutil.copy2(args.clean_dir / "economics.json", args.output_dir / "economics.json")
    shutil.copy2(args.evaluation, args.output_dir / "challenge-evaluation.json")
    shutil.copy2(args.winner, args.output_dir / "winner.json")
    schedule_sha = _sha256(args.output_dir / "wells_schedule.inc")
    if schedule_sha != clean["schedule_sha256"]:
        raise ValueError("copied final schedule SHA differs from clean OPM schedule SHA")

    manifest = {
        "status": "verified",
        "git_sha": args.git_sha,
        "github_run_id": args.github_run_id,
        "model_z_archive_sha256": _sha256(args.archive),
        "schedule_sha256": schedule_sha,
        "winner": winner.as_dict(),
        "clean_npv_mrub": float(clean["npv_mrub"]),
        "clean_npv_rub": float(clean["npv_mrub"]) * 1_000_000.0,
        "clean_max_wlpr": float(clean["compact_summary"]["max_wlpr"]),
        "verification": verification,
        "holdout": {
            "dynamic": evaluation["dynamic_selection"]["holdout"],
            "npv": evaluation["npv_selection"]["holdout"],
        },
        "reference_parity": evaluation["reference_parity"],
    }
    (args.output_dir / "final-submission-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    hf_manifest = RunManifest(
        run_id=f"final-{winner.name}",
        git_sha=args.git_sha,
        github_run_id=args.github_run_id,
        dataset_id=args.dataset_id,
        seed=9200,
        simulator_version="OPM Flow 2026.04",
        deck_sha256=manifest["model_z_archive_sha256"],
        schedule_sha256=schedule_sha,
        status="verified",
        npv_rub=manifest["clean_npv_rub"],
    )
    hf_manifest.write(args.output_dir / "hf-run-manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
