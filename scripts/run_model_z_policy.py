from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from aios_track2.deck import parse_deck
from aios_track2.eclipse_schedule import scale_schedule_with_policy
from aios_track2.ecl_summary import load_summary_case
from aios_track2.opm import FlowRequest, run_flow
from aios_track2.real_doe import FROZEN_DESIGN_SHA256, scenario_by_id
from aios_track2.summary_install import install_training_summary
from inspect_model_z import find_root_deck

FIELD_VECTORS = ("FOPT", "FWPT", "FWIT", "FOPR", "FWPR", "FWIR", "FLPR", "FPR")
WELL_VECTORS = ("WOPT", "WWPT", "WWIT", "WOPR", "WWPR", "WWIR", "WLPR", "WBHP", "WTHP", "WWCT")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _find_schedule(root: Path) -> Path:
    exact = sorted(root.rglob("Model_Z_sch.inc"))
    if len(exact) == 1:
        return exact[0]
    candidates = sorted(root.rglob("*_sch.inc"))
    if len(candidates) != 1:
        raise ValueError(f"expected one Model Z schedule include, found {len(candidates)}")
    return candidates[0]


def _export_compact_summary(output_dir: Path, wells: tuple[str, ...]) -> dict[str, object]:
    summary = load_summary_case(output_dir)
    payload: dict[str, np.ndarray] = {
        "dates": np.asarray([value.isoformat() for value in summary.dates], dtype="U10"),
        "wells": np.asarray(wells, dtype="U32"),
    }
    for keyword in FIELD_VECTORS:
        payload[f"field_{keyword}"] = summary.vector(keyword)
    for keyword in WELL_VECTORS:
        payload[f"well_{keyword}"] = np.stack([summary.vector(keyword, well) for well in wells], axis=1)
    np.savez_compressed(output_dir / "summary.npz", **payload)
    max_wlpr = float(np.max(payload["well_WLPR"]))
    return {
        "report_steps": len(summary.dates),
        "start_date": summary.dates[0].isoformat(),
        "end_date": summary.dates[-1].isoformat(),
        "max_wlpr": max_wlpr,
        "field_vectors": list(FIELD_VECTORS),
        "well_vectors": list(WELL_VECTORS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--scenario-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=2700)
    args = parser.parse_args()

    scenario = scenario_by_id(args.scenario_id)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "design_sha256": FROZEN_DESIGN_SHA256,
        "scenario": scenario.as_dict(),
    }

    with tempfile.TemporaryDirectory(prefix=f"model-z-doe-{scenario.scenario_id:02d}-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(args.archive) as zf:
            zf.extractall(root)
        deck, meta = find_root_deck(root)
        metadata = parse_deck(deck)
        wells = tuple(well.name for well in metadata.wells)
        if len(wells) != 103:
            raise ValueError(f"expected 103 Model Z wells, got {len(wells)}")
        install_report = install_training_summary(root)
        schedule = _find_schedule(root)
        original = schedule.read_text(encoding="utf-8", errors="strict")
        original_sha = _sha256_bytes(original.encode())

        if scenario.scenario_id == 0:
            modified = original
        else:
            modified = scale_schedule_with_policy(
                original,
                well_groups={well: 0 for well in wells},
                producer_group_nodes={0: (scenario.producer_2007, scenario.producer_2025)},
                injector_group_nodes={0: (scenario.injector_2007, scenario.injector_2025)},
                node_dates=(__import__("datetime").date(2007, 1, 1), __import__("datetime").date(2025, 1, 1)),
                effective_from=__import__("datetime").date(2007, 1, 1),
                max_wlpr=500.0,
            )
            schedule.write_text(modified, encoding="utf-8", newline="\n")

        modified_sha = _sha256_bytes(modified.encode())
        result = run_flow(
            FlowRequest(
                deck=deck,
                output_dir=args.output,
                extra_args=("--parsing-strictness=low",),
                timeout_seconds=args.timeout,
            )
        )
        manifest.update(
            {
                "status": result.status,
                "returncode": result.returncode,
                "runtime_seconds": result.runtime_seconds,
                "dimensions": meta["dimensions"],
                "well_count": len(wells),
                "summary_install_changed_files": list(install_report.changed_files),
                "schedule_sha256_before": original_sha,
                "schedule_sha256_after": modified_sha,
                "baseline_schedule_byte_identical": scenario.scenario_id == 0 and original_sha == modified_sha,
            }
        )
        if result.status == "success":
            compact = _export_compact_summary(args.output, wells)
            manifest["compact_summary"] = compact
            if compact["report_steps"] < 380:
                raise ValueError(f"unexpectedly short OPM trajectory: {compact['report_steps']}")
            if compact["max_wlpr"] > 500.0001:
                raise ValueError(f"actual OPM WLPR exceeds contract limit: {compact['max_wlpr']}")

    (args.output / "scenario-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if manifest["status"] != "success":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
