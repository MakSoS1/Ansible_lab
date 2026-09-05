from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from aios_track2.challenge_doe import (
    CHALLENGE_NODE_DATES,
    FROZEN_CHALLENGE_SHA256,
    challenge_design_sha256,
    challenge_scenario_by_id,
    deterministic_spatial_groups,
    schedule_role_names,
)
from aios_track2.challenge_schedule import scale_schedule_with_role_policies
from aios_track2.deck import parse_deck
from aios_track2.ecl_summary import load_summary_case
from aios_track2.model_z_economics import load_model_z_density_map, scenario_chdd
from aios_track2.opm import FlowRequest, run_flow
from aios_track2.reference_parity import load_model_z_reference, reference_parity_report
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
    parser.add_argument("--reference-workbook", type=Path)
    args = parser.parse_args()

    if challenge_design_sha256() != FROZEN_CHALLENGE_SHA256:
        raise RuntimeError("frozen challenge design SHA does not match its preregistered constant")
    scenario = challenge_scenario_by_id(args.scenario_id)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "design_sha256": FROZEN_CHALLENGE_SHA256,
        "scenario": scenario.as_dict(),
    }

    with tempfile.TemporaryDirectory(prefix=f"model-z-challenge-{scenario.scenario_id:02d}-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(args.archive) as zf:
            zf.extractall(root)
        deck, meta = find_root_deck(root)
        metadata = parse_deck(deck)
        wells = tuple(well.name for well in metadata.wells)
        dimensions = tuple(meta.get("dimensions") or ())
        if dimensions != (91, 102, 59):
            raise ValueError(f"unexpected Model Z dimensions: {dimensions}")
        if len(wells) != 103:
            raise ValueError(f"expected 103 Model Z wells, got {len(wells)}")

        install_report = install_training_summary(root)
        schedule = _find_schedule(root)
        original = schedule.read_text(encoding="utf-8", errors="strict")
        original_sha = _sha256_bytes(original.encode())
        producer_names, injector_names = schedule_role_names(original)
        metadata_by_name = {well.name: well for well in metadata.wells}
        unknown_roles = sorted((producer_names | injector_names) - set(metadata_by_name))
        if unknown_roles:
            raise ValueError(f"schedule uses wells absent from WELSPECS: {unknown_roles}")
        producer_groups = deterministic_spatial_groups(
            [metadata_by_name[name] for name in sorted(producer_names)],
            4,
        )
        injector_groups = deterministic_spatial_groups(
            [metadata_by_name[name] for name in sorted(injector_names)],
            2,
        )
        role_switching = sorted(producer_names & injector_names)

        if scenario.scenario_id == 0:
            modified = original
        else:
            modified = scale_schedule_with_role_policies(
                original,
                producer_well_groups=producer_groups,
                injector_well_groups=injector_groups,
                producer_group_nodes=scenario.producer_nodes(),
                injector_group_nodes=scenario.injector_nodes(),
                node_dates=CHALLENGE_NODE_DATES,
                effective_from=CHALLENGE_NODE_DATES[0],
                max_wlpr=500.0,
            )
            schedule.write_text(modified, encoding="utf-8", newline="\n")
        modified_sha = _sha256_bytes(modified.encode())
        (args.output / "wells_schedule.inc").write_text(modified, encoding="utf-8", newline="\n")

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
                "producer_role_wells": len(producer_names),
                "injector_role_wells": len(injector_names),
                "role_switching_wells": role_switching,
                "producer_groups": producer_groups,
                "injector_groups": injector_groups,
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

            density = load_model_z_density_map(root)
            calculation = scenario_chdd(
                args.output / "summary.npz",
                oil_density_t_m3=density.oil_t_m3,
                water_density_t_m3=density.water_t_m3,
                report_mode="all",
            )
            (args.output / "economics.json").write_text(
                json.dumps(calculation, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifest["npv_mrub"] = float(calculation["summary"]["totalChddM"])
            manifest["mixed_pvt_wells"] = list(density.mixed_wells)

            if scenario.scenario_id == 0 and args.reference_workbook is not None:
                reference = load_model_z_reference(args.reference_workbook)
                parity = reference_parity_report(calculation, reference)
                (args.output / "reference-parity.json").write_text(
                    json.dumps(parity, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                manifest["reference_parity"] = parity
                if not parity["passed"]:
                    raise ValueError(f"organizer reference parity failed: {parity['failures']}")

    (args.output / "scenario-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if manifest["status"] != "success":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
