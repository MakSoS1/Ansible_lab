from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from aios_track2.challenge_doe import (
    CHALLENGE_GROUPS,
    CHALLENGE_INJECTOR_GROUPS,
    CHALLENGE_NODE_DATES,
    CHALLENGE_PRODUCER_GROUPS,
    deterministic_spatial_groups,
    schedule_role_names,
)
from aios_track2.challenge_schedule import scale_schedule_with_role_policies
from aios_track2.deck import parse_deck
from aios_track2.ecl_summary import load_summary_case
from aios_track2.model_z_economics import load_model_z_density_map, scenario_chdd
from aios_track2.opm import FlowRequest, run_flow
from aios_track2.small_data import project_temporal_policy
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
    return {
        "report_steps": len(summary.dates),
        "start_date": summary.dates[0].isoformat(),
        "end_date": summary.dates[-1].isoformat(),
        "max_wlpr": float(np.max(payload["well_WLPR"])),
    }


def _load_vector(finalists_path: Path, name: str) -> np.ndarray:
    payload = json.loads(finalists_path.read_text(encoding="utf-8"))
    matches = [row for row in payload["finalists"] if row["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one finalist named {name!r}, found {len(matches)}")
    vector = np.asarray(matches[0]["vector"], dtype=float)
    if vector.shape != (CHALLENGE_GROUPS * len(CHALLENGE_NODE_DATES),):
        raise ValueError(f"finalist {name} has wrong vector shape {vector.shape}")
    return vector


def _perturb(vector: np.ndarray, *, seed: int | None, scale: float) -> np.ndarray:
    if seed is None:
        return vector.copy()
    rng = np.random.default_rng(seed)
    raw = vector + rng.normal(0.0, scale, size=vector.shape)
    return project_temporal_policy(
        raw[None, :],
        groups=CHALLENGE_GROUPS,
        nodes=len(CHALLENGE_NODE_DATES),
        lower=0.8,
        upper=1.2,
        max_delta=0.12,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--finalists", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=2700)
    parser.add_argument("--perturb-seed", type=int)
    parser.add_argument("--perturb-scale", type=float, default=0.02)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    base_vector = _load_vector(args.finalists, args.name)
    vector = _perturb(base_vector, seed=args.perturb_seed, scale=args.perturb_scale)
    matrix = vector.reshape(CHALLENGE_GROUPS, len(CHALLENGE_NODE_DATES))
    producer_nodes = {
        group: tuple(float(value) for value in matrix[group])
        for group in range(CHALLENGE_PRODUCER_GROUPS)
    }
    injector_nodes = {
        group: tuple(float(value) for value in matrix[CHALLENGE_PRODUCER_GROUPS + group])
        for group in range(CHALLENGE_INJECTOR_GROUPS)
    }
    label = args.name if args.perturb_seed is None else f"{args.name}__perturb_{args.perturb_seed}"
    result_payload: dict[str, object] = {
        "name": label,
        "base_name": args.name,
        "vector": vector.tolist(),
        "perturb_seed": args.perturb_seed,
        "perturb_scale": args.perturb_scale if args.perturb_seed is not None else 0.0,
    }

    with tempfile.TemporaryDirectory(prefix=f"model-z-finalist-{label}-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(args.archive) as zf:
            zf.extractall(root)
        deck, meta = find_root_deck(root)
        dimensions = tuple(meta.get("dimensions") or ())
        if dimensions != (91, 102, 59):
            raise ValueError(f"unexpected Model Z dimensions: {dimensions}")
        metadata = parse_deck(deck)
        wells = tuple(well.name for well in metadata.wells)
        if len(wells) != 103:
            raise ValueError(f"expected 103 wells, got {len(wells)}")

        install_report = install_training_summary(root)
        schedule = _find_schedule(root)
        original = schedule.read_text(encoding="utf-8", errors="strict")
        producer_names, injector_names = schedule_role_names(original)
        metadata_by_name = {well.name: well for well in metadata.wells}
        unknown = sorted((producer_names | injector_names) - set(metadata_by_name))
        if unknown:
            raise ValueError(f"schedule uses wells absent from WELSPECS: {unknown}")
        producer_groups = deterministic_spatial_groups(
            [metadata_by_name[name] for name in sorted(producer_names)],
            CHALLENGE_PRODUCER_GROUPS,
        )
        injector_groups = deterministic_spatial_groups(
            [metadata_by_name[name] for name in sorted(injector_names)],
            CHALLENGE_INJECTOR_GROUPS,
        )
        modified = scale_schedule_with_role_policies(
            original,
            producer_well_groups=producer_groups,
            injector_well_groups=injector_groups,
            producer_group_nodes=producer_nodes,
            injector_group_nodes=injector_nodes,
            node_dates=CHALLENGE_NODE_DATES,
            effective_from=CHALLENGE_NODE_DATES[0],
            max_wlpr=500.0,
        )
        schedule.write_text(modified, encoding="utf-8", newline="\n")
        schedule_sha = _sha256_bytes(modified.encode())
        (args.output / "wells_schedule.inc").write_text(modified, encoding="utf-8", newline="\n")

        flow_result = run_flow(
            FlowRequest(
                deck=deck,
                output_dir=args.output,
                extra_args=("--parsing-strictness=low",),
                timeout_seconds=args.timeout,
            )
        )
        result_payload.update(
            {
                "status": flow_result.status,
                "returncode": flow_result.returncode,
                "runtime_seconds": flow_result.runtime_seconds,
                "schedule_sha256": schedule_sha,
                "summary_install_changed_files": list(install_report.changed_files),
            }
        )
        if flow_result.status == "success":
            compact = _export_compact_summary(args.output, wells)
            if compact["max_wlpr"] > 500.0001:
                raise ValueError(f"actual OPM WLPR exceeds contract limit: {compact['max_wlpr']}")
            density = load_model_z_density_map(root)
            economics = scenario_chdd(
                args.output / "summary.npz",
                oil_density_t_m3=density.oil_t_m3,
                water_density_t_m3=density.water_t_m3,
                report_mode="all",
            )
            (args.output / "economics.json").write_text(
                json.dumps(economics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            result_payload["compact_summary"] = compact
            result_payload["npv_mrub"] = float(economics["summary"]["totalChddM"])

    (args.output / "candidate-result.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result_payload, ensure_ascii=False, indent=2))
    if result_payload.get("status") != "success":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
