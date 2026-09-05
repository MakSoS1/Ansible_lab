from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path

from aios_track2.opm import FlowRequest, run_flow
from aios_track2.summary_install import install_training_summary
from inspect_model_z import find_root_deck


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=9000)
    parser.add_argument("--parsing-strictness", choices=("normal", "low"), default="low")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="model-z-training-summary-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(args.archive) as zf:
            zf.extractall(root)
        deck, meta = find_root_deck(root)
        install_report = install_training_summary(root)
        summary_path = root / install_report.summary_path
        (args.output / "training-summary.inc").write_text(
            summary_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
        )

        extra_args = (f"--parsing-strictness={args.parsing_strictness}",)
        result = run_flow(
            FlowRequest(
                deck=deck,
                output_dir=args.output,
                extra_args=extra_args,
                timeout_seconds=args.timeout,
            )
        )
        manifest = {
            "status": result.status,
            "returncode": result.returncode,
            "runtime_seconds": result.runtime_seconds,
            "stdout_sha256": result.stdout_sha256,
            "stderr_sha256": result.stderr_sha256,
            "dimensions": meta["dimensions"],
            "well_count": meta["well_count"],
            "flow_args": list(extra_args),
            "summary_install": asdict(install_report),
            "output_files": [str(path) for path in result.output_files],
        }
        (args.output / "training-summary-manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2))
        if result.status != "success":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
