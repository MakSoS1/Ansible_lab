from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

from aios_track2.opm import FlowRequest, run_flow
from inspect_model_z import find_root_deck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model-z-baseline-") as td:
        root = Path(td)
        with zipfile.ZipFile(args.archive) as zf:
            zf.extractall(root)
        deck, meta = find_root_deck(root)
        result = run_flow(FlowRequest(deck=deck, output_dir=args.output, timeout_seconds=args.timeout))
        manifest = {
            "status": result.status,
            "returncode": result.returncode,
            "runtime_seconds": result.runtime_seconds,
            "stdout_sha256": result.stdout_sha256,
            "stderr_sha256": result.stderr_sha256,
            "dimensions": meta["dimensions"],
            "well_count": meta["well_count"],
            "output_files": [str(p) for p in result.output_files],
        }
        (args.output / "baseline-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        if result.status != "success":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
