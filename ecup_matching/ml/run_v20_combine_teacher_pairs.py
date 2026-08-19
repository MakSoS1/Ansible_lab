"""Combine multiple pair parquet files into one id1/id2 teacher-inference queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", action="append", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    frames = []
    raw_rows = 0
    for path in a.input:
        frame = pd.read_parquet(path, columns=["id1", "id2"])
        raw_rows += len(frame)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["id1", "id2"], keep="first")
    out = out.sort_values(["id1", "id2"], kind="mergesort").reset_index(drop=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.output, index=False)
    report = {"version":"v20-teacher-queue-v1","inputs":[str(p) for p in a.input],"raw_rows":int(raw_rows),"unique_rows":int(len(out))}
    a.output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
