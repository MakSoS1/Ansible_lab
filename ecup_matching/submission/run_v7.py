from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecup_matching.submission.predict_v7 import (
    predict_to_csv_v7,
    validate_v7_metadata,
)


def submission_root(run_file: Path) -> Path:
    return Path(run_file).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--items_path", type=Path, required=True)
    parser.add_argument("--matches_path", type=Path, required=True)
    args = parser.parse_args()

    root = submission_root(Path(__file__))
    metadata = validate_v7_metadata(
        json.loads((root / "model_v7_metadata.json").read_text(encoding="utf-8"))
    )

    predict_to_csv_v7(
        items_path=args.items_path,
        matches_path=args.matches_path,
        model_dir=root / "model_v7_teacher",
        output_path=args.output_path,
        max_length=int(metadata.get("max_length", 256)),
        max_chars=int(metadata.get("max_chars", 900)),
        batch_size=int(metadata.get("inference_batch_size", 64)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
