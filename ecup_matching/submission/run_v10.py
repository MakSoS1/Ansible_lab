from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecup_matching.submission.predict_v10 import predict_to_csv_v10, validate_v10_metadata


def submission_root(run_file: Path) -> Path:
    return Path(run_file).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--items_path", type=Path, required=True)
    parser.add_argument("--matches_path", type=Path, required=True)
    args = parser.parse_args()

    root = submission_root(Path(__file__))
    metadata = validate_v10_metadata(
        json.loads((root / "model_v10_metadata.json").read_text(encoding="utf-8"))
    )
    predict_to_csv_v10(
        items_path=args.items_path,
        matches_path=args.matches_path,
        model_dir=root / "model_v10_student",
        output_path=args.output_path,
        max_length=int(metadata["max_length"]),
        max_chars=int(metadata["max_chars"]),
        batch_size=int(metadata["inference_batch_size"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
