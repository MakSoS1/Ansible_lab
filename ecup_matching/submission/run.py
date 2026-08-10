from __future__ import annotations

import argparse
from pathlib import Path

from ecup_matching.submission.predict import predict_to_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True, help="output CSV file")
    parser.add_argument("--items_path", type=str, required=True, help="test items parquet path")
    parser.add_argument("--matches_path", type=str, required=True, help="test matches parquet path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    predict_to_csv(
        items_path=Path(args.items_path),
        matches_path=Path(args.matches_path),
        model_path=root / "model_v1.joblib",
        manifest_path=root / "model_v1_manifest.json",
        output_path=Path(args.output_path),
        chunk_size=50_000,
    )


if __name__ == "__main__":
    main()
