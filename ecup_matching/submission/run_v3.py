from __future__ import annotations

import argparse
from pathlib import Path

from ecup_matching.submission.predict_v3 import predict_to_csv_v3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", type=Path, required=True)
    parser.add_argument("--matches_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()
    predict_to_csv_v3(
        args.items_path,
        args.matches_path,
        Path("model_v2.joblib"),
        Path("model_v2_manifest.json"),
        Path("model_v3"),
        Path("model_v3_manifest.json"),
        args.output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
