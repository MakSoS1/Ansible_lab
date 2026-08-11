from __future__ import annotations

import argparse
from pathlib import Path

from ecup_matching.submission.predict_v5 import predict_to_csv_v5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--items_path", type=Path, required=True)
    parser.add_argument("--matches_path", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    predict_to_csv_v5(
        items_path=args.items_path,
        matches_path=args.matches_path,
        structured_model_path=root / "model_v5_structured.joblib",
        contrastive_model_dir=root / "model_v5_contrastive",
        teacher_model_dir=root / "model_v5_teacher",
        runtime_root=root,
        output_path=args.output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
