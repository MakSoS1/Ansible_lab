from __future__ import annotations

import argparse
from pathlib import Path

from ecup_matching.submission.predict_v11_no_contrastive import predict_to_csv_v11_no_contrastive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", required=True, type=Path)
    parser.add_argument("--matches_path", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    predict_to_csv_v11_no_contrastive(
        items_path=args.items_path,
        matches_path=args.matches_path,
        structured_model_path=root / "model_v5_structured.joblib",
        teacher_model_dir=root / "model_v5_teacher",
        category_model_path=root / "model_v11_category_shrunk.json",
        hgb_model_path=root / "model_v11_hgb_meta.joblib",
        runtime_root=root,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
