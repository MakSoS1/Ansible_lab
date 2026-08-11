from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecup_matching.submission.predict_v6 import predict_to_csv_v6


def submission_root(run_file: Path) -> Path:
    return Path(run_file).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--items_path", type=Path, required=True)
    parser.add_argument("--matches_path", type=Path, required=True)
    args = parser.parse_args()

    root = submission_root(Path(__file__))
    metadata = json.loads(
        (root / "model_v6_gate_metadata.json").read_text(encoding="utf-8")
    )
    coverage = float(metadata["coverage"])
    predict_to_csv_v6(
        coverage=coverage,
        items_path=args.items_path,
        matches_path=args.matches_path,
        structured_model_path=root / "model_v5_structured.joblib",
        contrastive_model_dir=root / "model_v5_contrastive",
        teacher_model_dir=root / "model_v5_teacher",
        category_model_path=root / "model_v6_category_shrunk.json",
        hgb_model_path=root / "model_v6_hgb_meta.joblib",
        runtime_root=root,
        output_path=args.output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
