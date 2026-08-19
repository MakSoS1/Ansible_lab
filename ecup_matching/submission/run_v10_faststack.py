from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecup_matching.submission.predict_v10_faststack import (
    CANDIDATE,
    assert_no_teacher_assets,
    predict_to_csv_v10_faststack,
)


MIN_STRICT_GRAPH_OOF = 0.595
RUNTIME_VERSION = "v10-faststack-overlap-v1"


def submission_root(run_file: Path) -> Path:
    return Path(run_file).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--items_path", type=Path, required=True)
    parser.add_argument("--matches_path", type=Path, required=True)
    args = parser.parse_args()

    root = submission_root(Path(__file__))
    assert_no_teacher_assets(root)
    metadata = json.loads((root / "NO_TEACHER_KEEPER.json").read_text(encoding="utf-8"))
    if metadata.get("candidate") != CANDIDATE:
        raise RuntimeError("v10 keeper candidate is not no_teacher")
    if float(metadata.get("strict_graph_oof_macro_ap", 0.0)) < MIN_STRICT_GRAPH_OOF:
        raise RuntimeError("v10 keeper violates frozen strict graph OOF gate")
    if metadata.get("selection_gold_metric_opened") is not False:
        raise RuntimeError("v10 keeper violates sealed-gold contract")
    if int(metadata.get("selection_gold_rows_scored", -1)) != 0:
        raise RuntimeError("v10 keeper scored sealed-gold rows")

    predict_to_csv_v10_faststack(
        items_path=args.items_path,
        matches_path=args.matches_path,
        structured_model_path=root / "model_v5_structured.joblib",
        contrastive_model_dir=root / "model_v5_contrastive",
        category_model_path=root / "model_v10_category_shrunk.json",
        hgb_model_path=root / "model_v10_hgb_meta.joblib",
        runtime_root=root,
        output_path=args.output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
