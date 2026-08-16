from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecup_matching.submission.predict_v15 import predict_to_csv_v15


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output_path", type=Path, required=True)
    p.add_argument("--items_path", type=Path, required=True)
    p.add_argument("--matches_path", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parent.parent.parent if Path(__file__).name != "run.py" else Path(__file__).resolve().parent
    meta_path = root / "model_v15_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("gold_metric_opened") is not False or int(meta.get("gold_rows_scored", -1)) != 0:
        raise RuntimeError("invalid v15 sealed-gold provenance")
    predict_to_csv_v15(
        items_path=a.items_path,
        matches_path=a.matches_path,
        checkpoint_path=root / "v15_model.pt",
        base_model_dir=root / "v15_base_config",
        output_path=a.output_path,
        batch_size=int(meta.get("inference_batch_size", 64)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
