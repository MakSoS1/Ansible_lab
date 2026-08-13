from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecup_matching.submission.predict_v11 import predict_to_csv_v11


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output_path", type=Path, required=True)
    p.add_argument("--items_path", type=Path, required=True)
    p.add_argument("--matches_path", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    metadata = json.loads((root / "V11_KEEPER.json").read_text(encoding="utf-8"))
    if metadata.get("sealed_gold_evaluated") is not False:
        raise RuntimeError("v11 sealed-gold contract violated")
    if metadata.get("runtime_neural_models") is not False:
        raise RuntimeError("v11 keeper unexpectedly requires a neural runtime")
    predict_to_csv_v11(
        items_path=a.items_path,
        matches_path=a.matches_path,
        model_path=root / "model_v11.joblib",
        output_path=a.output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
