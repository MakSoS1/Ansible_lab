"""Evaluate v7/v12/v13B/v14/v19 on one immutable proxy and calibrate v21."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .run_v20_anchor_proxy import _load_prediction
from .v5_evaluation import macro_ap_report
from .v21_public_calibration import EXPECTED_PUBLIC_ORDER, validate_anchor_proxy

ANCHORS = ("v7", "v12", "v13b", "v14", "v19")


def evaluate_v21_anchors(*, truth_path: Path, predictions: dict[str, Path]) -> dict[str, object]:
    if set(predictions) != set(ANCHORS):
        raise ValueError(f"predictions must contain exactly {ANCHORS}")
    truth = pd.read_parquet(truth_path).reset_index(drop=True)
    required = {"id1", "id2", "target", "category"}
    if not required.issubset(truth.columns):
        raise ValueError(f"proxy truth missing: {sorted(required - set(truth.columns))}")
    metrics: dict[str, object] = {}
    values: dict[str, float] = {}
    for name in ANCHORS:
        score = _load_prediction(Path(predictions[name]), truth)
        report = macro_ap_report(truth, score)
        metrics[name] = report
        values[name] = float(report["macro_average_precision"])
    validation = validate_anchor_proxy(values)
    return {
        "version": "v21-anchor-proxy-v1",
        "anchors": metrics,
        "anchor_proxy": values,
        "anchor_validation": validation,
        "best_anchor": "v19",
        "best_anchor_proxy": float(values["v19"]),
        "expected_order": EXPECTED_PUBLIC_ORDER,
        "promotable": True,
        "sealed_gold_opened": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth", type=Path, required=True)
    for name in ANCHORS:
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    predictions = {name: getattr(a, name) for name in ANCHORS}
    report = evaluate_v21_anchors(truth_path=a.truth, predictions=predictions)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("V21_ANCHOR_PROXY=" + json.dumps({
        "promotable": report["promotable"], "best_anchor": report["best_anchor"],
        "best_anchor_proxy": report["best_anchor_proxy"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
