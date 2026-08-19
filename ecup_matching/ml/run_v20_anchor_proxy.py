"""Evaluate exact historical-submission prediction CSVs on one immutable v20 proxy fixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from .v5_evaluation import macro_ap_report
from .v20_proxy import calibrate_proxy_axes


ANCHORS = ("v7", "v12", "v13B", "v14")


def _load_prediction(path: Path, truth: pd.DataFrame) -> pd.Series:
    pred = pd.read_csv(path)
    required = {"id1", "id2", "predict"}
    if not required.issubset(pred.columns):
        raise ValueError(f"prediction {path} missing columns: {sorted(required - set(pred.columns))}")
    if len(pred) != len(truth):
        raise ValueError(f"prediction {path} row count mismatch")
    if not pred[["id1", "id2"]].reset_index(drop=True).equals(truth[["id1", "id2"]].reset_index(drop=True)):
        raise ValueError(f"prediction {path} pair order mismatch")
    score = pd.to_numeric(pred["predict"], errors="raise").astype(float)
    if not score.map(lambda x: x == x and abs(x) != float("inf")).all():
        raise ValueError(f"prediction {path} contains non-finite values")
    return score


def evaluate_anchor_predictions(*, truth_path: Path, predictions: Mapping[str, Path]) -> dict[str, object]:
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
    calibration = calibrate_proxy_axes({
        "proxy_macro_ap": {"higher_is_better": True, "values": values}
    })
    best_anchor = max(values, key=values.get)
    payload = {
        "version": "v20-anchor-proxy-v1",
        "anchors": metrics,
        "proxy_calibration": calibration,
        "best_anchor": best_anchor,
        "best_anchor_proxy": float(values[best_anchor]),
        "expected_best_anchor": "v14",
        "promotable": bool(
            calibration["axes"]["proxy_macro_ap"]["promotable"] and best_anchor == "v14"
        ),
        "sealed_gold_opened": False,
    }
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth", type=Path, required=True)
    for name in ANCHORS:
        p.add_argument(f"--{name.lower()}", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    predictions = {name: getattr(a, name.lower()) for name in ANCHORS}
    report = evaluate_anchor_predictions(truth_path=a.truth, predictions=predictions)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print("V20_ANCHOR_PROXY=" + json.dumps({
        "promotable": report["promotable"], "best_anchor": report["best_anchor"],
        "best_anchor_proxy": report["best_anchor_proxy"]
    }, sort_keys=True), flush=True)
    return 0 if report["promotable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
