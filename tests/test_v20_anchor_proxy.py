from pathlib import Path
import pandas as pd

from ecup_matching.ml.run_v20_anchor_proxy import evaluate_anchor_predictions


def test_anchor_proxy_calibrates_exact_public_order(tmp_path: Path):
    rows = []
    for c in range(4):
        for i, target in enumerate([1, 1, 1, 1, 0, 0, 0, 0]):
            rows.append({
                "id1": c * 100 + i,
                "id2": c * 100 + i + 50,
                "target": target,
                "category": f"c{c}",
                "reason_code": "OTHER",
            })
    truth = pd.DataFrame(rows)
    truth_path = tmp_path / "truth.parquet"
    truth.to_parquet(truth_path, index=False)

    # Per category, v14 is perfect, v12 has one high-ranked negative, v13B
    # has two, and v7 has three. This creates the exact Public ordering.
    orders = {
        "v14": [0.95, 0.90, 0.85, 0.80, 0.40, 0.30, 0.20, 0.10],
        "v12": [0.95, 0.90, 0.85, 0.50, 0.80, 0.30, 0.20, 0.10],
        "v13B": [0.95, 0.90, 0.50, 0.40, 0.85, 0.80, 0.20, 0.10],
        "v7": [0.95, 0.50, 0.40, 0.30, 0.90, 0.85, 0.80, 0.10],
    }
    preds = {}
    for name, one_category_scores in orders.items():
        frame = truth[["id1", "id2"]].copy()
        frame["predict"] = one_category_scores * 4
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        preds[name] = path
    report = evaluate_anchor_predictions(truth_path=truth_path, predictions=preds)
    axis = report["proxy_calibration"]["axes"]["proxy_macro_ap"]
    assert axis["promotable"] is True
    assert axis["observed_order"] == ["v14", "v12", "v13B", "v7"]
    assert report["best_anchor"] == "v14"
