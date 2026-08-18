from pathlib import Path
import pandas as pd

from ecup_matching.ml.run_v20_anchor_proxy import evaluate_anchor_predictions


def test_anchor_proxy_calibrates_exact_public_order(tmp_path: Path):
    truth = pd.DataFrame({
        "id1": [1, 2, 3, 4] * 4,
        "id2": [11, 12, 13, 14] * 4,
        "target": [1, 0, 1, 0] * 4,
        "category": [f"c{i}" for i in range(4) for _ in range(4)],
        "reason_code": ["OTHER"] * 16,
    })
    truth_path = tmp_path / "truth.parquet"
    truth.to_parquet(truth_path, index=False)
    # Construct monotonically better rankings for each anchor.
    preds = {}
    for name, positive, negative in [
        ("v7", 0.55, 0.45), ("v13B", 0.65, 0.35),
        ("v12", 0.75, 0.25), ("v14", 0.85, 0.15),
    ]:
        frame = truth[["id1", "id2"]].copy()
        frame["predict"] = [positive if y else negative for y in truth.target]
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        preds[name] = path
    report = evaluate_anchor_predictions(truth_path=truth_path, predictions=preds)
    assert report["proxy_calibration"]["axes"]["proxy_macro_ap"]["promotable"] is True
    assert report["best_anchor"] == "v14"
