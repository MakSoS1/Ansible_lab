from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def macro_ap(frame: pd.DataFrame, score: np.ndarray) -> float:
    aps = []
    for category in sorted(frame["category"].astype(str).unique().tolist()):
        mask = frame["category"].astype(str).to_numpy() == category
        aps.append(float(average_precision_score(frame.loc[mask, "target"].to_numpy(), score[mask])))
    return float(np.mean(aps))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--oof", type=Path, required=True)
    p.add_argument("--matches", type=Path, required=True)
    p.add_argument("--items", type=Path, required=True)
    args = p.parse_args()

    oof = pd.read_parquet(args.oof)
    print("V12_OOF_COLUMNS=" + json.dumps(list(oof.columns)))
    print("V12_OOF_DTYPES=" + json.dumps({c: str(t) for c, t in oof.dtypes.items()}, sort_keys=True))
    if "row_index" not in oof:
        return 0
    matches = pd.read_parquet(args.matches, columns=["id1", "id2", "target"])
    items = pd.read_parquet(args.items, columns=["id", "category"]).drop_duplicates("id")
    cmap = dict(zip(items["id"].tolist(), items["category"].astype(str).tolist()))
    rows = oof["row_index"].to_numpy(np.int64)
    work = matches.iloc[rows].reset_index(drop=True)
    work["category"] = work["id1"].map(cmap)
    report = {}
    for col in oof.columns:
        if col in {"row_index", "fold"} or not pd.api.types.is_numeric_dtype(oof[col]):
            continue
        score = oof[col].to_numpy(dtype=np.float64)
        if len(score) != len(work) or not np.isfinite(score).all() or len(np.unique(score)) < 2:
            continue
        report[col] = macro_ap(work, score)
    print("V12_NUMERIC_OOF_AP=" + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
