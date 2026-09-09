from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.data import discover_layout, load_manifests, resolve_image_paths


def extract_reference_features(path: str | Path) -> np.ndarray:
    """Faithful adaptation of the public exact-task fast tabular feature recipe."""
    with Image.open(path) as im:
        rgb_im = im.convert("RGB")
        rgb = np.asarray(rgb_im, dtype=np.float32)
        hsv = np.asarray(rgb_im.convert("HSV"), dtype=np.float32)

    feats: list[float] = []

    def add_stats(ch: np.ndarray) -> None:
        q = np.percentile(ch, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        feats.extend(
            [
                float(ch.mean()),
                float(ch.std()),
                *map(float, q),
                float((ch > 240).mean()),
                float((ch > 250).mean()),
                float((ch < 15).mean()),
                float((ch < 5).mean()),
            ]
        )

    for c in range(3):
        add_stats(rgb[:, :, c])
    for c in range(3):
        add_stats(hsv[:, :, c])

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    add_stats(lum)

    feats.extend(
        [
            float((r / (g + 1e-5)).mean()),
            float((b / (g + 1e-5)).mean()),
            float((r / (b + 1e-5)).mean()),
        ]
    )

    for values in (r, g, b, lum, hsv[:, :, 1], hsv[:, :, 2]):
        hist, _ = np.histogram(values, bins=16, range=(0, 256), density=True)
        feats.extend(map(float, hist))

    h, w = lum.shape
    hs, ws = h // 3, w // 3
    for i in range(3):
        for j in range(3):
            cell_l = lum[i * hs : (i + 1) * hs, j * ws : (j + 1) * ws]
            cell_s = hsv[i * hs : (i + 1) * hs, j * ws : (j + 1) * ws, 1]
            feats.extend([float(cell_l.mean()), float(cell_l.std()), float(cell_s.mean())])

    center = lum[hs : 2 * hs, ws : 2 * ws]
    border_mask = np.ones_like(lum, dtype=bool)
    border_mask[hs : 2 * hs, ws : 2 * ws] = False
    cm = float(center.mean())
    bm = float(lum[border_mask].mean())
    feats.extend([cm, bm, cm / (bm + 1e-5)])
    return np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)


def extract_many(paths, workers: int) -> np.ndarray:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(extract_reference_features, map(str, paths), chunksize=8))
    return np.stack(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--output", default="outputs-reference-fast")
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    layout = discover_layout(args.data)
    train_df, test_df, _ = load_manifests(layout)
    train_paths, test_paths = resolve_image_paths(layout)
    y = train_df.label.to_numpy(int)

    x = extract_many(train_paths, args.workers)
    xt = extract_many(test_paths, args.workers)
    print("feature shapes", x.shape, xt.shape, flush=True)
    np.savez_compressed(out / "features.npz", X_train=x, X_test=xt, y=y)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    factories = {
        "CatBoost": lambda: CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6, verbose=0, random_seed=42, thread_count=4),
        "LightGBM": lambda: LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4),
        "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=600, max_depth=15, random_state=42, n_jobs=-1),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=600, max_depth=15, random_state=42, n_jobs=-1),
        "HistGB": lambda: HistGradientBoostingClassifier(max_iter=300, random_state=42),
    }

    saved = {}
    metrics = {"feature_count": int(x.shape[1]), "models": {}}
    for name, factory in factories.items():
        oof = np.zeros((len(y), 3), dtype=np.float32)
        test = np.zeros((len(xt), 3), dtype=np.float32)
        for tr, va in cv.split(x, y):
            model = factory()
            model.fit(x[tr], y[tr])
            oof[va] = model.predict_proba(x[va])
            test += model.predict_proba(xt) / 5.0
        acc = float(accuracy_score(y, oof.argmax(1)))
        metrics["models"][name] = acc
        saved[name] = (oof, test)
        print(name, acc, flush=True)

    names = list(saved)
    best_name = max(names, key=lambda n: metrics["models"][n])
    best_acc = metrics["models"][best_name]
    best_oof, best_test = saved[best_name]

    # Exhaustive pairwise probability blends are cheap and harder to overfit than a huge weight grid.
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for wa in np.arange(0.1, 1.0, 0.1):
                oof = wa * saved[a][0] + (1.0 - wa) * saved[b][0]
                acc = float(accuracy_score(y, oof.argmax(1)))
                if acc > best_acc:
                    best_acc = acc
                    best_name = f"{a}+{b}@{wa:.1f}"
                    best_oof = oof
                    best_test = wa * saved[a][1] + (1.0 - wa) * saved[b][1]

    metrics.update(
        {
            "selected": best_name,
            "selected_accuracy": best_acc,
            "confusion_matrix": confusion_matrix(y, best_oof.argmax(1)).tolist(),
            "test_distribution": dict(Counter(best_test.argmax(1).tolist())),
            "seconds": time.time() - started,
        }
    )
    np.save(out / "oof_probs.npy", best_oof)
    np.save(out / "test_probs.npy", best_test)
    pd.DataFrame({"id": test_df.id, "label": best_test.argmax(1)}).to_csv(out / "submission.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
