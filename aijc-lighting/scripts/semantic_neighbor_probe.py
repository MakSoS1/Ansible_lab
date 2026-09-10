from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data import discover_layout, load_manifests, resolve_image_paths
from src.semantic_neighbors import cosine_knn_probabilities, fuse_normalized_views


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ViewDataset(Dataset):
    def __init__(self, paths, view: str, image_size: int = 224):
        self.paths = list(paths)
        self.view = view
        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as im:
            im = im.convert("RGB")
            if self.view == "equalized":
                y, cb, cr = im.convert("YCbCr").split()
                y = ImageOps.equalize(y)
                im = Image.merge("YCbCr", (y, cb, cr)).convert("RGB")
            elif self.view == "autocontrast_gray":
                y = ImageOps.autocontrast(ImageOps.grayscale(im), cutoff=1)
                im = Image.merge("RGB", (y, y, y))
            return self.tf(im)


def extract_embeddings(model, paths, view: str, batch_size=24, workers=2):
    ds = ViewDataset(paths, view=view)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    rows = []
    with torch.inference_mode():
        for i, x in enumerate(dl):
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.ndim > 2:
                out = out.flatten(2).mean(-1)
            rows.append(out.cpu().numpy())
            if i % 10 == 0:
                print(view, min((i + 1) * batch_size, len(ds)), "/", len(ds), flush=True)
    return np.concatenate(rows).astype(np.float32)


def cv_probe(x, xt, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    results = {}
    saved = {}
    candidates = {
        "logreg_c03": lambda: LogisticRegression(C=0.3, max_iter=3000),
        "logreg_c1": lambda: LogisticRegression(C=1.0, max_iter=3000),
        "rbf_c1": lambda: SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
        "rbf_c3": lambda: SVC(C=3.0, kernel="rbf", probability=True, random_state=42),
    }
    for name, factory in candidates.items():
        oof = np.zeros((len(y), 3), np.float32)
        test = np.zeros((len(xt), 3), np.float32)
        for tr, va in cv.split(x, y):
            m = factory()
            m.fit(x[tr], y[tr])
            oof[va] = m.predict_proba(x[va])
            test += m.predict_proba(xt) / 5.0
        acc = float(accuracy_score(y, oof.argmax(1)))
        print(name, acc, flush=True)
        results[name] = acc
        saved[name] = (oof, test)
    return results, saved


def complement_rule_diagnostics(features, y):
    x = normalize(features)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)[:, :8]
    rows = np.arange(len(y))[:, None]
    ss = sim[rows, order]
    ll = y[order]
    diagnostics = {}
    for threshold in (0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97):
        mask = (ss[:, 1] >= threshold) & (ll[:, 0] != ll[:, 1])
        complement = 3 - ll[:, 0] - ll[:, 1]
        valid = mask & (complement >= 0) & (complement <= 2)
        coverage = float(valid.mean())
        acc = float((complement[valid] == y[valid]).mean()) if valid.any() else None
        diagnostics[str(threshold)] = {"coverage": coverage, "accuracy": acc, "count": int(valid.sum())}
    return diagnostics, order, ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--output", default="outputs-semantic-neighbors")
    ap.add_argument("--batch-size", type=int, default=24)
    args = ap.parse_args()

    import timm

    torch.set_num_threads(min(4, os.cpu_count() or 4))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    layout = discover_layout(args.data)
    train_df, test_df, _ = load_manifests(layout)
    train_paths, test_paths = resolve_image_paths(layout)
    paths = train_paths + test_paths
    y = train_df.label.to_numpy(int)

    model_name = "vit_small_patch14_dinov2.lvd142m"
    model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="token").eval()

    views = {}
    for view in ("clean", "equalized", "autocontrast_gray"):
        views[view] = extract_embeddings(model, paths, view, args.batch_size)
        np.save(out / f"embeddings_{view}.npy", views[view])

    n = len(y)
    variants = {
        "clean": (normalize(views["clean"][:n]), normalize(views["clean"][n:])),
        "equalized": (normalize(views["equalized"][:n]), normalize(views["equalized"][n:])),
        "gray": (normalize(views["autocontrast_gray"][:n]), normalize(views["autocontrast_gray"][n:])),
        "fused_clean_equalized": (
            fuse_normalized_views([views["clean"][:n], views["equalized"][:n]]),
            fuse_normalized_views([views["clean"][n:], views["equalized"][n:]]),
        ),
        "fused_all": (
            fuse_normalized_views([views[v][:n] for v in views]),
            fuse_normalized_views([views[v][n:] for v in views]),
        ),
    }

    report = {"model": model_name, "variants": {}, "complement_rule": {}}
    best_acc = -1.0
    best_name = None
    best_probs = None

    for variant, (x, xt) in variants.items():
        row = {}
        for k in (1, 3, 5, 9, 15, 25):
            oof = cosine_knn_probabilities(x, y, x, k=k, temperature=0.07, exclude_self=True)
            test = cosine_knn_probabilities(x, y, xt, k=k, temperature=0.07)
            acc = float(accuracy_score(y, oof.argmax(1)))
            row[f"knn_{k}"] = acc
            if acc > best_acc:
                best_acc = acc
                best_name = f"{variant}:knn_{k}"
                best_probs = test
        learned, saved = cv_probe(x, xt, y)
        row.update(learned)
        for name, acc in learned.items():
            if acc > best_acc:
                best_acc = acc
                best_name = f"{variant}:{name}"
                best_probs = saved[name][1]
        report["variants"][variant] = row
        diag, order, sims = complement_rule_diagnostics(x, y)
        report["complement_rule"][variant] = diag

        # Save the most similar cross-split neighbours for manual audit.
        cross = normalize(xt) @ normalize(x).T
        top = np.argsort(-cross, axis=1)[:, :5]
        audit_rows = []
        for qi in range(len(xt)):
            for rank, ri in enumerate(top[qi]):
                audit_rows.append(
                    {
                        "test_id": test_df.id.iloc[qi],
                        "rank": rank + 1,
                        "train_id": train_df.id.iloc[ri],
                        "train_label": int(y[ri]),
                        "cosine": float(cross[qi, ri]),
                    }
                )
        pd.DataFrame(audit_rows).to_csv(out / f"cross_neighbors_{variant}.csv", index=False)

    report.update({"best_name": best_name, "best_loo_or_cv_accuracy": best_acc, "seconds": time.time() - started})
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if best_probs is not None:
        np.save(out / "test_probs.npy", best_probs)
        pd.DataFrame({"id": test_df.id, "label": best_probs.argmax(1)}).to_csv(out / "submission.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
