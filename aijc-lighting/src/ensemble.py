from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .solutions import CVResult


def _norm(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def select_best(
    results: list[CVResult],
    y: np.ndarray,
    group_scores: dict[str, float] | None = None,
) -> dict:
    if not results:
        raise ValueError('at least one result is required')
    y = np.asarray(y, dtype=int)

    def reliable_score(r: CVResult) -> float:
        if group_scores and r.name in group_scores:
            return float(group_scores[r.name])
        return float(r.oof_accuracy)

    base = max(results, key=lambda r: (reliable_score(r), r.oof_accuracy, r.name))
    best = base
    trace = [
        {
            'name': r.name,
            'oof_accuracy': float(r.oof_accuracy),
            'reliable_score': reliable_score(r),
        }
        for r in results
    ]

    for i, a in enumerate(results):
        for b in results[i + 1:]:
            if group_scores and min(reliable_score(a), reliable_score(b)) < reliable_score(base) - 0.02:
                continue
            for alpha in np.arange(0.1, 1.0, 0.05):
                oof = _norm(alpha * a.oof_probs + (1.0 - alpha) * b.oof_probs)
                acc = float(accuracy_score(y, oof.argmax(1)))
                if acc > best.oof_accuracy + 1e-12:
                    test = _norm(alpha * a.test_probs + (1.0 - alpha) * b.test_probs)
                    best = CVResult(
                        name=f'ensemble_{a.name}__{b.name}__a{alpha:.2f}',
                        oof_probs=oof,
                        test_probs=test,
                        fold_scores=[],
                        oof_accuracy=acc,
                        metadata={'components': [a.name, b.name], 'alpha': float(alpha)},
                    )

    if best not in results:
        current = best
        for c in results:
            if c.name in current.metadata.get('components', []):
                continue
            for beta in (0.10, 0.20, 0.30):
                oof = _norm((1.0 - beta) * current.oof_probs + beta * c.oof_probs)
                acc = float(accuracy_score(y, oof.argmax(1)))
                if acc > best.oof_accuracy + 1e-12:
                    test = _norm((1.0 - beta) * current.test_probs + beta * c.test_probs)
                    best = CVResult(
                        name=f'ensemble3_{current.name}__{c.name}__b{beta:.2f}',
                        oof_probs=oof,
                        test_probs=test,
                        fold_scores=[],
                        oof_accuracy=acc,
                        metadata={
                            'components': current.metadata.get('components', []) + [c.name],
                            'beta': float(beta),
                        },
                    )

    return {
        'result': best,
        'base_best': base.name,
        'base_best_accuracy': float(base.oof_accuracy),
        'selected_name': best.name,
        'selected_accuracy': float(best.oof_accuracy),
        'candidates': trace,
    }


def write_submission(
    test_ids,
    probs: np.ndarray,
    path: Path | str,
    expected_rows: int | None = 300,
) -> Path:
    ids = [str(x) for x in test_ids]
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (len(ids), 3):
        raise ValueError(f'probability shape must be ({len(ids)}, 3), got {probs.shape}')
    if expected_rows is not None and len(ids) != expected_rows:
        raise ValueError(f'expected {expected_rows} test ids, got {len(ids)}')
    if not np.isfinite(probs).all():
        raise ValueError('submission probabilities contain NaN/Inf')
    labels = probs.argmax(axis=1).astype(int)
    if not set(np.unique(labels)).issubset({0, 1, 2}):
        raise ValueError('predicted labels must be in {0,1,2}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'id': ids, 'label': labels}).to_csv(path, index=False)
    return path
