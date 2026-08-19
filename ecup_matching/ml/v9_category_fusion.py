from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np
from sklearn.metrics import average_precision_score

from .v5_fixed_blend import percentile_rank


def _validated_weight_grid(weight_grid: Iterable[float]) -> tuple[float,...]:
    grid=tuple(float(x) for x in weight_grid)
    if not grid or any((not math.isfinite(x)) or x<0.0 or x>1.0 for x in grid):
        raise ValueError('weight_grid must contain finite values in [0,1]')
    if tuple(sorted(set(grid)))!=grid:
        raise ValueError('weight_grid must be sorted and unique')
    return grid


def _select_category_weights(
    score_a: np.ndarray,
    score_b: np.ndarray,
    target: np.ndarray,
    categories: np.ndarray,
    train_indices: np.ndarray,
    *,
    weight_grid: tuple[float,...],
) -> dict[str,float]:
    train=np.asarray(train_indices,dtype=np.int64); y=np.asarray(target,dtype=np.int8); cats=np.asarray(categories).astype(str)
    weights={}
    for cat in sorted(np.unique(cats).tolist()):
        idx=train[cats[train]==cat]
        cy=y[idx]
        if len(idx)==0 or len(np.unique(cy))<2:
            weights[cat]=0.5 if 0.5 in weight_grid else min(weight_grid,key=lambda x:abs(x-.5))
            continue
        best=None
        for w in weight_grid:
            score=w*score_a[idx]+(1.0-w)*score_b[idx]
            ap=float(average_precision_score(cy,score))
            # Stable tie-break toward the historical 0.5 fusion.
            key=(ap,-abs(w-.5),-w)
            if best is None or key>best[0]: best=(key,w)
        weights[cat]=float(best[1])
    return weights


def crossfit_category_fusion(
    category_score,
    hgb_score,
    target,
    categories,
    folds,
    *,
    weight_grid=(0.0,0.25,0.5,0.75,1.0),
) -> dict[str,object]:
    grid=_validated_weight_grid(weight_grid)
    a=percentile_rank(np.asarray(category_score,dtype=np.float64)); b=percentile_rank(np.asarray(hgb_score,dtype=np.float64))
    y=np.asarray(target,dtype=np.int8); cats=np.asarray(categories).astype(str); f=np.asarray(folds)
    if not (len(a)==len(b)==len(y)==len(cats)==len(f)): raise ValueError('all inputs must align')
    unique=sorted(np.unique(f).tolist())
    if len(unique)<2: raise ValueError('cross-fitting requires at least two folds')
    all_idx=np.arange(len(y),dtype=np.int64); out=np.full(len(y),np.nan); fold_weights={}
    for fold in unique:
        train=all_idx[f!=fold]; valid=all_idx[f==fold]
        weights=_select_category_weights(a,b,y,cats,train,weight_grid=grid); fold_weights[int(fold)]=weights
        w=np.asarray([weights[c] for c in cats[valid]],dtype=np.float64)
        out[valid]=w*a[valid]+(1.0-w)*b[valid]
    if not np.isfinite(out).all(): raise RuntimeError('category fusion did not score every row')
    return {'oof_score':out,'fold_weights':fold_weights,'weight_grid':list(grid)}


def fit_category_fusion_full(
    category_score,
    hgb_score,
    target,
    categories,
    *,
    weight_grid=(0.0,0.25,0.5,0.75,1.0),
) -> dict[str,object]:
    grid=_validated_weight_grid(weight_grid)
    a=percentile_rank(np.asarray(category_score,dtype=np.float64)); b=percentile_rank(np.asarray(hgb_score,dtype=np.float64))
    y=np.asarray(target,dtype=np.int8); cats=np.asarray(categories).astype(str)
    if not (len(a)==len(b)==len(y)==len(cats)): raise ValueError('all inputs must align')
    weights=_select_category_weights(a,b,y,cats,np.arange(len(y),dtype=np.int64),weight_grid=grid)
    return {'weights':weights,'weight_grid':list(grid)}


def apply_category_fusion(category_score,hgb_score,categories,weights: dict[str,float]) -> np.ndarray:
    a=percentile_rank(np.asarray(category_score,dtype=np.float64)); b=percentile_rank(np.asarray(hgb_score,dtype=np.float64)); cats=np.asarray(categories).astype(str)
    if not (len(a)==len(b)==len(cats)): raise ValueError('all inputs must align')
    missing=sorted(set(cats.tolist())-set(weights))
    if missing: raise ValueError(f'missing category fusion weights: {missing}')
    w=np.asarray([float(weights[c]) for c in cats],dtype=np.float64)
    if not np.isfinite(w).all() or np.any((w<0)|(w>1)): raise ValueError('invalid category fusion weights')
    return w*a+(1.0-w)*b


__all__=['crossfit_category_fusion','fit_category_fusion_full','apply_category_fusion']
