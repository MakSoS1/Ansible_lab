from __future__ import annotations

import numpy as np
import pytest


def test_crossfit_category_fusion_never_uses_held_fold_labels(monkeypatch):
    import ecup_matching.ml.v9_category_fusion as module

    n=100
    a=np.linspace(0,1,n)
    b=1-a
    y=np.asarray(([0,1]*50),dtype=np.int8)
    cats=np.asarray((['a']*50)+(['b']*50))
    folds=np.asarray([0,1,2,3,4]*20)
    seen=[]
    original=module._select_category_weights
    def wrapped(score_a,score_b,target,categories,train_indices,**kwargs):
        seen.append(set(np.asarray(train_indices,dtype=np.int64).tolist()))
        return original(score_a,score_b,target,categories,train_indices,**kwargs)
    monkeypatch.setattr(module,'_select_category_weights',wrapped)
    result=module.crossfit_category_fusion(a,b,y,cats,folds,weight_grid=(0.0,0.25,0.5,0.75,1.0))
    assert np.isfinite(result['oof_score']).all()
    for fold,train in zip(sorted(np.unique(folds)),seen,strict=True):
        assert train.isdisjoint(set(np.flatnonzero(folds==fold).tolist()))


def test_category_fusion_returns_convex_scores_and_complete_weights():
    from ecup_matching.ml.v9_category_fusion import crossfit_category_fusion

    rng=np.random.default_rng(23); n=200
    a=rng.random(n); b=rng.random(n); y=rng.integers(0,2,n,dtype=np.int8)
    cats=np.asarray(['a' if i%2 else 'b' for i in range(n)]); folds=np.asarray([i%5 for i in range(n)])
    result=crossfit_category_fusion(a,b,y,cats,folds,weight_grid=(0.0,0.5,1.0))
    lo=np.minimum(a,b); hi=np.maximum(a,b)
    assert np.all(result['oof_score']>=lo-1e-12)
    assert np.all(result['oof_score']<=hi+1e-12)
    assert set(result['fold_weights'])==set(range(5))
    for weights in result['fold_weights'].values():
        assert set(weights)=={'a','b'}
        assert set(weights.values()) <= {0.0,0.5,1.0}


def test_full_category_fusion_uses_exact_predeclared_grid():
    from ecup_matching.ml.v9_category_fusion import fit_category_fusion_full

    a=np.asarray([.1,.9,.2,.8,.3,.7,.4,.6])
    b=1-a; y=np.asarray([0,1,0,1,0,1,0,1],dtype=np.int8); cats=np.asarray(['a']*4+['b']*4)
    result=fit_category_fusion_full(a,b,y,cats,weight_grid=(0.0,0.5,1.0))
    assert set(result['weights'])=={'a','b'}
    assert set(result['weights'].values()) <= {0.0,0.5,1.0}
    assert result['weight_grid']==[0.0,0.5,1.0]
