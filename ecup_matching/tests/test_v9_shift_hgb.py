from __future__ import annotations

import numpy as np
import pytest

from ecup_matching.ml.v5_meta_blend import SIX_SIGNAL_NAMES


def test_shift_weights_apply_only_frozen_positive_prior_ratio():
    from ecup_matching.ml.v9_shift_hgb import shifted_category_balanced_sample_weight

    y=np.asarray([0,1,0,1,0,0],dtype=np.int8)
    cat=np.asarray(['a','a','b','b','b','b'])
    base=shifted_category_balanced_sample_weight(y,cat,positive_weight=1.0)
    shifted=shifted_category_balanced_sample_weight(y,cat,positive_weight=0.566880890615799)
    assert np.allclose(shifted[y==0],base[y==0])
    assert np.allclose(shifted[y==1],base[y==1]*0.566880890615799)


def test_shift_hgb_requires_binary_target_and_positive_finite_weight():
    from ecup_matching.ml.v9_shift_hgb import shifted_category_balanced_sample_weight

    with pytest.raises(ValueError): shifted_category_balanced_sample_weight(np.asarray([0,2]),np.asarray(['a','a']),positive_weight=.5)
    with pytest.raises(ValueError): shifted_category_balanced_sample_weight(np.asarray([0,1]),np.asarray(['a','a']),positive_weight=0.0)


def test_crossfit_shift_hgb_scores_every_row_without_using_held_labels(monkeypatch):
    import ecup_matching.ml.v9_shift_hgb as module

    scores={name:np.linspace(0,1,20)+(i*.01) for i,name in enumerate(SIX_SIGNAL_NAMES)}
    y=np.asarray(([0,1]*10),dtype=np.int8)
    cat=np.asarray(['a']*10+['b']*10)
    folds=np.asarray([0,1,2,3,4]*4)
    seen=[]
    original=module._fit_shift_model
    def wrapped(design,target,categories,train_indices,**kwargs):
        seen.append(set(train_indices.tolist()))
        return original(design,target,categories,train_indices,**kwargs)
    monkeypatch.setattr(module,'_fit_shift_model',wrapped)
    result=module.crossfit_shift_hgb_stack(scores,y,cat,folds,positive_weight=0.566880890615799,max_iter=5,min_samples_leaf=2,max_leaf_nodes=4,max_depth=2,l2_regularization=1.0)
    assert np.isfinite(result['oof_score']).all()
    for fold,train in zip(sorted(np.unique(folds)),seen,strict=True):
        assert train.isdisjoint(set(np.flatnonzero(folds==fold).tolist()))
    assert result['positive_weight']==pytest.approx(0.566880890615799)
