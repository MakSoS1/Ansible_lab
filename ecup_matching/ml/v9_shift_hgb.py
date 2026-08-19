from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .v5_hgb_stack import DEFAULT_HGB_PARAMS, _category_balanced_sample_weight, _design_matrix
from .v5_meta_blend import rank_matrix


def shifted_category_balanced_sample_weight(target, categories, *, positive_weight: float) -> np.ndarray:
    y=np.asarray(target,dtype=np.int8)
    cat=np.asarray(categories).astype(str)
    if y.ndim!=1 or cat.ndim!=1 or len(y)!=len(cat):
        raise ValueError('target/categories must be aligned one-dimensional arrays')
    if set(np.unique(y).tolist())!={0,1}:
        raise ValueError('target must contain exactly binary classes 0 and 1')
    positive_weight=float(positive_weight)
    if not math.isfinite(positive_weight) or positive_weight<=0.0:
        raise ValueError('positive_weight must be finite and positive')
    base=_category_balanced_sample_weight(cat)
    return base*np.where(y==1,positive_weight,1.0)


def _fit_shift_model(
    design: np.ndarray,
    target: np.ndarray,
    categories: np.ndarray,
    train_indices: np.ndarray,
    *,
    positive_weight: float,
    learning_rate: float,
    max_iter: int,
    max_leaf_nodes: int,
    max_depth: int,
    min_samples_leaf: int,
    l2_regularization: float,
    random_state: int,
) -> HistGradientBoostingClassifier:
    y=np.asarray(target,dtype=np.int8)
    cat=np.asarray(categories).astype(str)
    train_indices=np.asarray(train_indices,dtype=np.int64)
    y_train=y[train_indices]
    if set(np.unique(y_train).tolist())!={0,1}:
        raise ValueError('each HGB training partition must contain both classes')
    weight=shifted_category_balanced_sample_weight(y_train,cat[train_indices],positive_weight=positive_weight)
    model=HistGradientBoostingClassifier(
        learning_rate=float(learning_rate),max_iter=int(max_iter),max_leaf_nodes=int(max_leaf_nodes),
        max_depth=int(max_depth),min_samples_leaf=int(min_samples_leaf),l2_regularization=float(l2_regularization),
        early_stopping=False,random_state=int(random_state),
        categorical_features=[False,False,False,False,False,False,True],
    )
    model.fit(design[train_indices],y_train,sample_weight=weight)
    return model


def crossfit_shift_hgb_stack(
    scores: Mapping[str, object], target, categories, folds, *, positive_weight: float,
    learning_rate: float=0.05,max_iter: int=160,max_leaf_nodes: int=15,max_depth: int=3,
    min_samples_leaf: int=200,l2_regularization: float=5.0,random_state: int=20260811,
) -> dict[str,object]:
    ranks=rank_matrix(scores)
    y=np.asarray(target,dtype=np.int8); cat=np.asarray(categories).astype(str); fold_array=np.asarray(folds)
    if not (len(ranks)==len(y)==len(cat)==len(fold_array)):
        raise ValueError('scores, target, categories and folds must align')
    unique_folds=sorted(np.unique(fold_array).tolist())
    if len(unique_folds)<2: raise ValueError('cross-fitting requires at least two folds')
    category_names=tuple(sorted(np.unique(cat).tolist()))
    design=_design_matrix(ranks,cat,category_names)
    all_indices=np.arange(len(y),dtype=np.int64); oof=np.full(len(y),np.nan,dtype=np.float64); models={}
    for fold in unique_folds:
        train=all_indices[fold_array!=fold]; valid=all_indices[fold_array==fold]
        model=_fit_shift_model(
            design,y,cat,train,positive_weight=positive_weight,learning_rate=learning_rate,max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,max_depth=max_depth,min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,random_state=random_state,
        )
        oof[valid]=model.predict_proba(design[valid])[:,1]; models[int(fold)]=model
    if not np.isfinite(oof).all(): raise RuntimeError('shift HGB did not score every row')
    return {'oof_score':oof,'fold_models':models,'category_names':category_names,'rank_matrix':ranks,'design_matrix':design,
            'positive_weight':float(positive_weight),'params':{'learning_rate':float(learning_rate),'max_iter':int(max_iter),
            'max_leaf_nodes':int(max_leaf_nodes),'max_depth':int(max_depth),'min_samples_leaf':int(min_samples_leaf),
            'l2_regularization':float(l2_regularization),'early_stopping':False,'random_state':int(random_state)}}


def fit_shift_hgb_full(scores: Mapping[str,object],target,categories,*,positive_weight: float,**overrides) -> dict[str,object]:
    ranks=rank_matrix(scores); y=np.asarray(target,dtype=np.int8); cat=np.asarray(categories).astype(str)
    category_names=tuple(sorted(np.unique(cat).tolist())); design=_design_matrix(ranks,cat,category_names)
    params=dict(DEFAULT_HGB_PARAMS); params.update(overrides); params.pop('early_stopping',None)
    model=_fit_shift_model(
        design,y,cat,np.arange(len(y),dtype=np.int64),positive_weight=positive_weight,
        learning_rate=params['learning_rate'],max_iter=params['max_iter'],max_leaf_nodes=params['max_leaf_nodes'],
        max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'],l2_regularization=params['l2_regularization'],
        random_state=params['random_state'],
    )
    return {'model':model,'category_names':category_names,'params':{**params,'early_stopping':False},'positive_weight':float(positive_weight)}


__all__=['shifted_category_balanced_sample_weight','crossfit_shift_hgb_stack','fit_shift_hgb_full']
