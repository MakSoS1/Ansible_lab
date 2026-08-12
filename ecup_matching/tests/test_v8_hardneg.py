import numpy as np
import pandas as pd
import pytest

from ecup_matching.ml.v8_hardneg import (
    HardNegativeMacroPairBatchSamplerV8,
    attach_v8_hardness,
    pair_hardness_v8,
)


def test_near_duplicate_with_conflicting_model_is_harder_than_unrelated_pair():
    a='[CAT] Электроника [NAME] iphone 15 pro 128 gb [MODEL] a3101 [NUMERIC] 128 gb'
    b='[CAT] Электроника [NAME] iphone 15 pro 256 gb [MODEL] a3102 [NUMERIC] 256 gb'
    c='[CAT] Электроника [NAME] электрический чайник 1.7 л'
    hard=pair_hardness_v8(a,b)
    easy=pair_hardness_v8(a,c)
    assert hard>easy+0.30
    assert hard>0.7


def test_same_model_exact_match_is_not_mistaken_for_model_conflict():
    a='phone samsung s24 sm-s921b 128 gb'
    b='samsung galaxy s24 sm-s921b 128gb black'
    c='samsung galaxy s24 sm-s926b 256gb black'
    same=pair_hardness_v8(a,b)
    conflict=pair_hardness_v8(a,c)
    assert conflict>same


def test_attach_hardness_is_target_free_for_identical_pairs_and_texts():
    frame=pd.DataFrame({
        'id1':[1,1,2], 'id2':[2,3,3],
        'target':[0,1,0], 'category':['A']*3,
    })
    texts={1:'alpha x100 10 ml',2:'alpha x101 20 ml',3:'beta q9'}
    base=attach_v8_hardness(frame,texts)
    changed=frame.copy(); changed['target']=1-changed['target']
    other=attach_v8_hardness(changed,texts)
    np.testing.assert_allclose(base['negative_hardness'],other['negative_hardness'])


def test_sampler_draws_hard_negatives_more_often_but_keeps_positive_negative_mix():
    rows=[]
    for i in range(8):
        rows.append({'category':'A','target':1,'negative_hardness':0.0,'tag':f'p{i}'})
    for i,h in enumerate([.99,.95,.9,.85,.2,.15,.1,.05]):
        rows.append({'category':'A','target':0,'negative_hardness':h,'tag':f'n{i}'})
    frame=pd.DataFrame(rows)
    sampler=HardNegativeMacroPairBatchSamplerV8(frame,batch_size=8,seed=7,hard_negative_fraction=.75,hard_pool_fraction=.5)
    counts={i:0 for i in range(len(frame))}
    for epoch in range(50):
        sampler.epoch=epoch
        for batch in sampler:
            targets=frame.iloc[batch].target.to_numpy()
            assert (targets==1).any() and (targets==0).any()
            for i in batch: counts[i]+=1
    hard=sum(counts[i] for i in range(8,12))
    easy=sum(counts[i] for i in range(12,16))
    assert hard>easy*2.0


def test_sampler_is_deterministic_for_fixed_seed_and_epoch():
    frame=pd.DataFrame({
        'category':['A']*8,
        'target':[1,1,1,1,0,0,0,0],
        'negative_hardness':[0,0,0,0,.9,.8,.2,.1],
    })
    a=HardNegativeMacroPairBatchSamplerV8(frame,4,123,epoch=2)
    b=HardNegativeMacroPairBatchSamplerV8(frame,4,123,epoch=2)
    assert list(a)==list(b)
