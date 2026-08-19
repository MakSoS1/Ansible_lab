import numpy as np
import pandas as pd
import pytest

from ecup_matching.ml.v8_selection import anchor_disjoint_split, category_rank_percentile, rank_blend


def test_anchor_split_is_deterministic_disjoint_and_category_stratified():
    rows=[]
    for cat in ['A','B']:
        for anchor in range(10):
            for k in range(1+(anchor%3)):
                rows.append({'id1':f'{cat}-{anchor}','id2':f'{cat}-x-{anchor}-{k}','category':cat})
    frame=pd.DataFrame(rows)
    tune1,confirm1,meta1=anchor_disjoint_split(frame,seed=2026,tune_fraction=.5)
    tune2,confirm2,meta2=anchor_disjoint_split(frame,seed=2026,tune_fraction=.5)
    assert np.array_equal(tune1,tune2) and np.array_equal(confirm1,confirm2)
    assert not np.any(tune1 & confirm1)
    assert np.all(tune1 | confirm1)
    assert set(frame.loc[tune1,'id1']).isdisjoint(set(frame.loc[confirm1,'id1']))
    for cat in ['A','B']:
        assert frame.loc[tune1 & (frame.category==cat),'id1'].nunique()==5
        assert frame.loc[confirm1 & (frame.category==cat),'id1'].nunique()==5
    assert meta1==meta2
    assert meta1['anchor_overlap']==0


def test_anchor_split_rejects_anchor_in_multiple_categories():
    frame=pd.DataFrame({'id1':[1,1,2,3],'id2':[4,5,6,7],'category':['A','B','A','B']})
    with pytest.raises(ValueError,match='multiple categories'):
        anchor_disjoint_split(frame)


def test_category_rank_percentile_is_monotonic_within_category():
    frame=pd.DataFrame({'category':['A']*3+['B']*3})
    score=np.array([.2,.9,.5,10,5,0],float)
    rank=category_rank_percentile(frame,score)
    assert rank[1]>rank[2]>rank[0]
    assert rank[3]>rank[4]>rank[5]
    assert np.all((0<=rank)&(rank<=1))


def test_rank_blend_is_scale_invariant_per_model_and_category():
    frame=pd.DataFrame({'category':['A']*4+['B']*4})
    a=np.array([1,2,3,4,20,10,40,30],float)
    b=np.array([4,3,2,1,1,2,4,3],float)
    x=rank_blend(frame,{'a':a,'b':b},{'a':.25,'b':.75})
    y=rank_blend(frame,{'a':100*a+7,'b':np.exp(b)},{'a':.25,'b':.75})
    np.testing.assert_allclose(x,y)


def test_rank_blend_requires_weights_sum_to_one():
    frame=pd.DataFrame({'category':['A','A']})
    with pytest.raises(ValueError,match='sum'):
        rank_blend(frame,{'a':[1,2],'b':[2,1]},{'a':.2,'b':.2})
