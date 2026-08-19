import numpy as np
import pandas as pd
import pytest

from ecup_matching.ml.v8_testlike import (
    build_testlike_slice,
    pseudo_binary_labels,
    pseudo_macro_ap_report,
    soft_rank_report,
)


def test_pseudo_binary_labels_keeps_only_extremes_and_never_calls_soft_midpoints_labels():
    target=np.array([0.0,0.04,0.05,0.2,0.8,0.95,0.96,1.0])
    mask,labels=pseudo_binary_labels(target,low=0.05,high=0.95)
    assert mask.tolist()==[True,True,False,False,False,False,True,True]
    assert labels.tolist()==[0,0,1,1]


def test_testlike_slice_excludes_human_items_and_is_deterministic():
    llm=pd.DataFrame({
        'id1':np.arange(20),
        'id2':np.arange(100,120),
        'target':np.linspace(0,1,20),
    })
    human={1,5,103,118}
    a=build_testlike_slice(llm,human,max_rows=7,seed=2026)
    b=build_testlike_slice(llm,human,max_rows=7,seed=2026)
    pd.testing.assert_frame_equal(a,b)
    assert len(a)==7
    assert not (set(a.id1)|set(a.id2)) & human
    assert a.attrs['diagnostic_only'] is True
    assert a.attrs['selection_seed']==2026


def test_testlike_slice_preserves_complete_anchor_candidate_lists():
    llm=pd.DataFrame({
        'id1':[10,10,10,20,20,30,30,30,30,40],
        'id2':[101,102,103,201,202,301,302,303,304,401],
        'target':[0.01,0.2,0.99,0.02,0.98,0.1,0.3,0.8,0.97,0.5],
    })
    out=build_testlike_slice(llm,set(),max_rows=6,seed=17)
    assert len(out)<=6
    selected=set(out.id1)
    assert selected
    for anchor in selected:
        expected=llm.loc[llm.id1==anchor,['id1','id2','target']].reset_index(drop=True)
        actual=out.loc[out.id1==anchor,['id1','id2','target']].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,expected)
    assert out.attrs['grouping_key']=='id1'
    assert out.attrs['complete_groups'] is True


def test_testlike_slice_refuses_too_small_available_pool():
    llm=pd.DataFrame({'id1':[1,2],'id2':[3,4],'target':[0.0,1.0]})
    with pytest.raises(ValueError,match='available'):
        build_testlike_slice(llm,{1},max_rows=2,seed=1)


def test_pseudo_macro_ap_requires_both_extreme_classes_in_every_category():
    frame=pd.DataFrame({
        'category':['A','A','A','B','B','B'],
        'target':[0.0,1.0,0.5,0.0,1.0,0.4],
    })
    score=np.array([0.1,0.9,0.8,0.2,0.7,0.3])
    report=pseudo_macro_ap_report(frame,score,low=0.05,high=0.95)
    assert report['diagnostic_only'] is True
    assert report['pseudo_label_rows']==4
    assert report['macro_pseudo_average_precision']==pytest.approx(1.0)
    assert report['categories']==2

    bad=frame.copy(); bad.loc[bad.category=='B','target']=0.0
    with pytest.raises(ValueError,match='both pseudo classes'):
        pseudo_macro_ap_report(bad,score,low=0.05,high=0.95)


def test_soft_rank_report_is_monotonic_and_explicitly_diagnostic():
    frame=pd.DataFrame({
        'category':['A']*4+['B']*4,
        'target':[0.0,0.2,0.8,1.0,0.1,0.4,0.7,0.9],
    })
    good=np.asarray(frame.target,float)
    bad=-good
    rg=soft_rank_report(frame,good)
    rb=soft_rank_report(frame,bad)
    assert rg['diagnostic_only'] is True
    assert rg['macro_spearman']>0.99
    assert rb['macro_spearman']<-0.99
