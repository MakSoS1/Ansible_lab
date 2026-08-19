import numpy as np
import pandas as pd
import pytest

from ecup_matching.ml.v8_compare import (
    evaluate_grouped_candidates,
    tune_graph_config,
    tune_two_model_blend,
)


def _fixture():
    rows=[]
    for cat in ['A','B']:
        for anchor in range(8):
            for k in range(2):
                target=0.99 if k==0 else 0.01
                rows.append({'id':len(rows),'id1':f'{cat}-{anchor}','id2':f'{cat}-x-{anchor}-{k}','category':cat,'target':target})
    return pd.DataFrame(rows)


def test_two_model_blend_selects_on_tune_and_reports_untouched_confirm():
    frame=_fixture()
    y=(frame.target.to_numpy()>.5).astype(float)
    a=y.copy()
    b=1-y
    report=tune_two_model_blend(frame,{'good':a,'bad':b},seed=2026,weights=np.array([0,.5,1.0]))
    assert report['diagnostic_only'] is True
    assert report['selected_weights']['good']==pytest.approx(1.0)
    assert report['confirm']['macro_pseudo_average_precision']==pytest.approx(1.0)
    assert report['split']['anchor_overlap']==0


def test_graph_tuning_recomputes_features_per_split(monkeypatch):
    frame=_fixture()
    score=np.linspace(0,1,len(frame))
    seen=[]
    import ecup_matching.ml.v8_compare as module
    real=module.graph_features
    def spy(part,s):
        seen.append(set(part.id1))
        return real(part,s)
    monkeypatch.setattr(module,'graph_features',spy)
    report=tune_graph_config(frame,score,seed=2026,configs=[{'rb':0,'rt':0,'ep':0,'ap':0}])
    assert len(seen)==2
    assert seen[0].isdisjoint(seen[1])
    assert report['split']['anchor_overlap']==0


def test_grouped_evaluator_refuses_misaligned_or_nonfinite_scores():
    frame=_fixture()
    with pytest.raises(ValueError,match='aligned'):
        evaluate_grouped_candidates(frame,{'a':np.ones(len(frame)-1)})
    bad=np.ones(len(frame)); bad[0]=np.nan
    with pytest.raises(ValueError,match='finite'):
        evaluate_grouped_candidates(frame,{'a':bad})


def test_grouped_evaluator_reports_base_pseudo_ap_and_soft_rank_for_each_model():
    frame=_fixture()
    good=frame.target.to_numpy(float)
    bad=-good
    report=evaluate_grouped_candidates(frame,{'good':good,'bad':bad})
    assert report['diagnostic_only'] is True
    assert report['models']['good']['pseudo']['macro_pseudo_average_precision']==pytest.approx(1.0)
    assert report['models']['good']['soft']['macro_spearman']>0.99
    assert report['models']['bad']['soft']['macro_spearman']<-0.99
