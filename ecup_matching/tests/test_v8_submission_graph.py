import numpy as np
import pandas as pd
import pytest

from ecup_matching.ml.v8_submission_graph import apply_graph_to_prediction


def test_graph_postprocess_preserves_id_order_and_float64_ranking_precision():
    test=pd.DataFrame({'id':[7,8,9],'id1':[1,1,2],'id2':[2,3,3]})
    items=pd.DataFrame({'id':[1,2,3],'category':['A','A','A']})
    pred=pd.DataFrame({'id':[7,8,9],'predict':np.array([.9,.8,.7],dtype=np.float64)})
    out=apply_graph_to_prediction(test,items,pred,{'rb':0,'rt':0,'ep':.02,'ap':.01})
    assert out.id.tolist()==[7,8,9]
    assert out.predict.dtype==np.float64
    assert np.isfinite(out.predict).all()
    assert len(out)==len(test)


def test_graph_postprocess_rejects_cross_category_pair():
    test=pd.DataFrame({'id':[1],'id1':[10],'id2':[20]})
    items=pd.DataFrame({'id':[10,20],'category':['A','B']})
    pred=pd.DataFrame({'id':[1],'predict':[.5]})
    with pytest.raises(ValueError,match='cross-category'):
        apply_graph_to_prediction(test,items,pred,{'rb':0,'rt':0,'ep':0,'ap':0})


def test_graph_postprocess_rejects_prediction_id_reordering():
    test=pd.DataFrame({'id':[1,2],'id1':[10,11],'id2':[11,10]})
    items=pd.DataFrame({'id':[10,11],'category':['A','A']})
    pred=pd.DataFrame({'id':[2,1],'predict':[.5,.6]})
    with pytest.raises(ValueError,match='id order'):
        apply_graph_to_prediction(test,items,pred,{'rb':0,'rt':0,'ep':0,'ap':0})


def test_graph_postprocess_rejects_missing_item_category():
    test=pd.DataFrame({'id':[1],'id1':[10],'id2':[20]})
    items=pd.DataFrame({'id':[10],'category':['A']})
    pred=pd.DataFrame({'id':[1],'predict':[.5]})
    with pytest.raises(ValueError,match='missing category'):
        apply_graph_to_prediction(test,items,pred,{'rb':0,'rt':0,'ep':0,'ap':0})
