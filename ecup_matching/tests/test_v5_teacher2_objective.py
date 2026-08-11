import numpy as np

from ecup_matching.ml.v5_teacher2_objective import (
    pairwise_category_ranking_loss_numpy,
    source_loss_weights,
)


def test_pairwise_category_ranking_loss_rewards_correct_within_category_order():
    category=np.array(['a','a','a','b','b','b'])
    target=np.array([1,0,0,1,0,0])
    good=np.array([3.0,0.0,-1.0,2.0,0.5,-0.5])
    bad=np.array([-1.0,2.0,1.0,-2.0,1.0,0.5])
    assert pairwise_category_ranking_loss_numpy(good,target,category) < pairwise_category_ranking_loss_numpy(bad,target,category)


def test_source_weights_keep_human_dominant_and_weak_confidence_monotone():
    source=np.array(['human','weak','weak','weak'])
    confidence=np.array([1.0,1.0,0.6,0.3])
    w=source_loss_weights(source,confidence,human_weight=1.0,weak_scale=0.1)
    assert w.tolist()==[1.0,0.1,0.06,0.03]
    assert w[0] > w[1] > w[2] > w[3]
