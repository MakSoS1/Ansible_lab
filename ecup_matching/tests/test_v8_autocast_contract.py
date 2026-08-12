from __future__ import annotations

import inspect

from ecup_matching.submission import predict_v6


def test_contrastive_and_teacher_use_declared_cuda_autocast_policy():
    source = inspect.getsource(predict_v6)
    assert 'torch_autocast' in source
    contrastive = inspect.getsource(predict_v6._contrastive_scores_fast)
    teacher = inspect.getsource(predict_v6._teacher_selected_scores_fast)
    assert 'with torch_autocast(torch, config):' in contrastive
    assert 'with torch_autocast(torch, config):' in teacher
    assert 'hidden = model(**tokens).last_hidden_state' in contrastive
    assert 'torch.sigmoid(model(**tokens).logits.squeeze(-1))' in teacher
