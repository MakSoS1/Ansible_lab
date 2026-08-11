import numpy as np

from ecup_matching.ml.train_v5_biencoder_fold import (
    accumulation_should_step,
    contrastive_loss_numpy,
    trainable_layer_indices,
)


def test_margin_contrastive_loss_rewards_high_positive_and_low_negative_cosine():
    good_cosine = np.array([0.95, 0.90, 0.05, -0.10])
    bad_cosine = np.array([0.30, 0.20, 0.80, 0.70])
    target = np.array([1, 1, 0, 0])

    good = contrastive_loss_numpy(good_cosine, target, negative_margin=0.30)
    bad = contrastive_loss_numpy(bad_cosine, target, negative_margin=0.30)
    assert good < bad
    assert good >= 0.0


def test_trainable_layer_indices_select_only_tail_layers():
    assert trainable_layer_indices(12, last_n=4) == [8, 9, 10, 11]
    assert trainable_layer_indices(4, last_n=4) == [0, 1, 2, 3]
    assert trainable_layer_indices(3, last_n=1) == [2]
    for bad in ((0, 1), (4, 0), (4, 5)):
        try:
            trainable_layer_indices(bad[0], last_n=bad[1])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid layer request must fail")


def test_gradient_accumulation_steps_only_on_boundaries_and_final_microbatch():
    assert [
        accumulation_should_step(i, accumulation_steps=4, is_last_microbatch=False)
        for i in range(1, 9)
    ] == [False, False, False, True, False, False, False, True]
    assert accumulation_should_step(3, accumulation_steps=4, is_last_microbatch=True)
    for bad in (0, -1):
        try:
            accumulation_should_step(1, accumulation_steps=bad, is_last_microbatch=False)
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive accumulation_steps must fail")
