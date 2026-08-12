import pandas as pd

from ecup_matching.ml.v7_hardneg import (
    HardNegativeMacroPairBatchSampler,
    attach_target_free_pair_hardness,
)
from ecup_matching.ml.v7_neural import MacroPairBatchSampler


def test_target_free_hardness_ranks_near_duplicate_negative_above_easy_negative():
    frame = pd.DataFrame(
        {
            "id1": [1, 1],
            "id2": [2, 3],
            "category": ["phones", "phones"],
            "target": [0, 0],
        }
    )
    texts = {
        1: "[CAT] phones\n[NAME] samsung galaxy s24 128gb black\n[MODEL] sms921b",
        2: "[CAT] phones\n[NAME] samsung galaxy s24 256gb black\n[MODEL] sms921b",
        3: "[CAT] phones\n[NAME] apple iphone 13 red\n[MODEL] a2633",
    }
    got = attach_target_free_pair_hardness(frame, texts)
    assert got.loc[0, "negative_hardness"] > got.loc[1, "negative_hardness"]
    assert got["target"].tolist() == [0, 0]


def test_hard_negative_sampler_draws_from_high_hardness_pool_when_fraction_is_one():
    frame = pd.DataFrame(
        {
            "category": ["x"] * 8,
            "target": [1, 1, 1, 1, 0, 0, 0, 0],
            "negative_hardness": [0, 0, 0, 0, 0.1, 0.2, 0.9, 1.0],
        }
    )
    sampler = HardNegativeMacroPairBatchSampler(
        frame,
        batch_size=4,
        seed=17,
        hard_negative_fraction=1.0,
    )
    for batch in sampler:
        rows = frame.iloc[batch]
        negatives = rows[rows.target == 0]
        assert len(negatives) == 2
        assert set(negatives.index).issubset({6, 7})


def test_base_macro_sampler_remains_backward_compatible():
    frame = pd.DataFrame(
        {
            "category": ["x"] * 4,
            "target": [1, 1, 0, 0],
        }
    )
    batches = list(MacroPairBatchSampler(frame, batch_size=2, seed=1))
    assert batches
    for batch in batches:
        assert set(frame.iloc[batch].target.astype(int)) == {0, 1}
