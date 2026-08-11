import numpy as np

from ecup_matching.ml.train_v5_teacher_fold import (
    teacher_accumulation_should_step,
    teacher_trainable_layer_indices,
)


def test_teacher_trainable_layer_indices_select_tail_only():
    assert teacher_trainable_layer_indices(12, last_n=4) == [8, 9, 10, 11]
    assert teacher_trainable_layer_indices(6, last_n=2) == [4, 5]
    for layer_count, last_n in ((0, 1), (12, 0), (4, 5)):
        try:
            teacher_trainable_layer_indices(layer_count, last_n=last_n)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid teacher layer request must fail")


def test_teacher_gradient_accumulation_boundaries():
    actual = [
        teacher_accumulation_should_step(i, accumulation_steps=4, is_last_microbatch=False)
        for i in range(1, 9)
    ]
    assert actual == [False, False, False, True, False, False, False, True]
    assert teacher_accumulation_should_step(2, accumulation_steps=4, is_last_microbatch=True)
    try:
        teacher_accumulation_should_step(1, accumulation_steps=0, is_last_microbatch=False)
    except ValueError:
        pass
    else:
        raise AssertionError("zero accumulation must fail")
