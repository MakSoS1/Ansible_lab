import pytest

from ecup_matching.ml.v10_student_contract import (
    V10StudentConfig,
    validate_v10_student_config,
)


def test_v10_student_contract_accepts_short_sequence_without_weakening_v7():
    config = validate_v10_student_config(
        V10StudentConfig(
            max_length=128,
            curriculum_rows=600_000,
            effective_batch_size=64,
            epochs=0.10,
        )
    )
    assert config.max_length == 128


def test_v10_student_contract_keeps_bounded_runtime_sequence():
    with pytest.raises(ValueError, match="within"):
        validate_v10_student_config(
            V10StudentConfig(
                max_length=256,
                curriculum_rows=600_000,
                effective_batch_size=64,
                epochs=0.10,
            )
        )
