import pytest

from ecup_matching.submission.predict_v10 import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_LENGTH,
    validate_v10_metadata,
)


def test_v10_runtime_defaults_are_deliberately_small():
    assert DEFAULT_MAX_LENGTH <= 160
    assert DEFAULT_MAX_CHARS <= 700
    assert DEFAULT_BATCH_SIZE >= 64


def test_v10_metadata_requires_tiny_student_and_sealed_gold():
    payload = validate_v10_metadata(
        {
            "version": "v10-tiny-student",
            "base_model": "cointegrated/rubert-tiny2",
            "strict_oof_macro_average_precision": 0.5,
            "gold_metric_opened": False,
            "gold_rows_scored": 0,
            "max_length": 128,
            "max_chars": 650,
            "inference_batch_size": 128,
        }
    )
    assert payload["base_model"] == "cointegrated/rubert-tiny2"

    bad = dict(payload)
    bad["base_model"] = "ai-forever/ruBert-base"
    with pytest.raises(ValueError, match="tiny"):
        validate_v10_metadata(bad)

    bad = dict(payload)
    bad["gold_metric_opened"] = True
    with pytest.raises(ValueError, match="sealed gold"):
        validate_v10_metadata(bad)


def test_v10_metadata_rejects_slow_sequence_contract():
    with pytest.raises(ValueError, match="max_length"):
        validate_v10_metadata(
            {
                "version": "v10-tiny-student",
                "base_model": "cointegrated/rubert-tiny2",
                "strict_oof_macro_average_precision": 0.5,
                "gold_metric_opened": False,
                "gold_rows_scored": 0,
                "max_length": 256,
                "max_chars": 650,
                "inference_batch_size": 128,
            }
        )
