from ecup_matching.ml.run_v10_production_frozen import normalize_v10_production_payload


def test_v10_production_payload_is_not_reported_as_validation():
    payload = normalize_v10_production_payload(
        {
            "version": "v7-production-refit",
            "candidate": "legacy-name",
            "validation_metric_reported": False,
            "base_model": "legacy-model",
            "gold_metric_opened": False,
            "gold_rows_scored": 0,
            "max_length": 128,
            "max_chars": 650,
            "development_rows": 285210,
            "training_rows": 285210,
            "sealed_gold_rows": 80444,
        },
        base_model_revision="e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae",
        inference_batch_size=512,
    )
    assert payload["version"] == "v10-production-refit"
    assert payload["base_model"] == "cointegrated/rubert-tiny2"
    assert payload["runtime_architecture"] == "single-small-cross-encoder"
    assert payload["validation_metric_reported"] is False
    assert payload["gold_metric_opened"] is False
    assert payload["gold_rows_scored"] == 0
    assert payload["inference_batch_size"] == 512
