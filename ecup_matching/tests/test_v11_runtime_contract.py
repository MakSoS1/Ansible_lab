def test_v11_runtime_imports():
    from ecup_matching.submission.predict_v11_no_contrastive import predict_to_csv_v11_no_contrastive
    assert callable(predict_to_csv_v11_no_contrastive)
