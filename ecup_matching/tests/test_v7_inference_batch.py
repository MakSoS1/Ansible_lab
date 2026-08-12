import inspect

from ecup_matching.ml import run_v7_outer_oof as outer


def test_outer_oof_exposes_inference_batch_size_and_passes_it_to_predictor():
    signature = inspect.signature(outer.run_v7_outer_oof)
    assert "inference_batch_size" in signature.parameters
    assert signature.parameters["inference_batch_size"].default == 16
    source = inspect.getsource(outer.run_v7_outer_oof)
    assert "batch_size=inference_batch_size" in source


def test_outer_oof_cli_exposes_inference_batch_size_flag():
    source = inspect.getsource(outer.main)
    assert '"--inference-batch-size"' in source
    assert "inference_batch_size=args.inference_batch_size" in source
