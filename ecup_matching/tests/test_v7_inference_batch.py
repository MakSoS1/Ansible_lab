from ecup_matching.ml import run_v7_outer_oof_frozen_fastinfer as fast


def test_fastinfer_wrapper_pins_batch_64_without_touching_training_driver():
    assert fast.INFERENCE_BATCH_SIZE == 64


def test_fastinfer_predictor_forces_batch_64_and_preserves_other_arguments():
    calls = []

    def fake_predict(*args, **kwargs):
        calls.append((args, kwargs))
        return "score", "timing"

    original = fast._BASE_PREDICT_PAIRS
    try:
        fast._BASE_PREDICT_PAIRS = fake_predict
        got = fast.predict_pairs_batch64("model", tokenizer="tok", frame="frame", texts="texts", device="cuda", max_length=256, batch_size=16)
    finally:
        fast._BASE_PREDICT_PAIRS = original

    assert got == ("score", "timing")
    assert calls[0][0] == ("model",)
    assert calls[0][1]["batch_size"] == 64
    assert calls[0][1]["max_length"] == 256
    assert calls[0][1]["device"] == "cuda"
