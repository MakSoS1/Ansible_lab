from __future__ import annotations

from . import run_v7_outer_oof as base
from . import run_v7_outer_oof_frozen as frozen


INFERENCE_BATCH_SIZE = 64
_BASE_PREDICT_PAIRS = base.predict_pairs


def predict_pairs_batch64(*args, **kwargs):
    kwargs["batch_size"] = INFERENCE_BATCH_SIZE
    return _BASE_PREDICT_PAIRS(*args, **kwargs)


def main() -> int:
    original_predict = base.predict_pairs
    try:
        base.predict_pairs = predict_pairs_batch64
        return frozen.main()
    finally:
        base.predict_pairs = original_predict


if __name__ == "__main__":
    raise SystemExit(main())
