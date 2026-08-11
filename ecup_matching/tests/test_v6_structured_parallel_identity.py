"""The parallel structured phase must reproduce the serial phase exactly.

These tests drive the real ``_structured_scores_streaming`` with real feature
builders and real fitted estimators, so they cover the actual production path
rather than a stand-in.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ecup_matching.ml import features as ml_features
from ecup_matching.ml import features_v2 as ml_features_v2
from ecup_matching.ml import v5_sparse as ml_sparse
from ecup_matching.ml.features_v2 import FEATURE_NAMES_V2, build_features_v2_chunked
from ecup_matching.ml.v5_sparse import SPARSE_PAIR_FEATURE_NAMES, fit_sparse_item_encoder
from ecup_matching.submission.predict_v6 import _structured_scores_streaming


CATEGORIES = ("cat_0", "cat_1", "cat_2")
BASE_COLUMNS = tuple(name for name in FEATURE_NAMES_V2 if name != "category")
EXPLICIT_KEYS = ("бренд", "вес")


def _items(count: int) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    nouns = ["смартфон", "наушники беспроводные tws", "чехол силиконовый", "кабель type-c"]
    brands = ["samsung", "xiaomi", "bosch", "philips"]
    return pd.DataFrame(
        {
            "id": np.arange(count, dtype=np.int64),
            "name": [
                f"{nouns[i % len(nouns)]} {brands[i % len(brands)]} "
                f"{int(rng.integers(100, 9999))} гб {int(rng.integers(1, 40))} шт "
                f"модель a-{i % 97}"
                for i in range(count)
            ],
            "attributes": [
                json.dumps(
                    {
                        "бренд": [brands[i % len(brands)]],
                        "вес": [f"{i % 50} г"],
                        "цвет": ["черный" if i % 2 else "белый"],
                    },
                    ensure_ascii=False,
                )
                for i in range(count)
            ],
            "category": [CATEGORIES[i % len(CATEGORIES)] for i in range(count)],
        }
    )


def _pairs(items: pd.DataFrame, count: int) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    ids = items["id"].to_numpy()
    frame = pd.DataFrame(
        {
            "id1": rng.choice(ids, count),
            "id2": rng.choice(ids, count),
        }
    )
    category_by_id = items.set_index("id")["category"].astype(str)
    frame["category"] = frame["id1"].map(category_by_id)
    return frame


def _fit_specialists(feature_columns, width: int):
    rng = np.random.default_rng(31)
    models = {}
    for category in CATEGORIES:
        x = rng.normal(size=(64, width)).astype(np.float32)
        y = (rng.random(64) > 0.5).astype(int)
        y[0], y[1] = 0, 1
        models[category] = LogisticRegression(max_iter=200).fit(x, y)
    return {"feature_columns": list(feature_columns), "models": models}


def _fit_explicit(width: int):
    rng = np.random.default_rng(37)
    models = {}
    for category in CATEGORIES:
        x = rng.normal(size=(64, width)).astype(np.float32)
        y = (rng.random(64) > 0.5).astype(int)
        y[0], y[1] = 0, 1
        models[category] = LogisticRegression(max_iter=200).fit(x, y)
    return {
        "models": models,
        "key_spec": {category: list(EXPLICIT_KEYS) for category in CATEGORIES},
    }


def _structured_bundle(items: pd.DataFrame, pairs: pd.DataFrame):
    encoder = fit_sparse_item_encoder(
        items, max_char_features=512, max_word_features=256
    )
    probe = build_features_v2_chunked(
        items, pairs.head(4), attribute_importance=None, chunk_size=4
    )
    base_width = len([c for c in probe.columns if c != "category"])
    sparse_columns = (*BASE_COLUMNS, *SPARSE_PAIR_FEATURE_NAMES)
    return {
        "weak": _fit_specialists(BASE_COLUMNS, base_width),
        "sparse": {
            "encoder": encoder,
            "specialists": _fit_specialists(sparse_columns, base_width + len(SPARSE_PAIR_FEATURE_NAMES)),
        },
        "explicit": _fit_explicit(base_width + 3 * len(EXPLICIT_KEYS)),
        "typed_explicit": _fit_explicit(base_width + 3 * len(EXPLICIT_KEYS)),
    }


def _run(items, pairs, structured, *, chunk_size, workers):
    return _structured_scores_streaming(
        items=items,
        pairs=pairs,
        structured=structured,
        legacy_features=ml_features,
        legacy_features_v2=ml_features_v2,
        legacy_sparse=ml_sparse,
        chunk_size=chunk_size,
        workers=workers,
    )


SIGNALS = ("weak", "sparse", "explicit", "typed_explicit")


def test_parallel_structured_phase_is_bitwise_identical_to_serial():
    items = _items(160)
    pairs = _pairs(items, 600)
    structured = _structured_bundle(items, pairs)

    serial = _run(items, pairs, structured, chunk_size=100, workers=1)
    parallel = _run(items, pairs, structured, chunk_size=100, workers=4)

    for name in SIGNALS:
        assert np.array_equal(serial[name], parallel[name]), f"{name} diverged"
        assert np.isfinite(serial[name]).all()


def test_structured_scores_are_independent_of_worker_count():
    items = _items(120)
    pairs = _pairs(items, 420)
    structured = _structured_bundle(items, pairs)

    reference = _run(items, pairs, structured, chunk_size=70, workers=1)
    for workers in (2, 3, 8):
        candidate = _run(items, pairs, structured, chunk_size=70, workers=workers)
        for name in SIGNALS:
            assert np.array_equal(reference[name], candidate[name]), (
                f"{name} changed with workers={workers}"
            )


def test_structured_scores_preserve_input_row_order():
    """Each pair must land in its own input slot, whoever computed it."""
    items = _items(90)
    pairs = _pairs(items, 210)
    structured = _structured_bundle(items, pairs)

    full = _run(items, pairs, structured, chunk_size=50, workers=4)

    # A contiguous slice aligned to a chunk boundary is scored with exactly the
    # same chunk composition, so its values must match the full run byte for byte.
    window = pairs.iloc[50:100].reset_index(drop=True)
    sliced = _run(items, window, structured, chunk_size=50, workers=1)
    for name in SIGNALS:
        assert np.array_equal(sliced[name], full[name][50:100]), f"{name} is misaligned"


def test_changing_chunk_size_perturbs_scores_only_at_float32_epsilon():
    """Chunk size is not a free parameter: it changes sklearn batch shapes.

    ``predict_proba`` runs float32 GEMM whose accumulation order depends on the
    number of rows in the call, so a different chunk size shifts scores by a few
    ULPs. Harmless for ranking, but it means the production constant must stay
    pinned for byte-reproducible packaging.
    """
    items = _items(120)
    pairs = _pairs(items, 300)
    structured = _structured_bundle(items, pairs)

    reference = _run(items, pairs, structured, chunk_size=300, workers=1)
    rechunked = _run(items, pairs, structured, chunk_size=64, workers=1)

    for name in SIGNALS:
        delta = np.abs(reference[name] - rechunked[name]).max()
        assert delta < 1e-5, f"{name} moved by {delta}, far beyond float32 noise"


def test_production_structured_chunk_size_is_pinned():
    from ecup_matching.submission import predict_v6

    assert predict_v6.STRUCTURED_CHUNK_SIZE == 10_000, (
        "changing the chunk size perturbs float32 sklearn batches; it must be "
        "revalidated against the strict OOF gate before being changed"
    )
