from __future__ import annotations

import json
import multiprocessing
import sys

import numpy as np
import pandas as pd
import pytest

from ecup_matching.ml import features, features_v2
from ecup_matching.ml.features_v2 import build_pair_features_v2
from ecup_matching.submission.v6_fast import collect_chunked_scores
from ecup_matching.submission.v6_parallel import (
    parallel_supported,
    resolve_worker_count,
    run_structured_chunks,
    shared_fuzzy_ratios,
)


def _items(count: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    nouns = ["смартфон", "наушники беспроводные", "чехол силиконовый", "кабель type-c"]
    brands = ["samsung", "xiaomi", "bosch", "philips"]
    return pd.DataFrame(
        {
            "id": np.arange(count, dtype=np.int64),
            "name": [
                f"{nouns[i % len(nouns)]} {brands[i % len(brands)]} "
                f"{int(rng.integers(100, 9999))} гб {int(rng.integers(1, 40))} шт"
                for i in range(count)
            ],
            "attributes": [
                json.dumps(
                    {"бренд": [brands[i % len(brands)]], "вес": [f"{i % 50} г"]},
                    ensure_ascii=False,
                )
                for i in range(count)
            ],
            "category": [f"cat_{i % 4}" for i in range(count)],
        }
    )


def _pairs(item_count: int, count: int) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "id1": rng.integers(0, item_count, count).astype(np.int64),
            "id2": rng.integers(0, item_count, count).astype(np.int64),
        }
    )


# --- shared_fuzzy_ratios ----------------------------------------------------


def test_shared_fuzzy_ratios_preserves_feature_values():
    items = _items(60)
    pairs = _pairs(60, 120)
    baseline = build_pair_features_v2(items, pairs)
    with shared_fuzzy_ratios():
        memoized = build_pair_features_v2(items, pairs)
    pd.testing.assert_frame_equal(baseline, memoized)


def test_shared_fuzzy_ratios_restores_originals_even_on_error():
    original_ratio = features._ratio
    original_partial = features_v2._partial_ratio
    with pytest.raises(RuntimeError):
        with shared_fuzzy_ratios():
            assert features._ratio is not original_ratio
            raise RuntimeError("boom")
    assert features._ratio is original_ratio
    assert features_v2._partial_ratio is original_partial


def test_shared_fuzzy_ratios_actually_reuses_repeated_calls():
    items = _items(40)
    pairs = _pairs(40, 80)
    with shared_fuzzy_ratios() as cache:
        build_pair_features_v2(items, pairs)
        first = len(cache)
        assert first > 0
        build_pair_features_v2(items, pairs)
        assert len(cache) == first, "second identical pass must be served from cache"


def test_divergent_implementations_do_not_share_cache_entries():
    """A future legacy pin that changes _ratio must not read the typed value."""
    module = type(sys)("legacy_ecup.ml.features")

    def _ratio(a, b):  # noqa: D401 - deliberately different bytecode and result
        return 0.125

    def _partial_ratio(a, b):
        return 0.25

    module._ratio = _ratio
    module._partial_ratio = _partial_ratio
    sys.modules["legacy_ecup.ml.features"] = module
    try:
        with shared_fuzzy_ratios():
            assert module._ratio("abc", "abd") == 0.125
            assert features._ratio("abc", "abd") == pytest.approx(
                _ratio_reference("abc", "abd")
            )
    finally:
        del sys.modules["legacy_ecup.ml.features"]


def _ratio_reference(a, b):
    from difflib import SequenceMatcher

    return float(SequenceMatcher(None, a, b, autojunk=False).ratio())


# --- run_structured_chunks --------------------------------------------------


def _reference_score_chunk(pairs: pd.DataFrame):
    def score_chunk(start: int, end: int):
        chunk = pairs.iloc[start:end]
        left = chunk["id1"].to_numpy(dtype=np.float64)
        right = chunk["id2"].to_numpy(dtype=np.float64)
        return {"a": left + 0.5 * right, "b": np.sqrt(left * right + 1.0)}

    return score_chunk


def test_parallel_matches_serial_bitwise():
    pairs = _pairs(500, 4321)
    score_chunk = _reference_score_chunk(pairs)
    serial = collect_chunked_scores(
        row_count=len(pairs), chunk_size=500, signal_names=("a", "b"), score_chunk=score_chunk
    )
    parallel = run_structured_chunks(
        row_count=len(pairs),
        chunk_size=500,
        signal_names=("a", "b"),
        score_chunk=score_chunk,
        workers=4,
    )
    for name in ("a", "b"):
        assert np.array_equal(serial[name], parallel[name])


def test_single_worker_falls_back_to_serial_path():
    pairs = _pairs(200, 700)
    score_chunk = _reference_score_chunk(pairs)
    serial = collect_chunked_scores(
        row_count=len(pairs), chunk_size=100, signal_names=("a", "b"), score_chunk=score_chunk
    )
    single = run_structured_chunks(
        row_count=len(pairs),
        chunk_size=100,
        signal_names=("a", "b"),
        score_chunk=score_chunk,
        workers=1,
    )
    for name in ("a", "b"):
        assert np.array_equal(serial[name], single[name])


def test_progress_reports_monotonic_completion():
    pairs = _pairs(100, 950)
    seen: list[tuple[int, int]] = []
    run_structured_chunks(
        row_count=len(pairs),
        chunk_size=100,
        signal_names=("a", "b"),
        score_chunk=_reference_score_chunk(pairs),
        workers=3,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen[-1] == (len(pairs), len(pairs))
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)


def test_empty_input_returns_empty_signals():
    out = run_structured_chunks(
        row_count=0,
        chunk_size=10,
        signal_names=("a",),
        score_chunk=lambda start, end: np.empty(0),
        workers=4,
    )
    assert out["a"].shape == (0,)


def test_worker_count_never_exceeds_cap_and_stays_positive():
    assert resolve_worker_count(0) == 1
    assert resolve_worker_count(1000, max_workers_cap=20) == 20
    assert resolve_worker_count(None, cpu_count=20) == 19
    assert resolve_worker_count(None, cpu_count=1) == 1
    assert resolve_worker_count(None, cpu_count=2) == 1


def test_parallel_support_matches_fork_availability(monkeypatch):
    monkeypatch.delenv("ECUP_STRUCTURED_FORCE_SERIAL", raising=False)
    assert parallel_supported() == ("fork" in multiprocessing.get_all_start_methods())
    monkeypatch.setenv("ECUP_STRUCTURED_FORCE_SERIAL", "1")
    assert parallel_supported() is False


def test_pool_failure_degrades_to_serial_instead_of_failing_the_run(monkeypatch):
    """A container that forbids extra processes must still produce a submission."""
    pairs = _pairs(200, 900)
    score_chunk = _reference_score_chunk(pairs)
    expected = collect_chunked_scores(
        row_count=len(pairs), chunk_size=100, signal_names=("a", "b"), score_chunk=score_chunk
    )

    real_context = multiprocessing.get_context

    class RefusingContext:
        def Pool(self, *args, **kwargs):
            raise OSError("fork not permitted")

    monkeypatch.setattr(
        multiprocessing,
        "get_context",
        lambda method=None: RefusingContext() if method == "fork" else real_context(method),
    )

    out = run_structured_chunks(
        row_count=len(pairs),
        chunk_size=100,
        signal_names=("a", "b"),
        score_chunk=score_chunk,
        workers=4,
    )
    for name in ("a", "b"):
        assert np.array_equal(expected[name], out[name])
