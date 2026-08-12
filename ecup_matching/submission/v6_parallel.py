"""Runtime-only accelerators for the v6 structured phase.

Two independent, prediction-preserving optimizations live here.

``shared_fuzzy_ratios`` removes duplicated ``difflib`` work. The structured
phase scores every pair twice: once through the pinned ``legacy_ecup`` runtime
and once through the typed/canonicalized modules. ``features.py`` and
``features_v2.py`` are byte-identical between the pinned legacy commit and
HEAD, and ``textnorm`` differs only in ``extract_model_codes`` and
``extract_quantities``. Neither affects ``ItemNorm.name`` or
``ItemNorm.name_tokens``, so both passes call ``_ratio``/``_partial_ratio``
with exactly the same strings and recompute all eight SequenceMatcher results
per pair. Memoizing those calls returns the identical floats the original code
would have produced; it only avoids computing them twice.

``run_structured_chunks`` distributes the already-independent pair chunks over
worker processes. ``collect_chunked_scores`` reassembles by global row offset,
each chunk is a pure function of its own pairs, and the structured models are
inherited read-only through ``fork``, so the parallel result is bit-identical
to the serial result.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import sys
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from .v6_fast import batch_index_ranges, collect_chunked_scores


_FUZZY_MODULE_NAMES = (
    "legacy_ecup.ml.features",
    "legacy_ecup.ml.features_v2",
    "ecup_matching.ml.features",
    "ecup_matching.ml.features_v2",
)
_FUZZY_FUNCTION_NAMES = ("_ratio", "_partial_ratio")


_MISS = object()


def _implementation_tag(function, registry: dict) -> int:
    """Give byte-identical implementations one tag and divergent ones separate tags.

    The legacy and typed passes currently share the exact same ``features``
    source, so their results are interchangeable. If a future legacy pin ever
    diverges, the tags stop matching and the two passes simply stop sharing
    cache entries instead of returning each other's values.
    """
    code = getattr(function, "__code__", None)
    if code is None:
        return -id(function)
    try:
        key = (code.co_code, code.co_consts, code.co_names, code.co_argcount)
        hash(key)
    except TypeError:
        return -id(function)
    if key not in registry:
        registry[key] = len(registry)
    return registry[key]


def _memoized(function, cache: dict, tag: int):
    def wrapper(a, b):
        key = (tag, a, b)
        hit = cache.get(key, _MISS)
        if hit is _MISS:
            hit = function(a, b)
            cache[key] = hit
        return hit

    wrapper.__wrapped__ = function
    return wrapper


@contextlib.contextmanager
def shared_fuzzy_ratios(cache: dict | None = None):
    """Share ``_ratio``/``_partial_ratio`` results across the legacy and typed passes.

    Rebinding happens in every module namespace that resolves those names, so a
    single cache backs all of them. Original functions are always restored.
    """
    store = {} if cache is None else cache
    tags: dict = {}
    originals: list[tuple[object, str, object]] = []
    try:
        for module_name in _FUZZY_MODULE_NAMES:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            for function_name in _FUZZY_FUNCTION_NAMES:
                original = getattr(module, function_name, None)
                if original is None or hasattr(original, "__wrapped__"):
                    continue
                originals.append((module, function_name, original))
                setattr(
                    module,
                    function_name,
                    _memoized(original, store, _implementation_tag(original, tags)),
                )
        yield store
    finally:
        for module, function_name, original in originals:
            setattr(module, function_name, original)
        store.clear()


def resolve_worker_count(
    requested: int | None = None,
    *,
    cpu_count: int | None = None,
    max_workers_cap: int = 32,
) -> int:
    """Pick a worker count from the environment, honouring an explicit override."""
    if requested is not None:
        return max(1, min(int(requested), max_workers_cap))
    override = os.environ.get("ECUP_STRUCTURED_WORKERS", "").strip()
    if override:
        try:
            return max(1, min(int(override), max_workers_cap))
        except ValueError:
            pass
    detected = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    # Leave one core for the parent process and the allocator.
    return max(1, min(int(detected) - 1, max_workers_cap)) if detected > 2 else 1


def parallel_supported() -> bool:
    """Parallel chunking requires fork so workers inherit loaded models for free."""
    if os.environ.get("ECUP_STRUCTURED_FORCE_SERIAL", "").strip() == "1":
        return False
    return "fork" in multiprocessing.get_all_start_methods()


_WORKER_SCORE_CHUNK: Callable[[int, int], object] | None = None
_WORKER_SIGNAL_NAMES: tuple[str, ...] = ()


def _worker_initializer() -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = "1"


def _worker_entry(bounds: tuple[int, int]) -> dict[str, np.ndarray]:
    start, end = bounds
    if _WORKER_SCORE_CHUNK is None:
        raise RuntimeError("worker was not initialized with a score_chunk callable")
    # Environment variables set after fork cannot reliably resize BLAS/OpenMP
    # runtimes that were already loaded by NumPy/sklearn in the parent. Apply
    # an active runtime limit as well so N worker processes do not each spawn
    # their own N-thread native pools and oversubscribe the container.
    with threadpool_limits(limits=1), shared_fuzzy_ratios():
        payload = _WORKER_SCORE_CHUNK(start, end)
    expected = end - start
    if isinstance(payload, Mapping):
        values = payload
    elif len(_WORKER_SIGNAL_NAMES) == 1:
        values = {_WORKER_SIGNAL_NAMES[0]: payload}
    else:
        raise ValueError("score_chunk must return a mapping for multiple signals")
    out: dict[str, np.ndarray] = {}
    for name in _WORKER_SIGNAL_NAMES:
        if name not in values:
            raise ValueError(f"missing score signal {name!r}")
        array = np.asarray(values[name], dtype=np.float64)
        if array.ndim != 1 or len(array) != expected:
            raise ValueError(f"score signal {name!r} must contain exactly {expected} values")
        out[name] = array
    return out


def run_structured_chunks(
    *,
    row_count: int,
    chunk_size: int,
    signal_names: Sequence[str],
    score_chunk: Callable[[int, int], object],
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Score independent pair chunks, in parallel when fork is available.

    Falls back to the serial ``collect_chunked_scores`` path whenever fork is
    unavailable or only one worker is useful. Both paths return the same arrays.
    """
    global _WORKER_SCORE_CHUNK, _WORKER_SIGNAL_NAMES

    names = tuple(str(name) for name in signal_names)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("signal_names must contain unique non-empty names")
    if row_count == 0:
        return {name: np.empty(0, dtype=np.float64) for name in names}

    def serial() -> dict[str, np.ndarray]:
        with threadpool_limits(limits=1), shared_fuzzy_ratios():
            return collect_chunked_scores(
                row_count=int(row_count),
                chunk_size=int(chunk_size),
                signal_names=names,
                score_chunk=score_chunk,
            )

    bounds = list(batch_index_ranges(int(row_count), int(chunk_size)))
    worker_count = resolve_worker_count(workers)
    if worker_count <= 1 or len(bounds) <= 1 or not parallel_supported():
        return serial()

    buffers = {name: np.empty(int(row_count), dtype=np.float64) for name in names}
    _WORKER_SCORE_CHUNK = score_chunk
    _WORKER_SIGNAL_NAMES = names
    try:
        context = multiprocessing.get_context("fork")
        try:
            pool = context.Pool(
                processes=min(worker_count, len(bounds)),
                initializer=_worker_initializer,
            )
        except OSError as error:
            # A container that forbids extra processes must still produce a
            # submission; degrade to the serial path instead of failing the run.
            print(
                f"[v6] structured pool unavailable ({error}); falling back to serial",
                flush=True,
            )
            return serial()
        with pool:
            done = 0
            for (start, end), payload in zip(
                bounds,
                pool.imap(_worker_entry, bounds, chunksize=1),
                strict=True,
            ):
                for name in names:
                    values = payload[name]
                    if not np.isfinite(values).all():
                        raise ValueError(f"score signal {name!r} must be finite")
                    buffers[name][start:end] = values
                done += end - start
                if progress is not None:
                    progress(done, int(row_count))
    finally:
        _WORKER_SCORE_CHUNK = None
        _WORKER_SIGNAL_NAMES = ()
    return buffers
