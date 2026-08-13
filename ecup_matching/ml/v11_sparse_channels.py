from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer


FEATURE_NAMES = (
    "category",
    "name_word_cosine",
    "name_char_cosine",
    "attr_cosine",
    "name_exact_ci",
    "name_len_ratio",
    "attr_exact",
)


def _row_indices(items: pd.DataFrame, pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    row_of = {item_id: row for row, item_id in enumerate(items["id"].tolist())}
    try:
        left = np.fromiter((row_of[x] for x in pairs["id1"]), dtype=np.int64, count=len(pairs))
        right = np.fromiter((row_of[x] for x in pairs["id2"]), dtype=np.int64, count=len(pairs))
    except KeyError as exc:
        raise KeyError(f"pair references missing item: {exc.args[0]!r}") from exc
    return left, right


def _pair_cosine(texts: list[str], left: np.ndarray, right: np.ndarray, vectorizer: HashingVectorizer, batch_size: int = 50000) -> np.ndarray:
    matrix = vectorizer.transform(texts).tocsr()
    out = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), batch_size):
        stop = min(start + batch_size, len(left))
        prod = matrix[left[start:stop]].multiply(matrix[right[start:stop]])
        out[start:stop] = np.asarray(prod.sum(axis=1)).ravel().astype(np.float32, copy=False)
    del matrix
    gc.collect()
    np.clip(out, 0.0, 1.0, out=out)
    return out


def _hashing(*, n_features: int, analyzer: str, ngram_range: tuple[int, int]) -> HashingVectorizer:
    return HashingVectorizer(
        n_features=int(n_features),
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        analyzer=analyzer,
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b\w+\b" if analyzer == "word" else None,
        dtype=np.float32,
    )


def build_sparse_pair_features(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    max_name_features: int = 65536,
    max_char_features: int = 131072,
    max_attr_features: int = 65536,
) -> pd.DataFrame:
    required = {"id", "name", "attributes", "category"}
    missing = required - set(items.columns)
    if missing:
        raise ValueError(f"items missing columns: {sorted(missing)}")
    if not {"id1", "id2"}.issubset(pairs.columns):
        raise ValueError("pairs must contain id1 and id2")
    if items["id"].duplicated().any():
        raise ValueError("items.id must be unique")

    left, right = _row_indices(items, pairs)
    names = items["name"].fillna("").astype(str).tolist()
    attrs = items["attributes"].map(lambda x: "" if x is None else str(x)).tolist()

    word = _pair_cosine(
        names,
        left,
        right,
        _hashing(n_features=max_name_features, analyzer="word", ngram_range=(1, 2)),
    )
    char = _pair_cosine(
        names,
        left,
        right,
        _hashing(n_features=max_char_features, analyzer="char_wb", ngram_range=(3, 4)),
    )
    attr = _pair_cosine(
        attrs,
        left,
        right,
        _hashing(n_features=max_attr_features, analyzer="word", ngram_range=(1, 2)),
    )

    name_arr = np.asarray(names, dtype=object)
    attr_arr = np.asarray(attrs, dtype=object)
    lower = np.asarray([x.casefold().strip() for x in names], dtype=object)
    len_arr = np.fromiter((len(x) for x in names), dtype=np.float32, count=len(names))
    max_len = np.maximum(np.maximum(len_arr[left], len_arr[right]), 1.0)
    min_len = np.minimum(len_arr[left], len_arr[right])

    categories = items["category"].astype(str).to_numpy(dtype=object)
    left_cat = categories[left]
    right_cat = categories[right]
    if not np.array_equal(left_cat, right_cat):
        raise RuntimeError("pair endpoints disagree on category")

    frame = pd.DataFrame({
        "category": left_cat,
        "name_word_cosine": word,
        "name_char_cosine": char,
        "attr_cosine": attr,
        "name_exact_ci": (lower[left] == lower[right]).astype(np.float32),
        "name_len_ratio": (min_len / max_len).astype(np.float32),
        "attr_exact": (attr_arr[left] == attr_arr[right]).astype(np.float32),
    })
    return frame.loc[:, FEATURE_NAMES]


__all__ = ["FEATURE_NAMES", "build_sparse_pair_features"]
