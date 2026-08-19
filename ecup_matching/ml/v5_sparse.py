from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .textnorm import clean_text, normalize_item
from .v5_item_text import serialize_item_v5


SPARSE_PAIR_FEATURE_NAMES: tuple[str, ...] = (
    "name_char_tfidf_cosine",
    "name_word_tfidf_cosine",
    "full_char_tfidf_cosine",
    "full_word_tfidf_cosine",
)


@dataclass
class SparseItemEncoder:
    name_char: TfidfVectorizer
    name_word: TfidfVectorizer
    full_char: TfidfVectorizer
    full_word: TfidfVectorizer


def _corpora(items: pd.DataFrame) -> tuple[list[str], list[str]]:
    required = {"id", "name", "attributes", "category"}
    missing = required - set(items.columns)
    if missing:
        raise ValueError(f"items missing columns: {sorted(missing)}")
    names: list[str] = []
    full: list[str] = []
    for item_id, name, attributes, category in items[
        ["id", "name", "attributes", "category"]
    ].itertuples(index=False, name=None):
        norm = normalize_item(item_id, name, attributes, category)
        names.append(clean_text(name))
        full.append(serialize_item_v5(norm, max_chars=1000))
    return names, full


def fit_sparse_item_encoder(
    train_items: pd.DataFrame,
    *,
    max_char_features: int = 120_000,
    max_word_features: int = 60_000,
) -> SparseItemEncoder:
    if len(train_items) < 2:
        raise ValueError("at least two train items are required")
    if max_char_features <= 0 or max_word_features <= 0:
        raise ValueError("feature limits must be positive")
    names, full = _corpora(train_items)

    name_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_features=max_char_features,
        sublinear_tf=True,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )
    name_word = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w[\w+./-]*\b",
        ngram_range=(1, 2),
        min_df=1,
        max_features=max_word_features,
        sublinear_tf=True,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )
    full_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
        max_features=max_char_features,
        sublinear_tf=True,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )
    full_word = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w[\w+./=-]*\b",
        ngram_range=(1, 2),
        min_df=1,
        max_features=max_word_features,
        sublinear_tf=True,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )
    name_char.fit(names)
    name_word.fit(names)
    full_char.fit(full)
    full_word.fit(full)
    return SparseItemEncoder(name_char, name_word, full_char, full_word)


def _row_cosine(matrix, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # TfidfVectorizer emits L2-normalized rows, so row-wise dot is cosine.
    value = matrix[left].multiply(matrix[right]).sum(axis=1)
    return np.asarray(value).reshape(-1).astype(np.float32, copy=False)


def transform_sparse_pairs(
    encoder: SparseItemEncoder,
    items: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    if not {"id1", "id2"}.issubset(pairs.columns):
        raise ValueError("pairs must contain id1 and id2")
    names, full = _corpora(items)
    item_ids = items["id"].tolist()
    index = {item_id: row for row, item_id in enumerate(item_ids)}
    if len(index) != len(item_ids):
        raise ValueError("items contains duplicate id values")

    left: list[int] = []
    right: list[int] = []
    for id1, id2 in pairs[["id1", "id2"]].itertuples(index=False, name=None):
        if id1 not in index or id2 not in index:
            missing = id1 if id1 not in index else id2
            raise KeyError(f"pair references missing item {missing!r}")
        left.append(index[id1])
        right.append(index[id2])
    left_idx = np.asarray(left, dtype=np.int64)
    right_idx = np.asarray(right, dtype=np.int64)

    matrices = (
        encoder.name_char.transform(names),
        encoder.name_word.transform(names),
        encoder.full_char.transform(full),
        encoder.full_word.transform(full),
    )
    values = {
        name: _row_cosine(matrix, left_idx, right_idx)
        for name, matrix in zip(SPARSE_PAIR_FEATURE_NAMES, matrices, strict=True)
    }
    return pd.DataFrame(values, columns=SPARSE_PAIR_FEATURE_NAMES)
