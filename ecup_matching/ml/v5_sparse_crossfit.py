from __future__ import annotations

import pandas as pd

from .v5_sparse import fit_sparse_item_encoder, transform_sparse_pairs


def _pair_item_ids(pairs: pd.DataFrame) -> set[object]:
    if not {"id1", "id2"}.issubset(pairs.columns):
        raise ValueError("pairs must contain id1 and id2")
    return set(pairs["id1"].tolist()) | set(pairs["id2"].tolist())


def fit_transform_sparse_outer_fold(
    items: pd.DataFrame,
    train_pairs: pd.DataFrame,
    valid_pairs: pd.DataFrame,
    *,
    max_char_features: int = 120_000,
    max_word_features: int = 60_000,
) -> dict:
    """Fit sparse vocab/IDF on outer-train items and transform held items unseen."""
    train_ids = _pair_item_ids(train_pairs)
    valid_ids = _pair_item_ids(valid_pairs)
    all_needed = train_ids | valid_ids
    indexed = items.drop_duplicates("id", keep="first").set_index("id", drop=False)
    missing = all_needed - set(indexed.index.tolist())
    if missing:
        first = min(missing, key=lambda value: (type(value).__name__, repr(value)))
        raise KeyError(f"items missing {len(missing)} referenced ids; first={first!r}")

    train_items = indexed.loc[list(train_ids)].reset_index(drop=True)
    transform_items = indexed.loc[list(all_needed)].reset_index(drop=True)
    encoder = fit_sparse_item_encoder(
        train_items,
        max_char_features=max_char_features,
        max_word_features=max_word_features,
    )
    train_features = transform_sparse_pairs(encoder, transform_items, train_pairs)
    valid_features = transform_sparse_pairs(encoder, transform_items, valid_pairs)
    return {
        "train_features": train_features,
        "valid_features": valid_features,
        "train_item_count": int(len(train_ids)),
        "valid_item_count": int(len(valid_ids)),
        "name_word_vocabulary": set(encoder.name_word.vocabulary_.keys()),
        "name_char_vocabulary_size": int(len(encoder.name_char.vocabulary_)),
        "full_word_vocabulary_size": int(len(encoder.full_word.vocabulary_)),
        "full_char_vocabulary_size": int(len(encoder.full_char.vocabulary_)),
    }
