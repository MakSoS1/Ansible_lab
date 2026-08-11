from __future__ import annotations

from typing import Mapping

import pandas as pd

from .category_attrs import learn_attribute_importance
from .features import normalize_items
from .features_v2 import build_pair_features_v2
from .textnorm import ItemNorm


def fit_fold_attribute_importance(
    items: pd.DataFrame,
    train_pairs: pd.DataFrame,
    *,
    min_support: int = 20,
    item_cache: Mapping[object, ItemNorm] | None = None,
) -> dict[str, dict[str, float]]:
    """Learn category/key weights from the supplied outer-train pairs only."""
    cache = item_cache if item_cache is not None else normalize_items(items)
    return learn_attribute_importance(
        items,
        train_pairs,
        min_support=min_support,
        item_cache=cache,
    )


def build_fold_weighted_features(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    importance: Mapping[str, Mapping[str, float]],
    *,
    item_cache: Mapping[object, ItemNorm] | None = None,
) -> pd.DataFrame:
    """Build v2 features using train-fold-derived attribute importance.

    Pair targets are intentionally not accepted: held-fold labels cannot affect
    feature construction once ``importance`` has been learned from outer train.
    """
    cache = item_cache if item_cache is not None else normalize_items(items)
    return build_pair_features_v2(
        items,
        pairs,
        attribute_importance=importance,
        item_cache=dict(cache),
    )
