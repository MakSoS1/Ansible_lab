from __future__ import annotations

import pandas as pd

from .features import FEATURE_NAMES, build_pair_features


def build_features_chunked(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    chunk_size: int = 25_000,
) -> pd.DataFrame:
    """Build pair features with bounded normalization cache memory.

    The human set contains almost two unique items per pair, so keeping normalized
    Python objects for every item at once is wasteful on a small CI runner. Each
    chunk normalizes only the item IDs it references and releases them afterwards.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if len(pairs) == 0:
        return pd.DataFrame(columns=FEATURE_NAMES)

    item_index = items.set_index("id", drop=False)
    chunks: list[pd.DataFrame] = []
    for start in range(0, len(pairs), chunk_size):
        pair_chunk = pairs.iloc[start : start + chunk_size]
        ids = pd.unique(pd.concat([pair_chunk["id1"], pair_chunk["id2"]], ignore_index=True))
        missing = [item_id for item_id in ids if item_id not in item_index.index]
        if missing:
            raise KeyError(f"pair chunk references {len(missing)} missing items; first={missing[0]!r}")
        item_chunk = item_index.loc[ids].reset_index(drop=True)
        chunks.append(build_pair_features(item_chunk, pair_chunk))
    return pd.concat(chunks, ignore_index=True)
