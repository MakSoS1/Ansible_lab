from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd

from . import v7_neural as base


_TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", flags=re.IGNORECASE)
_IGNORED_TOKENS = {
    "cat", "name", "brand", "model", "identity", "numeric", "residual",
}


def _text_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in _TOKEN_RE.findall(str(text).casefold())
        if len(token) > 1 and token not in _IGNORED_TOKENS
    )


def _pair_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if intersection == 0:
        return 0.0
    # Containment is deliberately used instead of union Jaccard: a shorter title
    # embedded in a longer near-duplicate should still be considered hard.
    return float(intersection / max(1, min(len(left), len(right))))


def attach_target_free_pair_hardness(
    frame: pd.DataFrame,
    texts: Mapping[object, str],
    *,
    output_column: str = "negative_hardness",
) -> pd.DataFrame:
    required = {"id1", "id2", "target", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"hardness frame missing columns: {sorted(missing)}")
    work = frame.copy().reset_index(drop=True)
    ids = set(work["id1"].tolist()) | set(work["id2"].tolist())
    missing_ids = ids - set(texts)
    if missing_ids:
        first = min(missing_ids, key=lambda value: (type(value).__name__, repr(value)))
        raise KeyError(f"hardness text cache missing {len(missing_ids)} ids; first={first!r}")
    token_cache = {item_id: _text_tokens(texts[item_id]) for item_id in ids}
    hardness = np.fromiter(
        (
            _pair_similarity(token_cache[left], token_cache[right])
            for left, right in work[["id1", "id2"]].itertuples(index=False, name=None)
        ),
        dtype=np.float32,
        count=len(work),
    )
    work[output_column] = hardness
    return work


class HardNegativeMacroPairBatchSampler(base.MacroPairBatchSampler):
    """Macro-balanced sampler that spends part of negative slots on hard pairs."""

    def __init__(
        self,
        frame: pd.DataFrame,
        batch_size: int,
        seed: int,
        *,
        epoch: int = 0,
        hard_negative_fraction: float = 0.5,
        hardness_column: str = "negative_hardness",
    ):
        super().__init__(frame, batch_size, seed, epoch=epoch)
        if not 0.0 <= float(hard_negative_fraction) <= 1.0:
            raise ValueError("hard_negative_fraction must be in [0, 1]")
        self.hard_negative_fraction = float(hard_negative_fraction)
        self.hardness_column = str(hardness_column)
        if self.hard_negative_fraction > 0 and self.hardness_column not in self.frame:
            raise ValueError(f"missing hardness column: {self.hardness_column}")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        batches: list[list[int]] = []
        category_values = self.frame["category"].astype(str).to_numpy()
        targets = pd.to_numeric(self.frame["target"], errors="raise").to_numpy(float)
        hardness = (
            pd.to_numeric(self.frame[self.hardness_column], errors="raise").to_numpy(float)
            if self.hard_negative_fraction > 0
            else np.zeros(len(self.frame), dtype=float)
        )
        for category in self.categories:
            idx = np.flatnonzero(category_values == category)
            pos = idx[targets[idx] >= 0.5]
            neg = idx[targets[idx] < 0.5]
            hard_pool = neg
            if len(neg) and self.hard_negative_fraction > 0:
                order = sorted(neg.tolist(), key=lambda i: (-hardness[i], int(i)))
                hard_pool_size = max(1, int(np.ceil(len(order) * 0.5)))
                hard_pool = np.asarray(order[:hard_pool_size], dtype=int)
            for _ in range(self.batches_per_category):
                if len(pos) and len(neg) and self.batch_size >= 2:
                    positive_count = max(1, self.batch_size // 2)
                    negative_count = self.batch_size - positive_count
                    if negative_count == 0:
                        negative_count = 1
                        positive_count = self.batch_size - 1
                    batch = self._draw(rng, pos, positive_count)
                    hard_count = min(
                        negative_count,
                        int(round(negative_count * self.hard_negative_fraction)),
                    )
                    random_count = negative_count - hard_count
                    batch += self._draw(rng, hard_pool, hard_count)
                    batch += self._draw(rng, neg, random_count)
                else:
                    batch = self._draw(rng, idx, self.batch_size)
                rng.shuffle(batch)
                batches.append(batch)
        rng.shuffle(batches)
        return iter(batches)


def train_pair_phase_hardneg(*, weak: bool, frame: pd.DataFrame, texts, **kwargs):
    """Run the existing v7 objective, changing only human negative sampling."""
    if weak:
        return base.train_pair_phase(weak=True, frame=frame, texts=texts, **kwargs)
    hardened = attach_target_free_pair_hardness(frame, texts)
    original_sampler = base.MacroPairBatchSampler

    class _ConfiguredHardSampler(HardNegativeMacroPairBatchSampler):
        def __init__(self, sampler_frame, batch_size, seed, *, epoch=0):
            super().__init__(
                sampler_frame,
                batch_size,
                seed,
                epoch=epoch,
                hard_negative_fraction=0.5,
            )

    try:
        base.MacroPairBatchSampler = _ConfiguredHardSampler
        return base.train_pair_phase(weak=False, frame=hardened, texts=texts, **kwargs)
    finally:
        base.MacroPairBatchSampler = original_sampler
