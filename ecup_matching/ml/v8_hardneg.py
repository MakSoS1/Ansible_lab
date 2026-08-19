from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd

from . import v7_neural as base
from .v7_neural import MacroPairBatchSampler

_TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", flags=re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![0-9a-zа-яё])\d+(?:[.,]\d+)?(?![0-9a-zа-яё])", flags=re.IGNORECASE)
_IGNORED = {
    "cat", "name", "brand", "model", "identity", "numeric", "residual", "attr", "attributes"
}


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        t for t in _TOKEN_RE.findall(str(text).casefold())
        if len(t) > 1 and t not in _IGNORED
    )


def _model_tokens(tokens: frozenset[str]) -> frozenset[str]:
    # Short tokens such as s24 are often product-family names; require >=4 so a
    # shared family token does not hide a conflicting exact model identifier.
    return frozenset(
        t for t in tokens
        if len(t) >= 4 and any(ch.isalpha() for ch in t) and any(ch.isdigit() for ch in t)
    )


def _numbers(text: str) -> frozenset[str]:
    return frozenset(m.group(0).replace(",", ".") for m in _NUMBER_RE.finditer(str(text).casefold()))


def pair_hardness_v8(left_text: str, right_text: str) -> float:
    """Target-free hardness for product pairs.

    High score means the pair is lexically close while carrying identity-level
    model/SKU or numeric disagreement. This is intentionally not a match score;
    it is only used to decide which labelled negatives deserve training budget.
    """
    left = _tokens(left_text)
    right = _tokens(right_text)
    if not left or not right:
        lexical = 0.0
    else:
        lexical = len(left & right) / max(1, min(len(left), len(right)))

    lm, rm = _model_tokens(left), _model_tokens(right)
    model_conflict = 0.0
    if lm and rm:
        # A shared exact model is strong evidence against calling the pair a
        # model-conflict. Otherwise distinct model tokens make a close negative
        # particularly informative.
        model_conflict = 0.0 if (lm & rm) else 1.0

    ln, rn = _numbers(left_text), _numbers(right_text)
    numeric_conflict = 0.0
    if ln and rn and not (ln & rn):
        numeric_conflict = 1.0

    score = 0.65 * lexical + 0.25 * model_conflict + 0.10 * numeric_conflict
    return float(np.clip(score, 0.0, 1.0))


def attach_v8_hardness(
    frame: pd.DataFrame,
    texts: Mapping[object, str],
    *,
    output_column: str = "negative_hardness",
) -> pd.DataFrame:
    required = {"id1", "id2", "target", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"hardness frame missing columns: {sorted(missing)}")
    ids = set(frame["id1"].tolist()) | set(frame["id2"].tolist())
    missing_ids = ids - set(texts)
    if missing_ids:
        first = min(missing_ids, key=lambda x: (type(x).__name__, repr(x)))
        raise KeyError(f"hardness texts missing {len(missing_ids)} ids; first={first!r}")
    work = frame.copy().reset_index(drop=True)
    hardness = np.fromiter(
        (
            pair_hardness_v8(texts[left], texts[right])
            for left, right in work[["id1", "id2"]].itertuples(index=False, name=None)
        ),
        dtype=np.float32,
        count=len(work),
    )
    work[output_column] = hardness
    return work


class HardNegativeMacroPairBatchSamplerV8(MacroPairBatchSampler):
    """Macro-balanced batches with controlled oversampling of hard negatives."""

    def __init__(
        self,
        frame: pd.DataFrame,
        batch_size: int,
        seed: int,
        *,
        epoch: int = 0,
        hard_negative_fraction: float = 0.75,
        hard_pool_fraction: float = 0.25,
        hardness_column: str = "negative_hardness",
    ):
        super().__init__(frame, batch_size, seed, epoch=epoch)
        if not 0.0 <= float(hard_negative_fraction) <= 1.0:
            raise ValueError("hard_negative_fraction must be in [0,1]")
        if not 0.0 < float(hard_pool_fraction) <= 1.0:
            raise ValueError("hard_pool_fraction must be in (0,1]")
        if hardness_column not in self.frame:
            raise ValueError(f"missing hardness column {hardness_column!r}")
        self.hard_negative_fraction = float(hard_negative_fraction)
        self.hard_pool_fraction = float(hard_pool_fraction)
        self.hardness_column = str(hardness_column)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        batches: list[list[int]] = []
        cat_values = self.frame["category"].astype(str).to_numpy()
        targets = pd.to_numeric(self.frame["target"], errors="raise").to_numpy(float)
        hardness = pd.to_numeric(self.frame[self.hardness_column], errors="raise").to_numpy(float)
        if not np.isfinite(hardness).all():
            raise ValueError("hardness contains nonfinite values")

        for category in self.categories:
            idx = np.flatnonzero(cat_values == category)
            pos = idx[targets[idx] >= 0.5]
            neg = idx[targets[idx] < 0.5]
            hard_pool = neg
            if len(neg):
                ordered = np.asarray(
                    sorted(neg.tolist(), key=lambda i: (-float(hardness[i]), int(i))),
                    dtype=np.int64,
                )
                pool_n = max(1, int(np.ceil(len(ordered) * self.hard_pool_fraction)))
                hard_pool = ordered[:pool_n]
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


def train_pair_phase_v8_hardneg(*, weak: bool, frame: pd.DataFrame, texts, **kwargs):
    """Use v7 unchanged for weak pretraining; harden only the labelled human phase."""
    if weak:
        return base.train_pair_phase(weak=True, frame=frame, texts=texts, **kwargs)

    hardened = attach_v8_hardness(frame, texts)
    original_sampler = base.MacroPairBatchSampler

    class _ConfiguredV8HardSampler(HardNegativeMacroPairBatchSamplerV8):
        def __init__(self, sampler_frame, batch_size, seed, *, epoch=0):
            super().__init__(
                sampler_frame,
                batch_size,
                seed,
                epoch=epoch,
                hard_negative_fraction=0.75,
                hard_pool_fraction=0.25,
            )

    try:
        base.MacroPairBatchSampler = _ConfiguredV8HardSampler
        return base.train_pair_phase(
            weak=False,
            frame=hardened,
            texts=texts,
            **kwargs,
        )
    finally:
        base.MacroPairBatchSampler = original_sampler


__all__ = [
    "HardNegativeMacroPairBatchSamplerV8",
    "attach_v8_hardness",
    "pair_hardness_v8",
    "train_pair_phase_v8_hardneg",
]
