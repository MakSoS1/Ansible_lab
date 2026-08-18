from __future__ import annotations

import hashlib
import itertools
from collections import defaultdict
from typing import Iterable

import pandas as pd

from .textnorm import ItemNorm, clean_text, normalize_item
from .v20_strata import classify_pair_stratum


_GENERATOR_VERSION = "v20-real-item-blocks-v1"
_BRAND_KEYS = ("brand", "бренд", "manufacturer", "производитель")
_STOP = {"для", "the", "and", "with", "new", "товар", "шт", "pcs"}


def _brand(item: ItemNorm) -> str:
    for key, value in item.attrs.items():
        k = clean_text(key)
        if any(marker in k for marker in _BRAND_KEYS):
            v = clean_text(value)
            if v:
                return v
    return ""


def _block_signatures(item: ItemNorm) -> tuple[str, ...]:
    blocks: set[str] = set()
    for code in item.model_codes:
        blocks.add("model:" + code)
    brand = _brand(item)
    title_tokens = sorted(
        (t for t in item.name_tokens if len(t) >= 4 and t not in _STOP and not t.isdigit()),
        key=lambda value: (-len(value), value),
    )
    if brand:
        for token in title_tokens[:2]:
            blocks.add("brand-token:" + brand + "|" + token)
    if title_tokens:
        blocks.add("title:" + "|".join(sorted(title_tokens[:2])))
    for dim, value in sorted(item.quantities):
        if dim in {"storage_bytes", "volume_ml", "battery_mah", "count", "diagonal_in"}:
            blocks.add(f"quantity:{dim}:{value:g}")
    return tuple(sorted(blocks))


def _canonical_pair(a: object, b: object) -> tuple[object, object]:
    return (a, b) if str(a) < str(b) else (b, a)


def _priority(seed: int, category: str, block: str, a: object, b: object) -> bytes:
    return hashlib.sha256(f"{seed}\0{category}\0{block}\0{a}\0{b}".encode("utf-8")).digest()


def _fold_exclusion_sha(forbidden_ids: set[object]) -> str:
    payload = "\n".join(sorted(map(str, forbidden_ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generate_candidate_pairs(
    items: pd.DataFrame,
    *,
    forbidden_ids: Iterable[object],
    max_degree: int = 6,
    max_pairs_per_reason: int = 50_000,
    max_block_size: int = 40,
    seed: int = 2026,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"id", "name", "attributes", "category"}
    if not required.issubset(items.columns):
        raise ValueError(f"items missing required columns: {sorted(required - set(items.columns))}")
    if max_degree <= 0 or max_pairs_per_reason <= 0 or max_block_size < 2:
        raise ValueError("candidate caps must be positive")
    forbidden = set(forbidden_ids)
    work = items.loc[~items["id"].isin(forbidden), list(required)].copy().reset_index(drop=True)
    forbidden_rows = int(len(items) - len(work))
    norms: dict[object, ItemNorm] = {
        row.id: normalize_item(row.id, row.name, row.attributes, row.category)
        for row in work.itertuples(index=False)
    }

    blocks: dict[tuple[str, str], list[object]] = defaultdict(list)
    for item_id, norm in norms.items():
        for signature in _block_signatures(norm):
            blocks[(norm.category, signature)].append(item_id)

    raw: dict[tuple[object, object], tuple[bytes, dict[str, object]]] = {}
    skipped_large_blocks = 0
    for (category, block), ids in sorted(blocks.items()):
        unique_ids = sorted(set(ids), key=str)
        if len(unique_ids) < 2:
            continue
        if len(unique_ids) > max_block_size:
            # Deterministic bounded coverage rather than quadratic all-pairs.
            unique_ids = sorted(unique_ids, key=lambda x: _priority(seed, category, block, x, x))[:max_block_size]
            skipped_large_blocks += 1
        for a, b in itertools.combinations(unique_ids, 2):
            left, right = _canonical_pair(a, b)
            if left in forbidden or right in forbidden:
                continue
            stratum = classify_pair_stratum(norms[left], norms[right])
            record = {
                "id1": left,
                "id2": right,
                "category": stratum.category,
                "stratum": f"{stratum.category}|{stratum.reason_code}|{stratum.difficulty}",
                "reason_code": stratum.reason_code,
                "difficulty": stratum.difficulty,
                "generator_version": _GENERATOR_VERSION,
                "generator_block": block,
                "fold_exclusion_sha256": _fold_exclusion_sha(forbidden),
            }
            key = (left, right)
            priority = _priority(seed, category, block, left, right)
            previous = raw.get(key)
            if previous is None or priority < previous[0]:
                raw[key] = (priority, record)

    ordered = sorted(raw.values(), key=lambda pair: pair[0])
    degree: dict[object, int] = defaultdict(int)
    reason_count: dict[tuple[str, str], int] = defaultdict(int)
    chosen: list[dict[str, object]] = []
    for _, record in ordered:
        a, b = record["id1"], record["id2"]
        bucket = (str(record["category"]), str(record["reason_code"]))
        if degree[a] >= max_degree or degree[b] >= max_degree:
            continue
        if reason_count[bucket] >= max_pairs_per_reason:
            continue
        chosen.append(record)
        degree[a] += 1
        degree[b] += 1
        reason_count[bucket] += 1

    columns = [
        "id1", "id2", "category", "stratum", "reason_code", "difficulty",
        "generator_version", "generator_block", "fold_exclusion_sha256",
    ]
    out = pd.DataFrame(chosen, columns=columns)
    if "target" in out.columns:
        raise RuntimeError("target-free generator unexpectedly created a target")
    report = {
        "generator_version": _GENERATOR_VERSION,
        "input_items": int(len(items)),
        "usable_items": int(len(work)),
        "forbidden_rows": forbidden_rows,
        "candidate_pairs_before_caps": int(len(raw)),
        "output_pairs": int(len(out)),
        "max_degree": int(max_degree),
        "max_observed_degree": int(max(degree.values(), default=0)),
        "max_pairs_per_reason": int(max_pairs_per_reason),
        "blocks": int(len(blocks)),
        "large_blocks_bounded": int(skipped_large_blocks),
        "fold_exclusion_sha256": _fold_exclusion_sha(forbidden),
    }
    return out, report


__all__ = ["generate_candidate_pairs"]
