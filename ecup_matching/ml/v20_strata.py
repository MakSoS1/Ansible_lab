from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import pandas as pd

from .textnorm import ItemNorm, clean_text


REASON_CODES = (
    "SAME_MODEL",
    "MODEL_CONFLICT",
    "CAPACITY_CONFLICT",
    "SIZE_CONFLICT",
    "PACK_COUNT_CONFLICT",
    "VARIANT_CONFLICT",
    "ACCESSORY",
    "DIFFERENT_GENERATION",
    "BRAND_CONFLICT",
    "SPARSE_EVIDENCE",
    "OTHER",
)

_ACCESSORY = {
    "чехол", "кейс", "case", "cover", "ремешок", "кабель", "адаптер", "зарядка",
    "стекло", "пленка", "держатель", "насадка", "крышка", "фильтр", "картридж",
}
_SIZE_KEYS = ("размер", "size", "длина", "ширина", "высота", "диаметр")
_BRAND_KEYS = ("бренд", "brand", "производитель", "manufacturer")
_VARIANT_KEYS = ("цвет", "color", "пол", "gender", "сезон", "season", "материал", "material")
_GENERATION_KEYS = ("год", "year", "поколение", "generation")
_COUNT_DIMS = {"count"}
_CAPACITY_DIMS = {"storage_bytes", "volume_ml", "battery_mah"}


@dataclass(frozen=True)
class PairStratum:
    category: str
    reason_code: str
    difficulty: str
    token_jaccard: float
    model_overlap: int
    numeric_conflict: bool
    attribute_overlap: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return float(len(a & b) / max(len(union), 1))


def _attr_values(item: ItemNorm, keys: tuple[str, ...]) -> frozenset[str]:
    values: set[str] = set()
    for key, value in item.attrs.items():
        k = clean_text(key)
        if any(marker in k for marker in keys):
            v = clean_text(value)
            if v:
                values.add(v)
    return frozenset(values)


def _conflicting_attr(a: ItemNorm, b: ItemNorm, keys: tuple[str, ...]) -> bool:
    left = _attr_values(a, keys)
    right = _attr_values(b, keys)
    return bool(left and right and left.isdisjoint(right))


def _quantity_map(item: ItemNorm) -> dict[str, frozenset[float]]:
    tmp: dict[str, set[float]] = {}
    for dim, value in item.quantities:
        tmp.setdefault(dim, set()).add(float(value))
    return {dim: frozenset(vals) for dim, vals in tmp.items()}


def _quantity_conflict(a: ItemNorm, b: ItemNorm, dims: set[str]) -> bool:
    left = _quantity_map(a)
    right = _quantity_map(b)
    for dim in dims:
        if dim in left and dim in right and left[dim].isdisjoint(right[dim]):
            return True
    return False


def _looks_accessory(a: ItemNorm, b: ItemNorm) -> bool:
    at = set(a.name_tokens)
    bt = set(b.name_tokens)
    a_acc = bool(at & _ACCESSORY)
    b_acc = bool(bt & _ACCESSORY)
    if a_acc == b_acc:
        return False
    base = bt if a_acc else at
    acc = at if a_acc else bt
    common = base & acc
    return len(common) >= 1 or _jaccard(a.name_tokens, b.name_tokens) >= 0.25


def _difficulty(similarity: float, reason: str) -> str:
    if reason == "SPARSE_EVIDENCE":
        return "hard"
    if reason.endswith("CONFLICT") or reason in {"ACCESSORY", "DIFFERENT_GENERATION"}:
        if similarity >= 0.55:
            return "hard"
        if similarity >= 0.25:
            return "medium"
        return "easy"
    if reason == "SAME_MODEL":
        return "hard" if similarity < 0.35 else "medium" if similarity < 0.70 else "easy"
    return "hard" if 0.35 <= similarity <= 0.75 else "medium"


def _specific_model_codes(codes: frozenset[str]) -> frozenset[str]:
    """Keep the most specific model/SKU codes rather than generic shared fragments.

    Product titles often contain both a family token (for example ``s24``) and
    a vendor SKU (``sms921b``). A shared short family token must not hide a
    conflict between distinct longer SKUs. We therefore retain all codes whose
    length is within one character of the longest observed code.
    """
    if not codes:
        return frozenset()
    longest = max(map(len, codes))
    return frozenset(code for code in codes if len(code) >= max(2, longest - 1))


def classify_pair_stratum(left: ItemNorm, right: ItemNorm) -> PairStratum:
    if clean_text(left.category) != clean_text(right.category):
        category = f"{clean_text(left.category)}|{clean_text(right.category)}"
    else:
        category = clean_text(left.category)

    sim = _jaccard(left.name_tokens, right.name_tokens)
    attr_keys_left = frozenset(left.attrs)
    attr_keys_right = frozenset(right.attrs)
    attr_overlap = _jaccard(attr_keys_left, attr_keys_right)
    model_overlap = len(left.model_codes & right.model_codes)
    left_specific = _specific_model_codes(left.model_codes)
    right_specific = _specific_model_codes(right.model_codes)
    specific_overlap = left_specific & right_specific
    numeric_conflict = bool(left.numbers and right.numbers and left.numbers.isdisjoint(right.numbers))

    if left_specific and right_specific and not specific_overlap:
        reason = "MODEL_CONFLICT"
    elif specific_overlap or model_overlap:
        reason = "SAME_MODEL"
    elif _quantity_conflict(left, right, _CAPACITY_DIMS):
        reason = "CAPACITY_CONFLICT"
    elif _conflicting_attr(left, right, _SIZE_KEYS) or _quantity_conflict(left, right, {"length_mm", "diagonal_in"}):
        reason = "SIZE_CONFLICT"
    elif _quantity_conflict(left, right, _COUNT_DIMS):
        reason = "PACK_COUNT_CONFLICT"
    elif _looks_accessory(left, right):
        reason = "ACCESSORY"
    elif _conflicting_attr(left, right, _GENERATION_KEYS):
        reason = "DIFFERENT_GENERATION"
    elif _conflicting_attr(left, right, _BRAND_KEYS):
        reason = "BRAND_CONFLICT"
    elif _conflicting_attr(left, right, _VARIANT_KEYS):
        reason = "VARIANT_CONFLICT"
    elif len(left.name_tokens | right.name_tokens) <= 4 and not (left.attrs or right.attrs):
        reason = "SPARSE_EVIDENCE"
    else:
        reason = "OTHER"

    return PairStratum(
        category=category,
        reason_code=reason,
        difficulty=_difficulty(sim, reason),
        token_jaccard=sim,
        model_overlap=int(model_overlap),
        numeric_conflict=bool(numeric_conflict),
        attribute_overlap=float(attr_overlap),
    )


def target_band(value: float) -> str:
    p = float(value)
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("target must be finite and in [0,1]")
    if p <= 0.15:
        return "strong_negative"
    if p < 0.45:
        return "soft_negative"
    if p <= 0.55:
        return "uncertain"
    if p < 0.85:
        return "soft_positive"
    return "strong_positive"


def audit_pair_frame(frame: pd.DataFrame, items: Mapping[object, ItemNorm]) -> pd.DataFrame:
    required = {"id1", "id2"}
    if not required.issubset(frame.columns):
        raise ValueError("pair frame requires id1,id2")
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        if row.id1 not in items or row.id2 not in items:
            continue
        stratum = classify_pair_stratum(items[row.id1], items[row.id2])
        record = stratum.to_dict()
        if hasattr(row, "target"):
            record["target_band"] = target_band(float(row.target))
            record["hard_target"] = int(float(row.target) >= 0.5)
        rows.append(record)
    if not rows:
        return pd.DataFrame(columns=["category", "reason_code", "difficulty", "count"])
    detailed = pd.DataFrame(rows)
    group_cols = ["category", "reason_code", "difficulty"]
    if "target_band" in detailed:
        group_cols += ["target_band", "hard_target"]
    return detailed.groupby(group_cols, dropna=False, sort=True).size().rename("count").reset_index()


__all__ = ["REASON_CODES", "PairStratum", "classify_pair_stratum", "target_band", "audit_pair_frame"]
