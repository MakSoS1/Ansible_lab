from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

from .v20_policy import validate_fold_exclusion


SOURCE_CAPS = {
    "human": 1.0,
    "historical_weak": 0.75,
    "generated_llm": 0.90,
    "uncertain": 0.0,
}


def source_reliability_weight(source: str, reliability: float, *, admitted: bool) -> float:
    source = str(source)
    if source not in SOURCE_CAPS:
        raise ValueError(f"unknown source: {source}")
    if source != "human" and not admitted:
        return 0.0
    r = float(reliability)
    if not np.isfinite(r) or not 0.0 <= r <= 1.0:
        raise ValueError("reliability must be finite and in [0,1]")
    if source == "human":
        return 1.0
    return float(SOURCE_CAPS[source] * r)


def _stable_key(seed: int, *values: object) -> str:
    payload = "\0".join(map(str, (seed, *values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_source(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["id1", "id2", "target", "category", "reason_code", "source"])
    required = {"id1", "id2", "target", "category"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{source} frame missing: {sorted(required - set(frame.columns))}")
    out = frame.copy().reset_index(drop=True)
    out["source"] = source
    if "reason_code" not in out:
        out["reason_code"] = "OTHER"
    if "stratum_reliability" not in out:
        out["stratum_reliability"] = 1.0
    if "weak_weight" not in out:
        out["weak_weight"] = 1.0
    if "admitted" not in out:
        out["admitted"] = True if source in {"human", "historical_weak"} else False
    return out


def _auxiliary_columns(frame: pd.DataFrame) -> pd.DataFrame:
    reason = frame["reason_code"].astype(str).str.upper()
    frame = frame.copy()
    frame["aux_model_conflict"] = (reason == "MODEL_CONFLICT").astype(np.float32)
    frame["aux_numeric_conflict"] = reason.isin(
        ["CAPACITY_CONFLICT", "SIZE_CONFLICT", "PACK_COUNT_CONFLICT", "DIFFERENT_GENERATION"]
    ).astype(np.float32)
    frame["aux_variant_conflict"] = reason.isin(["VARIANT_CONFLICT", "BRAND_CONFLICT"]).astype(np.float32)
    frame["aux_accessory"] = (reason == "ACCESSORY").astype(np.float32)
    # Human/weak rows may carry deterministic reasons; generated rows are admitted rationale labels.
    frame["aux_mask"] = frame["reason_code"].notna().astype(np.float32)
    return frame


def build_gold_corpus(
    human: pd.DataFrame,
    historical_weak: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    forbidden_ids: Iterable[object],
    seed: int = 2026,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frames = [
        _normalize_source(human, "human"),
        _normalize_source(historical_weak, "historical_weak"),
        _normalize_source(generated, "generated_llm"),
    ]
    work = pd.concat(frames, ignore_index=True, sort=False)
    if work.empty:
        raise ValueError("gold corpus cannot be empty")
    forbidden = set(forbidden_ids)
    forbidden_mask = work["id1"].isin(forbidden) | work["id2"].isin(forbidden)
    forbidden_rows_removed = int(forbidden_mask.sum())
    work = work.loc[~forbidden_mask].copy().reset_index(drop=True)

    weights: list[float] = []
    for row in work.itertuples(index=False):
        reliability = float(getattr(row, "stratum_reliability", 1.0))
        base = float(getattr(row, "weak_weight", 1.0))
        admitted = bool(getattr(row, "admitted", True))
        w = source_reliability_weight(str(row.source), reliability, admitted=admitted)
        if str(row.source) == "historical_weak":
            w *= max(0.0, min(1.0, base))
        weights.append(float(w))
    work["match_weight"] = np.asarray(weights, dtype=np.float32)
    zero_weight_rows_removed = int((work["match_weight"] <= 0).sum())
    work = work.loc[work["match_weight"] > 0].copy().reset_index(drop=True)
    validate_fold_exclusion(work, forbidden)

    work["hard_target"] = (pd.to_numeric(work["target"], errors="raise").astype(float) >= 0.5).astype(np.int8)
    work["balance_key"] = (
        work["category"].astype(str) + "|" + work["hard_target"].astype(str) + "|" + work["reason_code"].astype(str)
    )
    work["sampling_key"] = [
        _stable_key(seed, row.source, row.category, row.hard_target, row.reason_code, row.id1, row.id2)
        for row in work.itertuples(index=False)
    ]
    work = _auxiliary_columns(work)
    work = work.sort_values(["balance_key", "sampling_key"], kind="mergesort").reset_index(drop=True)

    by_source = {str(k): int(v) for k, v in work["source"].value_counts().to_dict().items()}
    report = {
        "rows": int(len(work)),
        "by_source": by_source,
        "balance_keys": int(work["balance_key"].nunique()),
        "forbidden_rows_removed": forbidden_rows_removed,
        "zero_weight_rows_removed": zero_weight_rows_removed,
        "mean_match_weight": float(work["match_weight"].mean()) if len(work) else 0.0,
        "max_nonhuman_weight": float(work.loc[work["source"] != "human", "match_weight"].max()) if (work["source"] != "human").any() else 0.0,
    }
    return work, report


def balanced_sample(frame: pd.DataFrame, max_rows: int, *, seed: int = 2026) -> pd.DataFrame:
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if len(frame) <= max_rows:
        return frame.copy().reset_index(drop=True)
    if "balance_key" not in frame or "sampling_key" not in frame:
        raise ValueError("balanced_sample requires balance_key and sampling_key")
    groups = list(frame.groupby("balance_key", sort=True))
    quota = max(1, max_rows // max(1, len(groups)))
    chosen: list[pd.DataFrame] = []
    used: set[int] = set()
    for _, group in groups:
        take = min(quota, len(group))
        sample = group.sort_values("sampling_key", kind="mergesort").head(take)
        chosen.append(sample)
        used.update(sample.index.tolist())
    out = pd.concat(chosen, ignore_index=False) if chosen else frame.iloc[:0]
    remaining = max_rows - len(out)
    if remaining > 0:
        pool = frame.loc[~frame.index.isin(used)].sort_values("sampling_key", kind="mergesort")
        out = pd.concat([out, pool.head(remaining)], ignore_index=False)
    return out.sort_values("sampling_key", kind="mergesort").head(max_rows).reset_index(drop=True)


__all__ = ["SOURCE_CAPS", "source_reliability_weight", "build_gold_corpus", "balanced_sample"]
