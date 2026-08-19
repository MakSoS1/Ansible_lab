from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_MATCH_COLUMNS = {"id1", "id2", "target"}
REQUIRED_ITEM_COLUMNS = {"id", "name", "attributes", "category"}


def _normalize_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=\d)\s+(?=[a-zа-яё])", "", text, flags=re.IGNORECASE)
    return text


def _length_stats(series: pd.Series) -> dict[str, float]:
    values = series.fillna("").astype(str).str.len()
    if len(values) == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
    }


def _quantiles(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "p50": float(values.quantile(0.50)),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
    }


def _validate_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def build_profile(matches_path: Path, items_path: Path) -> dict:
    matches = pd.read_parquet(matches_path)
    items = pd.read_parquet(items_path)
    _validate_columns(matches, REQUIRED_MATCH_COLUMNS, "matches parquet")
    _validate_columns(items, REQUIRED_ITEM_COLUMNS, "items parquet")

    warnings: list[str] = []

    targets = pd.to_numeric(matches["target"], errors="coerce")
    target_counts: dict[str, int] = {}
    for key, value in targets.value_counts(dropna=False).sort_index().items():
        if pd.isna(key):
            label = "null"
        elif float(key).is_integer():
            label = str(int(key))
        else:
            label = str(float(key))
        target_counts[label] = int(value)

    referenced_ids = pd.concat([matches["id1"], matches["id2"]], ignore_index=True)

    normalized_names = items["name"].map(_normalize_name)
    item_rows = int(len(items))
    distinct_normalized_names = int(normalized_names.nunique(dropna=False))
    duplicate_normalized_name_rate = (
        float(1.0 - distinct_normalized_names / item_rows) if item_rows else 0.0
    )

    lookup = items[["id", "name", "category"]].copy()
    lookup["_present"] = True
    left = lookup.rename(
        columns={
            "id": "id1",
            "name": "name1",
            "category": "category1",
            "_present": "_present1",
        }
    )
    right = lookup.rename(
        columns={
            "id": "id2",
            "name": "name2",
            "category": "category2",
            "_present": "_present2",
        }
    )
    pairs = matches.merge(left, on="id1", how="left").merge(right, on="id2", how="left")

    present1 = pairs["_present1"].fillna(False).astype(bool)
    present2 = pairs["_present2"].fillna(False).astype(bool)
    complete_pair = present1 & present2
    missing_pair_count = int((~complete_pair).sum())
    if missing_pair_count:
        warnings.append(
            f"{missing_pair_count} pair rows reference one or more missing items"
        )

    same_category = pairs.loc[complete_pair, "category1"].eq(
        pairs.loc[complete_pair, "category2"]
    )
    same_category_rate = float(same_category.mean()) if len(same_category) else 0.0
    cross_category_count = int((~same_category).sum()) if len(same_category) else 0
    if cross_category_count:
        warnings.append(
            f"{cross_category_count} complete pair rows are cross-category pairs"
        )

    pair_category = pairs["category1"].where(present1, pairs["category2"])
    category_frame = pd.DataFrame(
        {"category": pair_category.astype("string"), "target": targets}
    )
    category_breakdown: dict[str, dict[str, float | int]] = {}
    for category, group in category_frame.dropna(subset=["category"]).groupby(
        "category", sort=True
    ):
        category_breakdown[str(category)] = {
            "pairs": int(len(group)),
            "positive_rate": float(group["target"].mean()),
        }

    name1_lengths = pairs["name1"].fillna("").astype(str).str.len()
    name2_lengths = pairs["name2"].fillna("").astype(str).str.len()
    pair_text_lengths = (name1_lengths + name2_lengths).where(complete_pair)

    profile = {
        "matches": {
            "rows": int(len(matches)),
            "unique_item_ids_referenced": int(referenced_ids.nunique(dropna=True)),
            "positive_rate": float(targets.mean()) if len(targets) else 0.0,
            "target_counts": target_counts,
        },
        "items": {
            "rows": item_rows,
            "categories": int(items["category"].nunique(dropna=True)),
            "name_null_rate": float(items["name"].isna().mean()) if item_rows else 0.0,
            "name_length": _length_stats(items["name"]),
            "attributes_null_rate": float(items["attributes"].isna().mean())
            if item_rows
            else 0.0,
            "attributes_length": _length_stats(items["attributes"]),
            "distinct_normalized_names": distinct_normalized_names,
            "duplicate_normalized_name_rate": duplicate_normalized_name_rate,
        },
        "pair_categories": {
            "complete_pairs": int(complete_pair.sum()),
            "missing_item_pairs": missing_pair_count,
            "same_category_rate": same_category_rate,
            "cross_category_pairs": cross_category_count,
        },
        "pair_text_length_quantiles": _quantiles(pair_text_lengths),
        "category_breakdown": category_breakdown,
        "warnings": warnings,
    }
    return profile


def render_markdown(profile: dict) -> str:
    matches = profile["matches"]
    items = profile["items"]
    pair_categories = profile["pair_categories"]
    lines = [
        "# E-CUP Matching Human Data Profile",
        "",
        "This report intentionally contains aggregate statistics only; no raw product names, attributes, or item IDs are included.",
        "",
        "## Overview",
        "",
        f"- Pair rows: {matches['rows']:,}",
        f"- Referenced unique item IDs: {matches['unique_item_ids_referenced']:,}",
        f"- Positive rate: {matches['positive_rate']:.6f}",
        f"- Item rows: {items['rows']:,}",
        f"- Categories: {items['categories']:,}",
        f"- Duplicate normalized-name rate: {items['duplicate_normalized_name_rate']:.6f}",
        f"- Same-category pair rate: {pair_categories['same_category_rate']:.6f}",
        "",
        "## Text statistics",
        "",
        f"- Name null rate: {items['name_null_rate']:.6f}",
        f"- Name length mean / median / p95: {items['name_length']['mean']:.2f} / {items['name_length']['median']:.2f} / {items['name_length']['p95']:.2f}",
        f"- Attributes null rate: {items['attributes_null_rate']:.6f}",
        f"- Attributes length mean / median / p95: {items['attributes_length']['mean']:.2f} / {items['attributes_length']['median']:.2f} / {items['attributes_length']['p95']:.2f}",
        "",
        "## Category breakdown",
        "",
        "| Category | Pairs | Positive rate |",
        "|---|---:|---:|",
    ]
    for category, values in profile["category_breakdown"].items():
        lines.append(
            f"| {category} | {values['pairs']:,} | {values['positive_rate']:.6f} |"
        )

    lines.extend(["", "## Warnings", ""])
    if profile["warnings"]:
        lines.extend(f"- {warning}" for warning in profile["warnings"])
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile E-CUP human-labelled matching data")
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    profile = build_profile(args.matches, args.items)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.md_out.write_text(render_markdown(profile), encoding="utf-8")
    print(f"Wrote aggregate profile to {args.json_out} and {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
