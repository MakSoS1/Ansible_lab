from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .ml.textnorm import clean_text, extract_quantities


def _load_payload(raw: Any) -> tuple[bool, Any]:
    if raw is None:
        return True, None
    if isinstance(raw, (dict, list)):
        return True, raw
    try:
        if isinstance(raw, float) and math.isnan(raw):
            return True, None
    except TypeError:
        pass
    try:
        return True, json.loads(str(raw))
    except Exception:
        return False, None


def _walk_payload(
    value: Any,
    *,
    prefix: str,
    paths: list[str],
    stats: dict[str, int],
    typed_dims: Counter[str],
) -> None:
    if isinstance(value, dict):
        if prefix:
            stats["nested_dict_count"] += 1
        for key, child in value.items():
            normalized_key = clean_text(key) or "<empty>"
            child_prefix = f"{prefix}.{normalized_key}" if prefix else normalized_key
            _walk_payload(
                child,
                prefix=child_prefix,
                paths=paths,
                stats=stats,
                typed_dims=typed_dims,
            )
        return

    if isinstance(value, list):
        if any(isinstance(child, dict) for child in value):
            stats["list_dict_count"] += 1
        if value and all(isinstance(child, str) for child in value):
            stats["list_str_count"] += 1
        if not value:
            return
        for child in value:
            # Keep the logical attribute path stable across list elements. That
            # mirrors how a robust parser should preserve structure without
            # making list position part of the semantic key.
            _walk_payload(
                child,
                prefix=prefix,
                paths=paths,
                stats=stats,
                typed_dims=typed_dims,
            )
        return

    if not prefix:
        return
    stats["leaf_count"] += 1
    paths.append(prefix)
    for dim, _ in extract_quantities(clean_text(value)):
        typed_dims[dim] += 1


def inspect_attribute_payload(raw: Any) -> dict[str, Any]:
    parse_success, obj = _load_payload(raw)
    if not parse_success:
        return {
            "parse_success": False,
            "top_level_type": "invalid",
            "nested_dict_count": 0,
            "list_str_count": 0,
            "list_dict_count": 0,
            "leaf_count": 0,
            "leaf_key_collision_count": 0,
            "colliding_leaf_keys": [],
            "typed_quantity_leaf_count": 0,
            "typed_dimensions": {},
        }

    if isinstance(obj, dict):
        top_level_type = "dict"
    elif isinstance(obj, list):
        top_level_type = "list"
    elif obj is None:
        top_level_type = "null"
    else:
        top_level_type = "scalar"

    stats = {
        "nested_dict_count": 0,
        "list_str_count": 0,
        "list_dict_count": 0,
        "leaf_count": 0,
    }
    paths: list[str] = []
    typed_dims: Counter[str] = Counter()
    _walk_payload(obj, prefix="", paths=paths, stats=stats, typed_dims=typed_dims)

    by_leaf: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        by_leaf[path.rsplit(".", 1)[-1]].add(path)
    colliding = sorted(leaf for leaf, full_paths in by_leaf.items() if len(full_paths) > 1)

    return {
        "parse_success": True,
        "top_level_type": top_level_type,
        **stats,
        "leaf_key_collision_count": int(len(colliding)),
        "colliding_leaf_keys": colliding,
        "typed_quantity_leaf_count": int(sum(typed_dims.values())),
        "typed_dimensions": dict(sorted(typed_dims.items())),
    }


def audit_parquet(path: Path, *, batch_size: int = 8192) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    total = 0
    parse_success = 0
    top_types: Counter[str] = Counter()
    sums: Counter[str] = Counter()
    collision_items = 0
    list_dict_items = 0
    typed_items = 0
    collision_keys: Counter[str] = Counter()
    typed_dims: Counter[str] = Counter()

    for batch in parquet.iter_batches(columns=["attributes"], batch_size=batch_size):
        values = batch.column(0).to_pylist()
        for raw in values:
            total += 1
            report = inspect_attribute_payload(raw)
            parse_success += int(report["parse_success"])
            top_types[report["top_level_type"]] += 1
            for key in ("nested_dict_count", "list_str_count", "list_dict_count", "leaf_count"):
                sums[key] += int(report[key])
            if report["leaf_key_collision_count"]:
                collision_items += 1
                collision_keys.update(report["colliding_leaf_keys"])
            if report["list_dict_count"]:
                list_dict_items += 1
            if report["typed_quantity_leaf_count"]:
                typed_items += 1
            typed_dims.update(report["typed_dimensions"])

    denominator = max(total, 1)
    return {
        "rows": int(total),
        "parse_success_rows": int(parse_success),
        "parse_success_rate": float(parse_success / denominator),
        "top_level_types": dict(sorted(top_types.items())),
        "nested_dict_occurrences": int(sums["nested_dict_count"]),
        "list_str_occurrences": int(sums["list_str_count"]),
        "list_dict_occurrences": int(sums["list_dict_count"]),
        "list_dict_items": int(list_dict_items),
        "list_dict_item_rate": float(list_dict_items / denominator),
        "leaf_values": int(sums["leaf_count"]),
        "mean_leaf_values_per_item": float(sums["leaf_count"] / denominator),
        "leaf_collision_items": int(collision_items),
        "leaf_collision_item_rate": float(collision_items / denominator),
        "top_colliding_leaf_keys": [
            {"leaf_key": key, "items": int(count)}
            for key, count in collision_keys.most_common(50)
        ],
        "typed_quantity_items": int(typed_items),
        "typed_quantity_item_rate": float(typed_items / denominator),
        "typed_dimensions": dict(sorted((key, int(value)) for key, value in typed_dims.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit raw E-CUP item attribute JSON structure")
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    report = audit_parquet(args.items, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
