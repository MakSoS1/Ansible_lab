"""Measure the shape of the pairs we validate on against the shape we are scored on.

Five iterations optimised local Macro AP on ``matches.parquet`` and moved the
public leaderboard by `0.0147` in total. D044 recorded the gap without
explaining it; D045 recorded that ``RETRIEVAL_PREVALENCE_RATIO`` is an
assumption nobody ever measured; D047 measured `fraction_degree_1 = 0.9708` on
one fold-0 score vector and noted, without being able to test it, that the
competition pairs come from retrieval where an anchor has many candidates.

If that note is right then the local number answers a different question than
the leaderboard does, and every candidate selected on it was selected on the
wrong axis. This driver settles it with the tables already on the runner:

1. the full human table's degree, prevalence and component structure;
2. the weak pool's degree, anchor-group and prevalence structure;
3. how far the two overlap at item level.

Exact *pair* overlap between the two tables is known to be zero, which is why
weak label quality was previously declared unverifiable. Item-level overlap is
a different question: human positive edges induce identity components, and any
weak pair whose endpoints are both human-covered can be labelled from that
closure without ever reading a weak target. That yields two things at once —
a retrieval-shaped evaluation set carrying human truth, and the first honest
measurement of how often the weak target agrees with it.

The proxy's negative assumption is stated rather than hidden: two items in
different human positive components are treated as a non-match. That is sound
only for items human labelling actually covers, so endpoints are restricted to
items appearing in at least one human row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# `ProgressReporter` lives on the v16 lineage, not on the v15 lineage this
# branch descends from, and the audit must run from either. Its phases are
# whole-array numpy operations with no incremental unit to tick anyway, so
# bounded silence is achieved by emitting a timestamped line per phase.


def _schema_report(path: Path | None) -> dict[str, object]:
    """Footer-only description; never materialises a 4 GB column."""
    if path is None:
        return {"exists": False}
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    handle = pq.ParquetFile(path)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(path.stat().st_size),
        "num_rows": int(handle.metadata.num_rows),
        "num_row_groups": int(handle.metadata.num_row_groups),
        "columns": [
            {"name": field.name, "type": str(field.type)} for field in handle.schema_arrow
        ],
    }


def _degree_report(id1: np.ndarray, id2: np.ndarray, *, label: str) -> dict[str, object]:
    endpoints = np.concatenate([id1, id2])
    unique, counts = np.unique(endpoints, return_counts=True)
    counts = counts.astype(np.float64)
    return {
        "label": label,
        "pairs": int(id1.shape[0]),
        "unique_items": int(unique.shape[0]),
        "degree_mean": float(counts.mean()),
        "degree_median": float(np.median(counts)),
        "degree_p90": float(np.percentile(counts, 90)),
        "degree_p99": float(np.percentile(counts, 99)),
        "degree_max": float(counts.max()),
        "fraction_degree_1": float((counts == 1).mean()),
        "fraction_degree_ge_3": float((counts >= 3).mean()),
        "fraction_degree_ge_10": float((counts >= 10).mean()),
    }


def _pair_keys(id1: np.ndarray, id2: np.ndarray) -> np.ndarray:
    """Order-independent exact pair keys.

    Packing into one int64 would overflow for real marketplace ids, so the two
    endpoints stay separate and are compared as a structured dtype.
    """
    lo = np.minimum(id1, id2).astype(np.int64)
    hi = np.maximum(id1, id2).astype(np.int64)
    stacked = np.ascontiguousarray(np.stack([lo, hi], axis=1))
    return stacked.view([("lo", np.int64), ("hi", np.int64)]).ravel()


def _components(id1: np.ndarray, id2: np.ndarray) -> dict[int, int]:
    """Union-find over positive edges only. Returns item id -> component root."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(id1.tolist(), id2.tolist()):
        a, b = int(a), int(b)
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {int(node): find(int(node)) for node in parent}


def _per_category(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for category, part in frame.groupby("category", sort=True):
        target = part["target"].to_numpy()
        out[str(category)] = {
            "rows": int(target.shape[0]),
            "positives": int(target.sum()),
            "prevalence": float(target.mean()),
        }
    return out


_STARTED = time.perf_counter()


def _emit(phase: str, payload: dict[str, object]) -> None:
    text = json.dumps(
        {"phase": phase, "elapsed_seconds": round(time.perf_counter() - _STARTED, 2), **payload},
        ensure_ascii=False,
        sort_keys=True,
    )
    print(text[:6000], flush=True)


def run_retrieval_audit(
    *,
    human_matches_path: Path,
    human_items_path: Path,
    output_path: Path,
    weak_matches_path: Path | None = None,
    items_path: Path | None = None,
    proxy_path: Path | None = None,
    max_proxy_rows: int = 4_000_000,
) -> dict[str, object]:
    started = time.perf_counter()
    report: dict[str, object] = {"generated_at": time.time()}

    report["inventory"] = {
        "human_matches": _schema_report(human_matches_path),
        "human_items": _schema_report(human_items_path),
        "weak_matches": _schema_report(weak_matches_path),
        "items": _schema_report(items_path),
    }
    _emit("inventory", report["inventory"])

    human = pd.read_parquet(human_matches_path, columns=["id1", "id2", "target"])
    human["id1"] = human["id1"].astype(np.int64)
    human["id2"] = human["id2"].astype(np.int64)
    human["target"] = human["target"].astype(np.int8)

    items = pd.read_parquet(human_items_path, columns=["id", "category"]).drop_duplicates("id")
    items["id"] = items["id"].astype(np.int64)
    category_by_id = pd.Series(
        items["category"].astype(str).to_numpy(), index=items["id"].to_numpy()
    )

    human["category"] = human["id1"].map(category_by_id)
    human_categorised = human.dropna(subset=["category"])

    positives = human.loc[human["target"] == 1]
    report["human"] = {
        "rows": int(len(human)),
        "prevalence": float(human["target"].mean()),
        "degree_all_edges": _degree_report(
            human["id1"].to_numpy(), human["id2"].to_numpy(), label="human_all"
        ),
        "degree_positive_edges": _degree_report(
            positives["id1"].to_numpy(), positives["id2"].to_numpy(), label="human_positive"
        ),
        "per_category": _per_category(human_categorised),
        "categorised_rows": int(len(human_categorised)),
    }

    root_by_item = _components(positives["id1"].to_numpy(), positives["id2"].to_numpy())
    sizes = pd.Series(list(root_by_item.values())).value_counts()
    report["human"]["components"] = {
        "count": int(sizes.shape[0]),
        "items_in_components": int(len(root_by_item)),
        "size_mean": float(sizes.mean()),
        "size_max": int(sizes.max()),
        "size_histogram": {str(k): int(v) for k, v in sizes.value_counts().head(12).items()},
    }
    _emit("human", report["human"])

    human_items_in_matches = np.union1d(human["id1"].to_numpy(), human["id2"].to_numpy())

    if weak_matches_path is None or not Path(weak_matches_path).exists():
        report["weak"] = {"exists": False}
        report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        return report

    weak_columns = [f.name for f in pq.ParquetFile(weak_matches_path).schema_arrow]
    wanted = [c for c in ("id1", "id2", "target") if c in weak_columns]
    weak = pd.read_parquet(weak_matches_path, columns=wanted)
    weak["id1"] = weak["id1"].astype(np.int64)
    weak["id2"] = weak["id2"].astype(np.int64)

    weak_report: dict[str, object] = {
        "exists": True,
        "rows": int(len(weak)),
        "columns": weak_columns,
        "degree": _degree_report(
            weak["id1"].to_numpy(), weak["id2"].to_numpy(), label="weak_all"
        ),
    }
    if "target" in weak.columns:
        target = weak["target"].to_numpy(dtype=np.float64)
        weak_report["target_mean"] = float(target.mean())
        weak_report["target_is_binary"] = bool(np.isin(np.unique(target), (0.0, 1.0)).all())
        weak_report["target_quantiles"] = {
            str(q): float(np.quantile(target, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)
        }
    anchor_sizes = weak.groupby("id1", sort=False).size()
    weak_report["anchor_groups"] = {
        "anchors": int(anchor_sizes.shape[0]),
        "candidates_per_anchor_mean": float(anchor_sizes.mean()),
        "candidates_per_anchor_median": float(anchor_sizes.median()),
        "candidates_per_anchor_p90": float(np.percentile(anchor_sizes.to_numpy(), 90)),
        "candidates_per_anchor_max": int(anchor_sizes.max()),
    }
    report["weak"] = weak_report
    _emit("weak", weak_report)

    weak_items = np.union1d(weak["id1"].to_numpy(), weak["id2"].to_numpy())
    shared = np.intersect1d(human_items_in_matches, weak_items, assume_unique=True)
    exact_overlap = int(
        np.intersect1d(
            _pair_keys(human["id1"].to_numpy(), human["id2"].to_numpy()),
            _pair_keys(weak["id1"].to_numpy(), weak["id2"].to_numpy()),
        ).shape[0]
    )

    report["overlap"] = {
        "human_items_in_matches": int(human_items_in_matches.shape[0]),
        "weak_items": int(weak_items.shape[0]),
        "shared_items": int(shared.shape[0]),
        "shared_fraction_of_human": float(
            shared.shape[0] / max(human_items_in_matches.shape[0], 1)
        ),
        "exact_canonical_pair_overlap": exact_overlap,
    }
    _emit("overlap", report["overlap"])

    # Retrieval-shaped proxy: weak topology, human component truth.
    covered = pd.Index(human_items_in_matches)
    both_covered = weak["id1"].isin(covered).to_numpy() & weak["id2"].isin(covered).to_numpy()
    proxy = weak.loc[both_covered]
    report["proxy_candidates_both_covered"] = int(both_covered.sum())
    if len(proxy) > max_proxy_rows:
        proxy = proxy.sample(n=max_proxy_rows, random_state=17)
    proxy = proxy.reset_index(drop=True)
    if "target" in proxy.columns:
        proxy = proxy.rename(columns={"target": "weak_target"})

    _emit("proxy-label-start", {"rows": int(len(proxy))})
    roots = pd.Series(root_by_item)
    # An item with no positive edge is its own singleton identity; the marker
    # must not collide with a real root, and real roots are non-negative ids.
    left = proxy["id1"].map(roots).fillna(-(proxy["id1"] + 1)).astype(np.int64)
    right = proxy["id2"].map(roots).fillna(-(proxy["id2"] + 1)).astype(np.int64)
    proxy["target"] = (left.to_numpy() == right.to_numpy()).astype(np.int8)
    proxy["category"] = proxy["id1"].map(category_by_id)
    proxy = proxy.dropna(subset=["category"]).reset_index(drop=True)

    proxy_report: dict[str, object] = {
        "rows": int(len(proxy)),
        "prevalence": float(proxy["target"].mean()) if len(proxy) else 0.0,
        "categories_present": int(proxy["category"].nunique()) if len(proxy) else 0,
    }
    if len(proxy):
        proxy_report["degree"] = _degree_report(
            proxy["id1"].to_numpy(), proxy["id2"].to_numpy(), label="proxy"
        )
        proxy_report["per_category"] = _per_category(proxy)

    # The weak label audit that exact-pair overlap could never deliver: the
    # component closure supplies human truth for rows the weak pool also holds.
    if "weak_target" in proxy.columns and len(proxy):
        truth = proxy["target"].to_numpy().astype(bool)
        soft = proxy["weak_target"].to_numpy(dtype=np.float64)
        hard = soft >= 0.5
        proxy_report["weak_label_audit"] = {
            "rows": int(len(proxy)),
            "weak_soft_mean_on_true_positive": float(soft[truth].mean()) if truth.any() else None,
            "weak_soft_mean_on_true_negative": float(soft[~truth].mean())
            if (~truth).any()
            else None,
            "weak_hard_precision": float(truth[hard].mean()) if hard.any() else None,
            "weak_hard_recall": float(hard[truth].mean()) if truth.any() else None,
            "weak_hard_positive_rate": float(hard.mean()),
            "human_positive_rate": float(truth.mean()),
        }
    report["proxy"] = proxy_report
    _emit("proxy", proxy_report)

    if proxy_path is not None and len(proxy):
        Path(proxy_path).parent.mkdir(parents=True, exist_ok=True)
        keep = [
            column
            for column in ("id1", "id2", "target", "weak_target", "category")
            if column in proxy.columns
        ]
        proxy[keep].to_parquet(proxy_path, index=False)
        report["proxy"]["path"] = str(proxy_path)

    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-matches", type=Path, required=True)
    parser.add_argument("--human-items", type=Path, required=True)
    parser.add_argument("--weak-matches", type=Path, default=None)
    parser.add_argument("--items", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy-output", type=Path, default=None)
    parser.add_argument("--max-proxy-rows", type=int, default=4_000_000)
    args = parser.parse_args()

    run_retrieval_audit(
        human_matches_path=args.human_matches,
        human_items_path=args.human_items,
        weak_matches_path=args.weak_matches,
        items_path=args.items,
        output_path=args.output,
        proxy_path=args.proxy_output,
        max_proxy_rows=args.max_proxy_rows,
    )


if __name__ == "__main__":
    main()
