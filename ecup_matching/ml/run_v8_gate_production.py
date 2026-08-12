from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .run_v6_gate_production import fit_v6_gate_production


V8_GATE_OOF: dict[float, tuple[float, float]] = {
    0.55: (0.5966896566149946, 0.5983324552728202),
    0.70: (0.598287140395421, 0.6000750225512788),
    0.85: (0.5999300791828578, 0.6021573018691804),
    0.95: (0.6006003614522999, 0.6031727746091184),
}
METRIC_ATOL = 1e-12
QUALITY_GATE = 0.60


def _frozen_metrics(coverage: float) -> tuple[float, float]:
    coverage = float(coverage)
    for key, metrics in V8_GATE_OOF.items():
        if abs(coverage - key) <= METRIC_ATOL:
            return metrics
    raise ValueError(f"coverage must be one of {tuple(V8_GATE_OOF)}")


def fit_v8_gate_production(
    *,
    coverage: float,
    base_oof_macro_ap: float,
    graph_oof_macro_ap: float,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    anchor_oof_path: Path,
    typed_fusion_oof_path: Path,
    category_output_path: Path,
    hgb_output_path: Path,
    metadata_output_path: Path,
    expected_split_sha: str,
) -> dict:
    expected_base, expected_graph = _frozen_metrics(coverage)
    base_oof_macro_ap = float(base_oof_macro_ap)
    graph_oof_macro_ap = float(graph_oof_macro_ap)
    if not np.isfinite(base_oof_macro_ap) or abs(base_oof_macro_ap - expected_base) > METRIC_ATOL:
        raise ValueError(
            f"frozen base OOF metric mismatch for coverage={coverage}: "
            f"{base_oof_macro_ap} != {expected_base}"
        )
    if not np.isfinite(graph_oof_macro_ap) or abs(graph_oof_macro_ap - expected_graph) > METRIC_ATOL:
        raise ValueError(
            f"frozen fold-local graph OOF metric mismatch for coverage={coverage}: "
            f"{graph_oof_macro_ap} != {expected_graph}"
        )
    if graph_oof_macro_ap < QUALITY_GATE:
        raise ValueError(
            f"fold-local graph OOF must be >= {QUALITY_GATE:.2f}; got {graph_oof_macro_ap}"
        )

    payload = fit_v6_gate_production(
        coverage=float(coverage),
        # Keep the old v6 >=0.60 invariant intact. For v8 the selected quality
        # statistic is the pre-frozen, leakage-free, fold-local graph OOF AP.
        selected_oof_macro_ap=graph_oof_macro_ap,
        items_path=Path(items_path),
        matches_path=Path(matches_path),
        manifest_path=Path(manifest_path),
        anchor_oof_path=Path(anchor_oof_path),
        typed_fusion_oof_path=Path(typed_fusion_oof_path),
        category_output_path=Path(category_output_path),
        hgb_output_path=Path(hgb_output_path),
        metadata_output_path=Path(metadata_output_path),
        expected_split_sha=str(expected_split_sha),
    )
    if payload.get("selection_gold_metric_opened") is not False or int(
        payload.get("selection_gold_rows_scored", -1)
    ) != 0:
        raise RuntimeError("underlying v6 production refit violated sealed-gold contract")

    result = dict(payload)
    result.update(
        {
            "version": "v8-gate-production-refit",
            "base_strict_oof_macro_ap": base_oof_macro_ap,
            "fold_local_graph_strict_oof_macro_ap": graph_oof_macro_ap,
            "quality_gate_macro_ap": QUALITY_GATE,
            "quality_gate_basis": "fold-local graph OOF",
            "graph_config": {"rb": 0.0, "rt": 0.0, "ep": 0.02, "ap": 0.01},
            "graph_selection_gold_metric_opened": False,
            "graph_selection_gold_rows_scored": 0,
            "selection_gold_metric_opened": False,
            "selection_gold_rows_scored": 0,
        }
    )
    Path(metadata_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=float, required=True)
    parser.add_argument("--base-oof-macro-ap", type=float, required=True)
    parser.add_argument("--graph-oof-macro-ap", type=float, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchor-oof", type=Path, required=True)
    parser.add_argument("--typed-fusion-oof", type=Path, required=True)
    parser.add_argument("--category-output", type=Path, required=True)
    parser.add_argument("--hgb-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--expected-split-sha", required=True)
    args = parser.parse_args()
    fit_v8_gate_production(
        coverage=args.coverage,
        base_oof_macro_ap=args.base_oof_macro_ap,
        graph_oof_macro_ap=args.graph_oof_macro_ap,
        items_path=args.items,
        matches_path=args.matches,
        manifest_path=args.manifest,
        anchor_oof_path=args.anchor_oof,
        typed_fusion_oof_path=args.typed_fusion_oof,
        category_output_path=args.category_output,
        hgb_output_path=args.hgb_output,
        metadata_output_path=args.metadata_output,
        expected_split_sha=args.expected_split_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
