from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_gate40_refit_accepts_frozen_v9_metrics_and_preserves_sealed_gold(tmp_path, monkeypatch):
    from ecup_matching.ml import run_v9_gate_production as module

    captured = {}

    def fake_fit_v6_gate_production(**kwargs):
        captured.update(kwargs)
        payload = {
            "version": "v6-gate-production-refit",
            "coverage": kwargs["coverage"],
            "strict_selected_oof_macro_ap": kwargs["selected_oof_macro_ap"],
            "quality_gate_macro_ap": kwargs["quality_gate_macro_ap"],
            "selection_gold_metric_opened": False,
            "selection_gold_rows_scored": 0,
        }
        Path(kwargs["metadata_output_path"]).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(module, "fit_v6_gate_production", fake_fit_v6_gate_production)
    meta = tmp_path / "meta.json"
    result = module.fit_v9_gate40_production(
        coverage=0.40,
        base_oof_macro_ap=0.595505427416499,
        graph_oof_macro_ap=0.597005931143384,
        target_stress_mean=0.4515676235464289,
        items_path=tmp_path / "items.parquet",
        matches_path=tmp_path / "matches.parquet",
        manifest_path=tmp_path / "manifest.json",
        anchor_oof_path=tmp_path / "anchor.parquet",
        typed_fusion_oof_path=tmp_path / "typed.parquet",
        category_output_path=tmp_path / "category.json",
        hgb_output_path=tmp_path / "hgb.joblib",
        metadata_output_path=meta,
        expected_split_sha="a" * 64,
    )

    assert captured["coverage"] == pytest.approx(0.40)
    assert captured["selected_oof_macro_ap"] == pytest.approx(0.597005931143384)
    assert captured["quality_gate_macro_ap"] == pytest.approx(0.597005931143384)
    assert result["version"] == "v9-gate40-production-refit"
    assert result["base_strict_oof_macro_ap"] == pytest.approx(0.595505427416499)
    assert result["fold_local_graph_strict_oof_macro_ap"] == pytest.approx(0.597005931143384)
    assert result["target_stress_mean"] == pytest.approx(0.4515676235464289)
    assert result["leaderboard_anchor_v7_observed_by_owner"] == pytest.approx(0.36)
    assert result["leaderboard_anchor_used_for_fitting"] is False
    assert result["selection_gold_metric_opened"] is False
    assert result["selection_gold_rows_scored"] == 0
    assert json.loads(meta.read_text(encoding="utf-8")) == result


def test_v9_gate40_rejects_metric_drift_before_touching_data(tmp_path):
    from ecup_matching.ml.run_v9_gate_production import fit_v9_gate40_production

    with pytest.raises(ValueError, match="frozen graph OOF"):
        fit_v9_gate40_production(
            coverage=0.40,
            base_oof_macro_ap=0.595505427416499,
            graph_oof_macro_ap=0.60,
            target_stress_mean=0.4515676235464289,
            items_path=tmp_path / "items.parquet",
            matches_path=tmp_path / "matches.parquet",
            manifest_path=tmp_path / "manifest.json",
            anchor_oof_path=tmp_path / "anchor.parquet",
            typed_fusion_oof_path=tmp_path / "typed.parquet",
            category_output_path=tmp_path / "category.json",
            hgb_output_path=tmp_path / "hgb.joblib",
            metadata_output_path=tmp_path / "meta.json",
            expected_split_sha="a" * 64,
        )


def test_generic_v6_quality_threshold_can_be_lowered_only_explicitly():
    from ecup_matching.ml.run_v6_gate_production import validate_selected_oof_metric

    with pytest.raises(ValueError, match="quality gate"):
        validate_selected_oof_metric(0.597005931143384, quality_gate_macro_ap=0.60)
    assert validate_selected_oof_metric(
        0.597005931143384,
        quality_gate_macro_ap=0.597005931143384,
    ) == pytest.approx(0.597005931143384)
