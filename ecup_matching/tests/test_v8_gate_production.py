from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_gate85_refit_uses_graph_strict_oof_as_quality_gate(tmp_path, monkeypatch):
    from ecup_matching.ml import run_v8_gate_production as module

    captured = {}

    def fake_fit_v6_gate_production(**kwargs):
        captured.update(kwargs)
        payload = {
            'version': 'v6-gate-production-refit',
            'coverage': kwargs['coverage'],
            'strict_selected_oof_macro_ap': kwargs['selected_oof_macro_ap'],
            'selection_gold_metric_opened': False,
            'selection_gold_rows_scored': 0,
        }
        Path(kwargs['metadata_output_path']).write_text(json.dumps(payload), encoding='utf-8')
        return payload

    monkeypatch.setattr(module, 'fit_v6_gate_production', fake_fit_v6_gate_production)
    meta = tmp_path/'meta.json'
    result = module.fit_v8_gate_production(
        coverage=0.85,
        base_oof_macro_ap=0.5999300791828578,
        graph_oof_macro_ap=0.6021573018691804,
        items_path=tmp_path/'items.parquet',
        matches_path=tmp_path/'matches.parquet',
        manifest_path=tmp_path/'manifest.json',
        anchor_oof_path=tmp_path/'anchor.parquet',
        typed_fusion_oof_path=tmp_path/'typed.parquet',
        category_output_path=tmp_path/'category.json',
        hgb_output_path=tmp_path/'hgb.joblib',
        metadata_output_path=meta,
        expected_split_sha='a'*64,
    )
    assert captured['coverage'] == pytest.approx(0.85)
    assert captured['selected_oof_macro_ap'] == pytest.approx(0.6021573018691804)
    assert result['version'] == 'v8-gate-production-refit'
    assert result['base_strict_oof_macro_ap'] == pytest.approx(0.5999300791828578)
    assert result['fold_local_graph_strict_oof_macro_ap'] == pytest.approx(0.6021573018691804)
    assert result['quality_gate_basis'] == 'fold-local graph OOF'
    assert result['selection_gold_metric_opened'] is False
    saved=json.loads(meta.read_text())
    assert saved == result


def test_gate55_is_rejected_because_graph_oof_stays_below_point60(tmp_path):
    from ecup_matching.ml.run_v8_gate_production import fit_v8_gate_production

    with pytest.raises(ValueError, match='graph OOF'):
        fit_v8_gate_production(
            coverage=0.55,
            base_oof_macro_ap=0.5966896566149946,
            graph_oof_macro_ap=0.5983324552728202,
            items_path=tmp_path/'items.parquet',
            matches_path=tmp_path/'matches.parquet',
            manifest_path=tmp_path/'manifest.json',
            anchor_oof_path=tmp_path/'anchor.parquet',
            typed_fusion_oof_path=tmp_path/'typed.parquet',
            category_output_path=tmp_path/'category.json',
            hgb_output_path=tmp_path/'hgb.joblib',
            metadata_output_path=tmp_path/'meta.json',
            expected_split_sha='a'*64,
        )


def test_refit_rejects_metric_drift_before_touching_data(tmp_path):
    from ecup_matching.ml.run_v8_gate_production import fit_v8_gate_production

    with pytest.raises(ValueError, match='frozen base OOF'):
        fit_v8_gate_production(
            coverage=0.85,
            base_oof_macro_ap=0.61,
            graph_oof_macro_ap=0.6021573018691804,
            items_path=tmp_path/'items.parquet',
            matches_path=tmp_path/'matches.parquet',
            manifest_path=tmp_path/'manifest.json',
            anchor_oof_path=tmp_path/'anchor.parquet',
            typed_fusion_oof_path=tmp_path/'typed.parquet',
            category_output_path=tmp_path/'category.json',
            hgb_output_path=tmp_path/'hgb.joblib',
            metadata_output_path=tmp_path/'meta.json',
            expected_split_sha='a'*64,
        )
