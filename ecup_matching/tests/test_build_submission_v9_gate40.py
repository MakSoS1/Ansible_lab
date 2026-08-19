from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest


def _source_bundle(path: Path) -> None:
    old_meta = {
        "coverage": 0.95,
        "strict_selected_oof_macro_ap": 0.6006003614522999,
        "selection_gold_metric_opened": False,
        "selection_gold_rows_scored": 0,
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("run.py", "STALE=True\n")
        zf.writestr("ecup_matching/submission/predict_v5.py", "STALE=True\n")
        zf.writestr("ecup_matching/submission/predict_v6.py", "STALE=True\n")
        zf.writestr("model_v5_structured.joblib", b"structured")
        zf.writestr("model_v5_contrastive/config.json", "{}")
        zf.writestr("model_v5_teacher/config.json", "{}")
        zf.writestr("model_v6_category_shrunk.json", '{"old":true}')
        zf.writestr("model_v6_hgb_meta.joblib", b"old-hgb")
        zf.writestr("model_v6_gate_metadata.json", json.dumps(old_meta))


def _gate40_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    cat = tmp_path / "category.json"
    cat.write_text('{"coverage":0.4}', encoding="utf-8")
    hgb = tmp_path / "hgb.joblib"
    hgb.write_bytes(b"gate40-hgb")
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "version": "v9-gate40-production-refit",
                "coverage": 0.40,
                "base_strict_oof_macro_ap": 0.595505427416499,
                "fold_local_graph_strict_oof_macro_ap": 0.597005931143384,
                "strict_selected_oof_macro_ap": 0.597005931143384,
                "target_stress_mean": 0.4515676235464289,
                "target_stress_ratio": 0.566880890615799,
                "leaderboard_anchor_v7_observed_by_owner": 0.36,
                "leaderboard_anchor_used_for_fitting": False,
                "split_sha256": "aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b",
                "graph_config": {"rb": 0.0, "rt": 0.0, "ep": 0.02, "ap": 0.01},
                "selection_gold_metric_opened": False,
                "selection_gold_rows_scored": 0,
            }
        ),
        encoding="utf-8",
    )
    return cat, hgb, meta


def test_final_v9_builder_replaces_stale_runtime_and_preserves_frozen_evidence(tmp_path, monkeypatch):
    import ecup_matching.submission.build_submission_v9_gate40 as module

    source = tmp_path / "source.zip"
    _source_bundle(source)
    cat, hgb, meta = _gate40_files(tmp_path)
    graph = tmp_path / "v8_graph.py"
    graph.write_text("GRAPH=True\n", encoding="utf-8")
    post = tmp_path / "v8_submission_graph.py"
    post.write_text("POST=True\n", encoding="utf-8")

    def fake_copy_runtime_closure(dest):
        root = Path(dest)
        p = root / "ecup_matching/submission"
        p.mkdir(parents=True, exist_ok=True)
        (p / "predict_v5.py").write_text("def _legacy_text_cache(*a,norm_cache=None,**k): pass\n", encoding="utf-8")
        (p / "predict_v6.py").write_text(
            "def predict_to_csv_v6(**k): pass\n# _structured_scores_streaming run_structured_chunks torch_autocast\n",
            encoding="utf-8",
        )
        (p / "v6_parallel.py").write_text("max_workers_cap: int = 8\n", encoding="utf-8")
        return [
            "ecup_matching/submission/predict_v5.py",
            "ecup_matching/submission/predict_v6.py",
            "ecup_matching/submission/v6_parallel.py",
        ]

    monkeypatch.setattr(module, "copy_runtime_closure", fake_copy_runtime_closure)
    monkeypatch.setattr(module, "runtime_import_closure", lambda: ())

    out = tmp_path / "final.zip"
    result = module.build_v9_gate40(
        source_v6_zip=source,
        gate40_category_path=cat,
        gate40_hgb_path=hgb,
        gate40_metadata_path=meta,
        output_zip=out,
        v8_graph_source=graph,
        v8_submission_graph_source=post,
        source_commit="a" * 40,
    )
    assert result["archive_bytes"] == out.stat().st_size
    assert result["coverage"] == pytest.approx(0.40)
    with zipfile.ZipFile(out) as zf:
        assert zf.read("model_v6_hgb_meta.joblib") == b"gate40-hgb"
        saved = json.loads(zf.read("model_v6_gate_metadata.json"))
        assert saved["coverage"] == pytest.approx(0.40)
        assert saved["leaderboard_anchor_used_for_fitting"] is False
        assert "STALE=True" not in zf.read("ecup_matching/submission/predict_v6.py").decode()
        run = zf.read("run.py").decode()
        assert "EXPECTED_COVERAGE = 0.40" in run
        assert "apply_graph_to_prediction" in run
        final_meta = json.loads(zf.read("v9_metadata.json"))
        assert final_meta["version"] == "v9-gate40-fp16-graph"
        assert final_meta["validation"]["fold_local_graph_strict_oof_macro_ap"] == pytest.approx(0.597005931143384)
        assert final_meta["validation"]["target_stress_mean"] == pytest.approx(0.4515676235464289)
        assert final_meta["runtime"]["structured_worker_cap"] == 8
        assert final_meta["runtime"]["cuda_autocast"] == "float16"
        assert final_meta["sealed_gold_opened"] is False
