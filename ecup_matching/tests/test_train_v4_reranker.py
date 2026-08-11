from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

import ecup_matching.ml.train_v4_reranker as v4
from ecup_matching.ml.train_v4_reranker import (
    V3_MACRO_AP,
    build_v4_metrics_payload,
    select_v4_candidate,
    shrink_category_alphas,
)
from ecup_matching.ml.v4_manifest import select_manifest_alphas


def _per_category(value: float = 0.5) -> dict[str, float]:
    return {f"category-{i:02d}": value + i * 1e-5 for i in range(20)}


def test_select_v4_candidate_requires_strict_v3_improvement() -> None:
    selected = select_v4_candidate(
        {
            "v4a": {"selected_macro_average_precision": 0.5250},
            "v4b": {"selected_macro_average_precision": 0.5410},
            "v4c": {"selected_macro_average_precision": 0.5390},
        }
    )
    assert selected == ("v4b", pytest.approx(0.5410))


def test_select_v4_candidate_returns_none_when_nothing_beats_v3() -> None:
    selected = select_v4_candidate(
        {
            "v4a": {"selected_macro_average_precision": V3_MACRO_AP},
            "v4b": {"selected_macro_average_precision": V3_MACRO_AP - 1e-6},
        }
    )
    assert selected is None


def test_shrink_category_alphas_stays_between_global_and_raw_optimum() -> None:
    raw = {"large": 0.90, "small": 0.10, "equal": 0.45}
    support = {"large": 10_000, "small": 100, "equal": 500}
    shrunk = shrink_category_alphas(
        raw,
        global_alpha=0.45,
        category_support=support,
        prior_strength=1_000,
    )
    assert 0.45 < shrunk["large"] < 0.90
    assert 0.10 < shrunk["small"] < 0.45
    assert shrunk["equal"] == pytest.approx(0.45)
    assert abs(shrunk["large"] - 0.90) < abs(shrunk["small"] - 0.10)


def test_select_manifest_alphas_uses_only_global_alpha_for_global_winner() -> None:
    stage = {
        "selected_blend": "global",
        "global_alpha_neural": 0.55,
        "shrunk_category_alphas": {"Электроника": 0.80, "Одежда": 0.20},
    }
    assert select_manifest_alphas(stage) == {"__global__": pytest.approx(0.55)}


def test_select_manifest_alphas_preserves_shrunk_categories_with_global_fallback() -> None:
    stage = {
        "selected_blend": "shrunk-category",
        "global_alpha_neural": 0.45,
        "shrunk_category_alphas": {"Электроника": 0.70, "Одежда": 0.30},
    }
    assert select_manifest_alphas(stage) == {
        "__global__": pytest.approx(0.45),
        "Электроника": pytest.approx(0.70),
        "Одежда": pytest.approx(0.30),
    }


def test_build_v4_metrics_payload_records_complete_comparable_evidence() -> None:
    stages = {
        "v4a": {
            "neural_macro_average_precision": 0.535,
            "selected_macro_average_precision": 0.540,
            "selected_per_category_ap": _per_category(0.50),
            "selected_blend": "global",
            "global_alpha_neural": 0.55,
        },
        "v4b": {
            "neural_macro_average_precision": 0.539,
            "selected_macro_average_precision": 0.544,
            "selected_per_category_ap": _per_category(0.51),
            "selected_blend": "shrunk-category",
            "global_alpha_neural": 0.60,
        },
        "v4c": {
            "neural_macro_average_precision": 0.538,
            "selected_macro_average_precision": 0.543,
            "selected_per_category_ap": _per_category(0.505),
            "selected_blend": "global",
            "global_alpha_neural": 0.60,
        },
    }
    payload = build_v4_metrics_payload(
        stages=stages,
        selected_stage="v4b",
        selected_macro_average_precision=0.544,
        validation_rows=73_131,
        validation_item_overlap=0,
        base_model="ai-forever/ruBert-base",
        base_model_revision="0123456789abcdef0123456789abcdef01234567",
        cuda_device="NVIDIA GeForce RTX 2060 SUPER",
        train_rows_human=292_523,
        train_rows_weak=600_000,
        total_seconds=1234.5,
    )
    assert payload["version"] == "v4-strong-reranker"
    assert payload["baseline_version"] == "v3"
    assert payload["baseline_macro_average_precision"] == pytest.approx(V3_MACRO_AP)
    assert payload["selected_stage"] == "v4b"
    assert payload["selected_macro_average_precision"] == pytest.approx(0.544)
    assert payload["accepted_as_improvement"] is True
    assert payload["validation_rows"] == 73_131
    assert payload["validation_item_overlap"] == 0
    assert payload["base_model_revision"] == "0123456789abcdef0123456789abcdef01234567"
    assert "NVIDIA" in payload["cuda_device"]
    assert payload["train_rows_human"] == 292_523
    assert payload["train_rows_weak"] == 600_000
    assert set(payload["stages"]) == {"v4a", "v4b", "v4c"}
    assert len(payload["stages"]["v4b"]["selected_per_category_ap"]) == 20
    assert math.isfinite(payload["total_seconds"])


def test_build_v4_metrics_payload_rejects_non_comparable_validation() -> None:
    with pytest.raises(ValueError, match="73,131"):
        build_v4_metrics_payload(
            stages={
                "v4a": {
                    "selected_macro_average_precision": 0.54,
                    "selected_per_category_ap": _per_category(),
                }
            },
            selected_stage="v4a",
            selected_macro_average_precision=0.54,
            validation_rows=73_130,
            validation_item_overlap=0,
            base_model="ai-forever/ruBert-base",
            base_model_revision="0123456789abcdef0123456789abcdef01234567",
            cuda_device="NVIDIA GeForce RTX 2060 SUPER",
            train_rows_human=292_523,
            train_rows_weak=0,
            total_seconds=1.0,
        )


def test_train_v4_prepares_memory_heavy_inputs_before_loading_bert(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []

    monkeypatch.setattr(v4, "_verify_model_revision", lambda *_args, **_kwargs: None)

    def fake_structured(*_args, **_kwargs):
        events.append("structured")
        return {"selected_candidate": v4.STRUCTURED_COLUMN}

    human = pd.DataFrame(
        [{
            "id1": 1,
            "id2": 2,
            "target": 1.0,
            "category": "A",
            "sample_weight": 1.0,
            "text_a": "a",
            "text_b": "b",
            "source": "human",
        }]
    )
    mixed = human.copy()
    valid = human.copy()

    def fake_curricula(**_kwargs):
        events.append("curricula")
        return human, mixed, valid, {"v4_human_rows": 1, "v4_weak_rows": 0}

    aligned = pd.DataFrame(
        [{"id1": 1, "id2": 2, "target": 1.0, "category": "A", v4.STRUCTURED_COLUMN: 0.5}]
    )

    def fake_align(*_args, **_kwargs):
        events.append("align")
        return aligned

    class StopAfterLoad(RuntimeError):
        pass

    def fake_load(*_args, **_kwargs):
        events.append("load")
        raise StopAfterLoad

    monkeypatch.setattr(v4, "train_structured_ablation", fake_structured)
    monkeypatch.setattr(v4, "_prepare_v4_curricula", fake_curricula)
    monkeypatch.setattr(v4, "_align_structured_validation", fake_align)
    monkeypatch.setattr(
        v4.pd,
        "read_parquet",
        lambda *_args, **_kwargs: pd.DataFrame(columns=[v4.STRUCTURED_COLUMN]),
    )
    monkeypatch.setattr(v4, "_load_local_model", fake_load)

    with pytest.raises(StopAfterLoad):
        v4.train_v4(
            human_items_path=tmp_path / "items_human.parquet",
            human_matches_path=tmp_path / "matches.parquet",
            llm_matches_path=tmp_path / "matches_llm.parquet",
            full_items_path=tmp_path / "items.parquet",
            output_dir=tmp_path / "out",
        )

    assert events == ["structured", "curricula", "align", "load"]
