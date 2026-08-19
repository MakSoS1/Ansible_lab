from __future__ import annotations

import pandas as pd

from ecup_matching.ml.run_v20_teacher_bakeoff import (
    CANONICAL_TEACHER_CANDIDATES,
    select_two_fold_teacher_pair,
)


def test_canonical_pool_has_modern_and_russian_controls_but_no_yandex():
    ids = {str(v["model_id"]) for v in CANONICAL_TEACHER_CANDIDATES.values()}
    assert "Qwen/Qwen3.5-4B" in ids
    assert "google/gemma-4-E2B-it" in ids
    assert "utter-project/EuroLLM-1.7B-Instruct" in ids
    assert "ai-forever/FRED-T5-1.7B" in ids
    assert not any("yandex" in value.lower() for value in ids)


def _teacher_report(name: str, family: str, *, eligible: bool = True, rps: float = 2.0) -> dict:
    return {
        "eligible": eligible,
        "model_id": name,
        "revision": (family[0] if family else "a") * 40,
        "family": family,
        "rows_per_second": rps,
    }


def _pair(a: str, b: str, *, precision: float, critical: float, coverage: float, eligible: bool = True) -> dict:
    return {
        "teachers": [a, b],
        "eligible": eligible,
        "consensus_precision": precision,
        "critical_precision": critical,
        "coverage": coverage,
        "rows_per_second": 1.0,
    }


def test_pair_must_pass_both_folds_and_uses_worst_fold_quality():
    teacher_reports = {
        0: {
            "qwen": _teacher_report("Qwen/Qwen3.5-4B", "qwen35"),
            "gemma": _teacher_report("google/gemma-4-E2B-it", "gemma4"),
            "euro": _teacher_report("utter-project/EuroLLM-1.7B-Instruct", "eurollm"),
        },
        1: {
            "qwen": _teacher_report("Qwen/Qwen3.5-4B", "qwen35"),
            "gemma": _teacher_report("google/gemma-4-E2B-it", "gemma4"),
            "euro": _teacher_report("utter-project/EuroLLM-1.7B-Instruct", "eurollm"),
        },
    }
    pair_reports = {
        0: [
            _pair("qwen", "gemma", precision=0.999, critical=0.998, coverage=0.75),
            _pair("qwen", "euro", precision=0.997, critical=0.997, coverage=0.90),
        ],
        1: [
            _pair("qwen", "gemma", precision=0.996, critical=0.996, coverage=0.76),
            _pair("qwen", "euro", precision=0.998, critical=0.998, coverage=0.89),
        ],
    }
    selected = select_two_fold_teacher_pair(teacher_reports, pair_reports)
    assert selected["selected"] == ["qwen", "euro"]
    assert selected["best_pair"]["consensus_precision"] == 0.997
    assert selected["best_pair"]["critical_precision"] == 0.997


def test_pair_failing_one_fold_is_not_eligible():
    teacher_reports = {
        0: {"qwen": _teacher_report("Qwen/Qwen3.5-4B", "qwen35"), "gemma": _teacher_report("google/gemma-4-E2B-it", "gemma4")},
        1: {"qwen": _teacher_report("Qwen/Qwen3.5-4B", "qwen35"), "gemma": _teacher_report("google/gemma-4-E2B-it", "gemma4")},
    }
    pair_reports = {
        0: [_pair("qwen", "gemma", precision=0.999, critical=0.999, coverage=0.8)],
        1: [_pair("qwen", "gemma", precision=0.999, critical=0.999, coverage=0.8, eligible=False)],
    }
    selected = select_two_fold_teacher_pair(teacher_reports, pair_reports, fail_closed=False)
    assert selected["selected"] is None
    assert selected["eligible_pairs"] == 0
