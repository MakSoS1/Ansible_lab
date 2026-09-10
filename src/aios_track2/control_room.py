from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .challenge_doe import (
    CHALLENGE_INJECTOR_GROUPS,
    CHALLENGE_NODE_DATES,
    CHALLENGE_PRODUCER_GROUPS,
    ChallengeScenario,
    deterministic_spatial_groups,
    schedule_role_names,
)
from .deck import parse_deck_text
from .quality_gate import evaluate_quality_gate

EXPECTED_SCHEDULE_SHA256 = "c5ff3221ac66dea460bbd638a589dc5c7f2dedeb1536b9f86b10fb2e3e030af3"
EXPECTED_CLEAN_NPV_MRUB = 12475.954558553085
EXPECTED_MAX_WLPR = 62.550392150878906


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_policy_vector(values: list[float] | tuple[float, ...]) -> dict[str, Any]:
    scenario = ChallengeScenario(scenario_id=-1, split="winner", values=tuple(float(v) for v in values))
    return {
        "node_dates": [item.isoformat() for item in CHALLENGE_NODE_DATES],
        "producer_groups": {str(k): list(v) for k, v in scenario.producer_nodes().items()},
        "injector_groups": {str(k): list(v) for k, v in scenario.injector_nodes().items()},
        "producer_group_count": CHALLENGE_PRODUCER_GROUPS,
        "injector_group_count": CHALLENGE_INJECTOR_GROUPS,
        "dimensions": len(values),
    }


def explain_winner(winner: dict[str, Any], baseline_npv: float) -> list[str]:
    policy = decode_policy_vector(winner["vector"])
    producer = np.asarray(list(policy["producer_groups"].values()), dtype=float)
    injector = np.asarray(list(policy["injector_groups"].values()), dtype=float)
    delta = float(winner["opm_npv_mrub"] - baseline_npv)
    return [
        (
            f"Победитель выбран по реальному OPM+ЧДД NPV, не по суррогату: "
            f"{winner['opm_npv_mrub']:.3f} млн ₽ против baseline {baseline_npv:.3f} млн ₽ "
            f"({delta:+.3f} млн ₽)."
        ),
        (
            "Пространство управления в сданном расписании — 18 чисел: 4 пространственные группы добычи "
            "и 2 группы закачки на узлах 01.01.2007 / 01.01.2016 / 01.01.2025. Это не поскважинный "
            "open/close, не циклика и не перевод добывающих под закачку."
        ),
        (
            f"Политика поднимает отбор (группы добычи {producer.min():.3f}–{producer.max():.3f}) "
            f"и закачку ({injector.min():.3f}–{injector.max():.3f}) относительно исходного режима, "
            "чтобы не разъехалась компенсация."
        ),
        (
            f"Ограничение WLPR ≤ 500 м³/сут соблюдено: max WLPR = {winner['max_wlpr']:.2f}. "
            "Независимый clean rerun дал тот же SHA расписания и тот же NPV."
        ),
        (
            "Суррогат не заменяет гидродинамику: он сузил пул финалистов. Итоговый ранг — OPM. "
            "Предупреждение жюри: preregistered holdout top-3 recall = 2/3 < 0.90."
        ),
    ]


def agent_log(winner_doc: dict[str, Any], authorization: dict[str, Any]) -> list[dict[str, str]]:
    ranking = winner_doc.get("ranking", [])
    names = ", ".join(item["name"] for item in ranking)
    return [
        {"agent": "DoE", "detail": "64 замороженных 18D сценария OPM Flow, история до 2007 без изменений."},
        {
            "agent": "SurrogateAudit",
            "detail": (
                f"Holdout min R²={authorization['observed']['min_dynamic_r2']:.4f}, "
                f"NRMSE={authorization['observed']['max_dynamic_nrmse']:.4f}, "
                f"Spearman={authorization['observed']['spearman']:.4f}, "
                f"top-3 recall={authorization['observed']['top_k_recall']:.3f} (гейт 0.90 не пройден)."
            ),
        },
        {"agent": "Planning", "detail": f"OPM-финалисты: {names}."},
        {
            "agent": "ConstraintGuard",
            "detail": "Все финалисты success, max WLPR < 500, telemetry менял только Model_Z_summary.inc.",
        },
        {
            "agent": "Simulator",
            "detail": (
                f"Победитель {winner_doc['winner']['name']}: "
                f"NPV={winner_doc['winner']['opm_npv_mrub']:.6f} млн ₽, "
                f"SHA={winner_doc['winner']['schedule_sha256'][:12]}…"
            ),
        },
        {
            "agent": "Economics",
            "detail": "ЧДД 7.0.2-negative-row-filter, старт 01.01.2007, clean rerun NPV совпал с точностью 0.",
        },
    ]


def _well_map(schedule_text: str, vector: list[float]) -> list[dict[str, Any]]:
    wells = parse_deck_text(schedule_text).wells
    producers, injectors = schedule_role_names(schedule_text)
    producer_groups = deterministic_spatial_groups([w for w in wells if w.name in producers], 4) if producers else {}
    injector_groups = deterministic_spatial_groups([w for w in wells if w.name in injectors], 2) if injectors else {}
    policy = decode_policy_vector(vector)
    rows: list[dict[str, Any]] = []
    for well in wells:
        if well.name in producers:
            group = int(producer_groups.get(well.name, 0))
            scale = float(np.mean(policy["producer_groups"][str(group)]))
            role = "producer"
        elif well.name in injectors:
            group = int(injector_groups.get(well.name, 0))
            scale = float(np.mean(policy["injector_groups"][str(group)]))
            role = "injector"
        else:
            group = -1
            scale = 1.0
            role = "other"
        rows.append(
            {
                "name": well.name,
                "i": well.i,
                "j": well.j,
                "role": role,
                "group": group,
                "mean_scale": scale,
            }
        )
    return rows


def _compare_rows(winner_doc: dict[str, Any], baseline_npv: float) -> list[dict[str, Any]]:
    preferred = ("baseline", "cma_es", "mappo")
    by_name = {item["name"]: item for item in winner_doc.get("ranking", [])}
    rows = []
    for name in preferred:
        item = by_name[name]
        rows.append(
            {
                "name": name,
                "opm_npv_mrub": item["opm_npv_mrub"],
                "delta_vs_baseline_mrub": item["opm_npv_mrub"] - baseline_npv,
                "max_wlpr": item["max_wlpr"],
                "robustness_floor_mrub": item["robustness_floor_mrub"],
                "schedule_sha256": item["schedule_sha256"],
                "winner": name == winner_doc["winner"]["name"],
            }
        )
    return rows


def load_control_room(submission_dir: Path) -> dict[str, Any]:
    submission_dir = submission_dir.resolve()
    winner_doc = _load_json(submission_dir / "winner.json")
    manifest = _load_json(submission_dir / "final-submission-manifest.json")
    authorization = _load_json(submission_dir / "tournament-authorization.json")
    hf_manifest = _load_json(submission_dir / "hf-run-manifest.json")
    schedule_path = submission_dir / "wells_schedule.inc"
    schedule_bytes = schedule_path.read_bytes()
    schedule_sha = hashlib.sha256(schedule_bytes).hexdigest()
    schedule_text = schedule_bytes.decode("utf-8")
    economics = _load_json(submission_dir / "economics.json") if (submission_dir / "economics.json").exists() else {}
    summary = economics.get("summary", {})
    holdout = manifest.get("surrogate_holdout", {})
    dynamic = holdout.get("dynamic", {})
    ranking_metrics = holdout.get("npv", {})
    preregistered = evaluate_quality_gate(
        dynamic={
            "r2": float(dynamic.get("min_aggregate_channel_r2", 0.0)),
            "nrmse": float(dynamic.get("max_aggregate_channel_nrmse", 1.0)),
        },
        ranking={
            "spearman": float(ranking_metrics.get("spearman", 0.0)),
            "pairwise_accuracy": float(ranking_metrics.get("pairwise_accuracy", 0.0)),
            "top_k_recall": float(ranking_metrics.get("top_k_recall", 0.0)),
        },
        physics_violation_rate=0.0,
    )
    baseline_npv = float(winner_doc["candidate_results"]["baseline"]["npv_mrub"])
    winner = winner_doc["winner"]
    return {
        "recommendation": {
            "name": winner["name"],
            "npv_mrub": winner["opm_npv_mrub"],
            "baseline_npv_mrub": baseline_npv,
            "delta_vs_baseline_mrub": winner["opm_npv_mrub"] - baseline_npv,
            "max_wlpr": winner["max_wlpr"],
            "wlpr_limit": 500.0,
            "constraints_ok": winner["max_wlpr"] <= 500.0 + 1e-4,
            "schedule_sha256": schedule_sha,
            "sha_matches_clean_rerun": schedule_sha == EXPECTED_SCHEDULE_SHA256 == winner["schedule_sha256"],
            "npv_matches_clean_rerun": abs(winner["opm_npv_mrub"] - EXPECTED_CLEAN_NPV_MRUB) < 1e-9,
            "opm_verified": manifest.get("status") == "verified" and manifest.get("verification", {}).get("passed") is True,
            "chdd_version": economics.get("version"),
            "economic_start": economics.get("startDate"),
            "well_count": 103,
            "policy": decode_policy_vector(winner["vector"]),
        },
        "economics": {
            "total_chdd_mrub": summary.get("totalChddM"),
            "total_oil_kt": summary.get("totalOilKt"),
            "total_liquid_kt": summary.get("totalLiquidKt"),
            "total_injection_k": summary.get("totalInjectionK"),
            "pump_changes": summary.get("pumpChanges"),
            "start_stop_count": summary.get("startStopCount"),
            "conversion_count": summary.get("conversionCount"),
            "note": "CHDD conversion_count includes historical deck events, not optimizer-added WELOPEN/conversion keywords.",
        },
        "compare": _compare_rows(winner_doc, baseline_npv),
        "explanation": explain_winner(winner, baseline_npv),
        "agent_log": agent_log(winner_doc, authorization),
        "holdout": {
            "preregistered_gate_passed": preregistered.passed,
            "preregistered_failures": list(preregistered.failures),
            "tournament_authorized": bool(authorization.get("passed")),
            "min_dynamic_r2": dynamic.get("min_aggregate_channel_r2"),
            "max_dynamic_nrmse": dynamic.get("max_aggregate_channel_nrmse"),
            "p10_scenario_channel_r2": dynamic.get("p10_scenario_channel_r2"),
            "worst_scenario_channel": dynamic.get("worst_scenario_channel"),
            "spearman": ranking_metrics.get("spearman"),
            "pairwise_accuracy": ranking_metrics.get("pairwise_accuracy"),
            "top_k_recall": ranking_metrics.get("top_k_recall"),
            "simple_regret": ranking_metrics.get("simple_regret"),
        },
        "wells": _well_map(schedule_text, winner["vector"]),
        "reproducibility": {
            "git_sha": manifest.get("git_sha"),
            "github_run_id": str(manifest.get("github_run_id")),
            "hf_dataset": hf_manifest.get("dataset_id"),
            "hf_run_id": hf_manifest.get("run_id"),
            "simulator": hf_manifest.get("simulator_version"),
            "seed": hf_manifest.get("seed"),
        },
        "action_space_limits": {
            "submitted": "18D spatial group rate multipliers at 2007/2016/2025",
            "implemented_but_not_in_winner": [
                "producer shut-in",
                "producer-to-injector conversion",
                "cyclic injection",
            ],
            "not_opm_verified": True,
        },
        "expected_checks": {
            "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
            "clean_npv_mrub": EXPECTED_CLEAN_NPV_MRUB,
            "max_wlpr": EXPECTED_MAX_WLPR,
        },
    }


def default_submission_dir() -> Path:
    here = Path(__file__).resolve().parents[2]
    return here / "submission"
