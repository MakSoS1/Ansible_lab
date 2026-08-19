"""Score modern teacher candidates on two fold-safe human audits and select an independent pair."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .v20_teacher_bakeoff import score_pair, score_teacher


CANONICAL_TEACHER_CANDIDATES: dict[str, dict[str, object]] = {
    "qwen35": {
        "model_id": "Qwen/Qwen3.5-4B",
        "family": "qwen",
        "backend": "openai-http",
        "quantization": "Q4_K_M",
        "artifact_repo": "bartowski/Qwen_Qwen3.5-4B-GGUF",
        "artifact_file": "Qwen_Qwen3.5-4B-Q4_K_M.gguf",
        "artifact_sha256": "13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983",
    },
    "pollux": {
        "model_id": "ai-forever/Pollux-4B-Judge",
        "family": "qwen",
        "backend": "openai-http",
        "quantization": "Q8_0",
        "artifact_repo": "ledgergap/Pollux-4B-Judge-GGUF",
        "artifact_file": "Pollux-4B-Judge.Q8_0.gguf",
        "artifact_sha256": "resolved-and-verified-at-runtime",
    },
    "gemma4": {
        "model_id": "google/gemma-4-E2B-it",
        "family": "gemma4",
        "backend": "openai-http",
        "quantization": "Q4_K_M",
        "artifact_repo": "bartowski/google_gemma-4-E2B-it-GGUF",
        "artifact_file": "google_gemma-4-E2B-it-Q4_K_M.gguf",
        "artifact_sha256": "923c4c86177d2ee173a7f5b4fa3d0ac65f5962ab15e6d6a5bc250aec4fd7bf7e",
    },
    "eurollm": {
        "model_id": "utter-project/EuroLLM-1.7B-Instruct",
        "family": "eurollm",
        "backend": "transformers-causal",
        "quantization": "none",
    },
    "fred": {
        "model_id": "ai-forever/FRED-T5-1.7B",
        "family": "fred-t5",
        "backend": "transformers-seq2seq",
        "quantization": "none",
    },
}


def _pair_key(teachers: list[str] | tuple[str, str]) -> tuple[str, str]:
    if len(teachers) != 2:
        raise ValueError("teacher pair must have exactly two names")
    return tuple(sorted((str(teachers[0]), str(teachers[1]))))


def select_two_fold_teacher_pair(
    teacher_reports: dict[int, dict[str, dict[str, object]]],
    pair_reports: dict[int, list[dict[str, object]]],
    *,
    fail_closed: bool = True,
) -> dict[str, object]:
    for fold in (0, 1):
        if fold not in teacher_reports or fold not in pair_reports:
            raise ValueError("two-fold teacher evidence requires folds 0 and 1")

    pair_maps: dict[int, dict[tuple[str, str], dict[str, object]]] = {}
    for fold in (0, 1):
        pair_maps[fold] = {
            _pair_key(list(report.get("teachers") or [])): report
            for report in pair_reports[fold]
            if len(list(report.get("teachers") or [])) == 2
        }

    eligible: list[dict[str, object]] = []
    for key in sorted(set(pair_maps[0]) & set(pair_maps[1])):
        names = list(key)
        fold_reports = [pair_maps[0][key], pair_maps[1][key]]
        if not all(bool(r.get("eligible")) for r in fold_reports):
            continue
        if not all(
            name in teacher_reports[fold] and bool(teacher_reports[fold][name].get("eligible"))
            for fold in (0, 1)
            for name in names
        ):
            continue
        families = {
            str(teacher_reports[0][name].get("family", ""))
            for name in names
        }
        if len(families) != 2:
            continue
        aggregate = {
            "teachers": names,
            "eligible": True,
            "consensus_precision": min(float(r.get("consensus_precision", 0.0)) for r in fold_reports),
            "critical_precision": min(float(r.get("critical_precision", 0.0)) for r in fold_reports),
            "coverage": min(float(r.get("coverage", 0.0)) for r in fold_reports),
            "rows_per_second": min(float(r.get("rows_per_second", 0.0)) for r in fold_reports),
            "fold_reports": {"0": fold_reports[0], "1": fold_reports[1]},
        }
        eligible.append(aggregate)

    if not eligible:
        if fail_closed:
            raise RuntimeError("no eligible teacher pair across both folds")
        return {
            "version": "v20-two-fold-teacher-selection-v1",
            "selected": None,
            "best_pair": None,
            "eligible_pairs": 0,
        }

    def rank(pair: dict[str, object]) -> tuple[float, float, float, float, str]:
        return (
            float(pair["consensus_precision"]),
            float(pair["critical_precision"]),
            float(pair["coverage"]),
            float(pair["rows_per_second"]),
            "|".join(map(str, pair["teachers"])),
        )

    best = max(eligible, key=rank)
    return {
        "version": "v20-two-fold-teacher-selection-v1",
        "selected": list(best["teachers"]),
        "best_pair": best,
        "eligible_pairs": int(len(eligible)),
    }


def labels_jsonl_to_frame(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            base = {"id1": int(rec["id1"]), "id2": int(rec["id2"]), "valid": bool(rec.get("valid"))}
            if not rec.get("valid"):
                rows.append({**base, "pred": None, "reason_code": "", "uncertain": True})
                continue
            decision = dict(rec["decision"])
            verdict = str(decision.get("verdict", "UNCERTAIN")).upper()
            rows.append(
                {
                    **base,
                    "pred": 1 if verdict == "MATCH" else (0 if verdict == "NON_MATCH" else None),
                    "reason_code": str(decision.get("reason_code", "")),
                    "uncertain": verdict == "UNCERTAIN",
                }
            )
    return pd.DataFrame(rows, columns=["id1", "id2", "pred", "reason_code", "valid", "uncertain"])


def run_from_config(config: dict[str, Any]) -> dict[str, object]:
    folds = dict(config.get("folds") or {})
    teacher_reports: dict[int, dict[str, dict[str, object]]] = {}
    pair_reports: dict[int, list[dict[str, object]]] = {}
    for fold in (0, 1):
        fold_cfg = dict(folds.get(str(fold)) or folds.get(fold) or {})
        truth_path = Path(str(fold_cfg.get("truth", "")))
        teacher_cfgs = dict(fold_cfg.get("teachers") or {})
        if not truth_path.is_file() or not teacher_cfgs:
            raise ValueError(f"fold {fold} config is incomplete")
        truth = pd.read_parquet(truth_path)
        reports: dict[str, dict[str, object]] = {}
        label_frames: dict[str, pd.DataFrame] = {}
        for name, raw_cfg in sorted(teacher_cfgs.items()):
            cfg = dict(raw_cfg)
            labels_path = Path(str(cfg.get("labels", "")))
            manifest_path = Path(str(cfg.get("manifest", "")))
            if not labels_path.is_file() or not manifest_path.is_file():
                raise ValueError(f"missing teacher evidence for {name} fold {fold}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            labels = labels_jsonl_to_frame(labels_path)
            label_frames[str(name)] = labels
            reports[str(name)] = score_teacher(truth, labels, manifest)
        teacher_reports[fold] = reports
        fold_pairs: list[dict[str, object]] = []
        for first, second in combinations(sorted(reports), 2):
            report = score_pair(
                truth,
                label_frames[first],
                label_frames[second],
                reports[first],
                reports[second],
            )
            report["teachers"] = [first, second]
            fold_pairs.append(report)
        pair_reports[fold] = fold_pairs

    selection = select_two_fold_teacher_pair(teacher_reports, pair_reports, fail_closed=False)
    return {
        "version": "v20-teacher-bakeoff-report-v1",
        "canonical_candidates": CANONICAL_TEACHER_CANDIDATES,
        "teacher_reports": {str(k): v for k, v in teacher_reports.items()},
        "pair_reports": {str(k): v for k, v in pair_reports.items()},
        "selection": selection,
        "sealed_gold_opened": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text(encoding="utf-8"))
    report = run_from_config(config)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selection = dict(report["selection"])
    if selection.get("selected") is None:
        raise RuntimeError("no eligible teacher pair across both folds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
