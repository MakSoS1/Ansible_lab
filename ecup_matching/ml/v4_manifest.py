from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


def _alpha(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be finite and in [0,1]")
    return number


def select_manifest_alphas(stage_report: Mapping[str, Any]) -> dict[str, float]:
    selected = str(stage_report.get("selected_blend", ""))
    global_alpha = _alpha(stage_report.get("global_alpha_neural"), "global alpha")
    if selected == "global":
        return {"__global__": global_alpha}
    if selected != "shrunk-category":
        raise ValueError(f"unsupported selected blend: {selected!r}")
    raw = stage_report.get("shrunk_category_alphas")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("shrunk-category winner is missing category alphas")
    result = {"__global__": global_alpha}
    for category, value in raw.items():
        key = str(category)
        if not key or key == "__global__":
            raise ValueError("invalid category alpha key")
        result[key] = _alpha(value, f"category alpha {key!r}")
    return result


def finalize_v4_manifest(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "manifest.json"
    if not metrics_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("v4 metrics.json and manifest.json are required")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_stage = metrics.get("selected_stage")
    stages = metrics.get("stages")
    if not isinstance(selected_stage, str) or not isinstance(stages, dict):
        raise ValueError("v4 metrics are missing selected stage evidence")
    stage = stages.get(selected_stage)
    if not isinstance(stage, dict):
        raise ValueError("selected v4 stage report is missing")
    if manifest.get("selected_model_stage") != selected_stage:
        raise ValueError("manifest selected model stage disagrees with metrics")
    manifest["category_alphas"] = select_manifest_alphas(stage)
    manifest["selected_blend"] = stage.get("selected_blend")
    manifest["global_alpha_neural"] = _alpha(
        stage.get("global_alpha_neural"), "global alpha"
    )
    manifest["validation_macro_ap"] = float(
        stage.get("selected_macro_average_precision")
    )
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    return manifest
