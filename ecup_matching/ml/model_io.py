from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def save_model_bundle(model, model_path: Path, manifest_path: Path, manifest: dict[str, Any]) -> None:
    model_path = Path(model_path)
    manifest_path = Path(manifest_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path, compress=3)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model_bundle(model_path: Path, manifest_path: Path):
    model = joblib.load(Path(model_path))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return model, manifest
