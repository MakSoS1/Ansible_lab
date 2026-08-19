from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .v5_validation import manifest_sha256, validate_manifest_no_overlap


IMMUTABLE_SPLIT_SHA = "aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b"
DEFAULT_FROZEN_SPLIT_PATH = (
    Path(__file__).resolve().parents[1] / "validation" / "v5_immutable_split_manifest.json"
)


def load_frozen_split_manifest(
    path: Path | str = DEFAULT_FROZEN_SPLIT_PATH,
    *,
    expected_sha: str = IMMUTABLE_SPLIT_SHA,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"frozen split manifest is missing: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"frozen split manifest is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("frozen split manifest must be a JSON object")

    actual_sha = manifest_sha256(manifest)
    if actual_sha != expected_sha:
        raise ValueError(
            f"immutable split SHA mismatch: expected={expected_sha}, actual={actual_sha}"
        )

    if manifest.get("version") != 1:
        raise ValueError("unexpected frozen split version")
    if manifest.get("seed") != 2026:
        raise ValueError("unexpected frozen split seed")
    if manifest.get("n_folds") != 5:
        raise ValueError("frozen split must contain exactly five development folds")
    if manifest.get("row_count") != 365_654:
        raise ValueError("frozen split row_count must be exactly 365654")
    if manifest.get("component_count") != 345_654:
        raise ValueError("frozen split component_count must be exactly 345654")

    gold = manifest.get("gold_rows")
    folds = manifest.get("fold_rows")
    if not isinstance(gold, list) or len(gold) != 80_444:
        raise ValueError("frozen split gold_rows must contain exactly 80444 rows")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("frozen split fold_rows must contain exactly five lists")
    if any(not isinstance(rows, list) for rows in folds):
        raise ValueError("every frozen development fold must be a list")
    if sum(len(rows) for rows in folds) != 285_210:
        raise ValueError("frozen development folds must contain exactly 285210 rows")

    flat = [*gold, *(row for rows in folds for row in rows)]
    if len(flat) != 365_654 or len(set(flat)) != 365_654:
        raise ValueError("frozen split rows must cover the human table exactly once")
    if min(flat) != 0 or max(flat) != 365_653:
        raise ValueError("frozen split row indices must span the complete human table")
    return manifest


def validate_frozen_split_against_matches(
    matches: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, int]:
    if len(matches) != int(manifest.get("row_count", -1)):
        raise ValueError(
            f"human match row count changed: manifest={manifest.get('row_count')} actual={len(matches)}"
        )
    required = {"id1", "id2"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"human matches missing columns: {sorted(missing)}")
    report = validate_manifest_no_overlap(matches, manifest)
    if report["row_coverage"] != len(matches):
        raise ValueError(f"frozen split row coverage mismatch: {report}")
    if report["duplicate_rows"] != 0:
        raise ValueError(f"frozen split contains duplicate rows: {report}")
    if report["missing_rows"] != 0:
        raise ValueError(f"frozen split contains missing rows: {report}")
    if report["cross_split_item_overlap"] != 0:
        raise ValueError(f"frozen split has cross-split item overlap: {report}")
    return report
