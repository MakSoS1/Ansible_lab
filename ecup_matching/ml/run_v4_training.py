from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping, Any

from ecup_matching.build_submission_v4 import build_submission_v4

from .train_v4_reranker import main as train_main
from .v4_manifest import finalize_v4_manifest


def _output_dir_from_argv(argv: list[str]) -> Path:
    try:
        index = argv.index("--output-dir")
        value = argv[index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("--output-dir is required") from exc
    return Path(value)


def submission_eligible(
    metrics: Mapping[str, Any], manifest: Mapping[str, Any]
) -> bool:
    if metrics.get("accepted_as_improvement") is not True:
        return False
    raw_alphas = manifest.get("category_alphas")
    if not isinstance(raw_alphas, Mapping) or not raw_alphas:
        return False
    try:
        return any(float(value) > 0.0 for value in raw_alphas.values())
    except (TypeError, ValueError):
        return False


def main() -> int:
    output_dir = _output_dir_from_argv(sys.argv[1:])
    code = int(train_main())
    if code != 0:
        return code
    if "--benchmark-only" in sys.argv[1:]:
        return 0

    manifest = finalize_v4_manifest(output_dir)
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    if not submission_eligible(metrics, manifest):
        print(
            "[v4] submission skipped: no strict v3 improvement with a positive neural route; "
            "metrics and checkpoints are retained",
            flush=True,
        )
        return 0

    build_submission_v4(
        structured_model_path=output_dir / "structured-anchor" / "model.joblib",
        structured_manifest_path=output_dir / "structured-anchor" / "manifest.json",
        neural_model_dir=output_dir / "model",
        neural_manifest_path=output_dir / "manifest.json",
        output_path=output_dir / "ecup-v4-submission.zip",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
