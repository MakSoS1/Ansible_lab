from __future__ import annotations

import sys
from pathlib import Path

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


def main() -> int:
    output_dir = _output_dir_from_argv(sys.argv[1:])
    code = int(train_main())
    if code != 0:
        return code
    if "--benchmark-only" in sys.argv[1:]:
        return 0

    finalize_v4_manifest(output_dir)
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
