from __future__ import annotations

import sys
from pathlib import Path

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
    if "--benchmark-only" not in sys.argv[1:]:
        finalize_v4_manifest(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
