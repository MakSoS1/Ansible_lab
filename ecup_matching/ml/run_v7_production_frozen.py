"""Run the v7 production refit against the frozen split artifact.

``run_v7_production`` inherits ``_build_immutable_manifest`` from the OOF
driver, which *recomputes* the split from feature code and then checks its
SHA. That check correctly refused to run: the recomputed split hashed to
``d1b31023…`` instead of the immutable ``aae58fb4…``, so a production model
would have been trained against a different fold assignment than the one every
v7 diagnostic used.

The frozen manifest is the authority. Loading it also skips rebuilding pair
features over all 365,654 human rows, which is several minutes of
single-threaded work with no GPU activity at all.
"""

from __future__ import annotations

from . import run_v7_production as base
from .run_v7_outer_oof_frozen import _load_immutable_manifest


def main() -> int:
    base._build_immutable_manifest = _load_immutable_manifest
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
