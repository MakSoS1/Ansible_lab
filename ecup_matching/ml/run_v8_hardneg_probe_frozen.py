from __future__ import annotations

from . import run_v7_fold0_probe as base
from .run_v7_outer_oof_frozen import _load_immutable_manifest
from .v8_hardneg import train_pair_phase_v8_hardneg


def main() -> int:
    original_train = base.train_pair_phase
    original_split = base._build_immutable_manifest
    try:
        base.train_pair_phase = train_pair_phase_v8_hardneg
        base._build_immutable_manifest = _load_immutable_manifest
        return base.main()
    finally:
        base.train_pair_phase = original_train
        base._build_immutable_manifest = original_split


if __name__ == "__main__":
    raise SystemExit(main())
