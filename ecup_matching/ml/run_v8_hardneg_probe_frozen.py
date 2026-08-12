from __future__ import annotations

from . import run_v7_fold0_probe as base
from .v8_hardneg import train_pair_phase_v8_hardneg


def main() -> int:
    original = base.train_pair_phase
    try:
        base.train_pair_phase = train_pair_phase_v8_hardneg
        return base.main()
    finally:
        base.train_pair_phase = original


if __name__ == "__main__":
    raise SystemExit(main())
