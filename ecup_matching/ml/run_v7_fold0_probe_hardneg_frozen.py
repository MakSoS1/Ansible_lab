from __future__ import annotations

from . import run_v7_fold0_probe as probe
from .run_v7_outer_oof_frozen import _load_immutable_manifest
from .v7_hardneg import train_pair_phase_hardneg


def main() -> int:
    probe._build_immutable_manifest = _load_immutable_manifest
    probe.train_pair_phase = train_pair_phase_hardneg
    return probe.main()


if __name__ == "__main__":
    raise SystemExit(main())
