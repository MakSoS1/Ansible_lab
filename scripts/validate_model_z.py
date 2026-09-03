from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

from inspect_model_z import find_root_deck

FINAL_MODEL_Z_DIMENSIONS = (91, 102, 59)
FINAL_MODEL_Z_WELL_COUNT = 103


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="model-z-check-") as td:
        root = Path(td)
        with zipfile.ZipFile(args.archive) as zf:
            zf.extractall(root)
        deck, meta = find_root_deck(root)
        assert tuple(meta["dimensions"]) == FINAL_MODEL_Z_DIMENSIONS, meta
        assert meta["well_count"] == FINAL_MODEL_Z_WELL_COUNT, meta
        print(
            f"validated final Model Z: deck={deck.relative_to(root)} "
            f"dimensions={meta['dimensions']} wells={FINAL_MODEL_Z_WELL_COUNT}"
        )


if __name__ == "__main__":
    main()
