from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

from inspect_model_z import find_root_deck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="model-z-check-") as td:
        root = Path(td)
        with zipfile.ZipFile(args.archive) as zf:
            zf.extractall(root)
        deck, meta = find_root_deck(root)
        assert tuple(meta["dimensions"]) == (91, 102, 59), meta
        assert meta["well_count"] == 109, meta
        print(f"validated Model Z: deck={deck.relative_to(root)} dimensions={meta['dimensions']} wells=109")


if __name__ == "__main__":
    main()
