from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from aios_track2.deck import parse_deck


def find_root_deck(root: Path) -> tuple[Path, dict]:
    candidates = sorted(root.rglob("*.DATA")) + sorted(root.rglob("*.data"))
    if not candidates:
        raise FileNotFoundError("Model Z archive contains no .DATA deck")
    ranked = []
    for path in candidates:
        try:
            meta = parse_deck(path)
            ranked.append((len(meta.wells), meta.dimensions is not None, path, meta))
        except Exception:
            continue
    if not ranked:
        raise RuntimeError("none of the .DATA files could be parsed")
    ranked.sort(key=lambda x: (x[1], x[0], x[2].stat().st_size), reverse=True)
    _, _, path, meta = ranked[0]
    return path, {"dimensions": meta.dimensions, "well_count": len(meta.wells)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("--extract-to", type=Path)
    args = ap.parse_args()
    target = args.extract_to or Path(tempfile.mkdtemp(prefix="model-z-"))
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as zf:
        zf.extractall(target)
    deck, meta = find_root_deck(target)
    schedules = sorted(str(p.relative_to(target)) for p in target.rglob("*schedule*.inc"))
    includes = sorted(str(p.relative_to(target)) for p in target.rglob("*.inc"))
    flow = shutil.which("flow")
    opm_version = None
    if flow:
        proc = subprocess.run([flow, "--version"], capture_output=True, text=True, check=False)
        opm_version = (proc.stdout or proc.stderr).strip().splitlines()[:3]
    try:
        import opm.io.ecl as opm_ecl
        opm_python = sorted(name for name in dir(opm_ecl) if not name.startswith("_"))
    except Exception as exc:
        opm_python = [f"unavailable:{type(exc).__name__}:{exc}"]
    result = {
        "archive": str(args.archive),
        "extract_root": str(target),
        "root_deck": str(deck.relative_to(target)),
        **meta,
        "schedule_candidates": schedules,
        "include_count": len(includes),
        "include_examples": includes[:30],
        "opm_version": opm_version,
        "opm_python_symbols": opm_python,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
