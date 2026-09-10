from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from aios_track2.hfstore import RunManifest, upload_run_directory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("manifest", type=Path)
    args = ap.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest = RunManifest(**data)
    upload_run_directory(args.directory, manifest, token=os.environ.get("HF_TOKEN", ""))


if __name__ == "__main__":
    main()
