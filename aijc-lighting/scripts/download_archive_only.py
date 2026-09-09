from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from download_mailru import DEFAULT_FILENAME, DEFAULT_URL, PROJECT_ROOT, download_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--zip-path", type=Path, default=PROJECT_ROOT / DEFAULT_FILENAME)
    args = parser.parse_args()

    if args.zip_path.exists() and zipfile.is_zipfile(args.zip_path):
        print(f"Using existing valid archive {args.zip_path} ({args.zip_path.stat().st_size} bytes)")
        return 0
    args.zip_path.unlink(missing_ok=True)
    download_file(args.url, args.zip_path)
    if not zipfile.is_zipfile(args.zip_path):
        raise RuntimeError(f"Downloaded file is not a valid ZIP: {args.zip_path}")
    print(f"Archive-only stage verified ZIP: {args.zip_path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
