from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

from openpyxl import load_workbook


def inspect_archive(archive: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="aios-chdd-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
        files = sorted(p for p in root.rglob("*") if p.is_file())
        python_files: dict[str, str] = {}
        workbooks: dict[str, object] = {}
        for path in files:
            rel = str(path.relative_to(root))
            if path.suffix.lower() == ".py":
                python_files[rel] = path.read_text(encoding="utf-8", errors="replace")
            elif path.suffix.lower() in {".xlsx", ".xlsm"}:
                wb = load_workbook(path, read_only=True, data_only=False)
                workbooks[rel] = {
                    "sheets": [
                        {
                            "title": ws.title,
                            "max_row": ws.max_row,
                            "max_column": ws.max_column,
                        }
                        for ws in wb.worksheets
                    ]
                }
                wb.close()
        return {
            "archive": str(archive),
            "files": [str(p.relative_to(root)) for p in files],
            "python_files": python_files,
            "workbooks": workbooks,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_archive(args.archive), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
