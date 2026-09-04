from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path


WELL_KEYWORDS = (
    "WELSPECS",
    "COMPDAT",
    "WCONPROD",
    "WCONINJE",
    "WELOPEN",
    "WELTARG",
    "WGRUPCON",
    "WPIMULT",
    "WECON",
    "WELSEGS",
    "COMPSEGS",
)

FLUID_KEYWORDS = (
    "DENSITY",
    "PVTW",
    "PVCDO",
    "PVDO",
    "PVTO",
    "PVDG",
    "PVTG",
)


def strip_comments(text: str) -> str:
    return re.sub(r"--[^\n]*", "", text)


def records_for_keyword(text: str, keyword: str) -> list[str]:
    clean = strip_comments(text)
    lines = clean.splitlines()
    records: list[str] = []
    in_block = False
    buffer: list[str] = []
    for raw in lines:
        line = raw.strip()
        upper = line.upper()
        if not in_block:
            if upper == keyword or upper.startswith(keyword + " "):
                in_block = True
            continue
        if line == "/":
            if buffer:
                records.append(" ".join(buffer))
                buffer = []
            in_block = False
            continue
        if not line:
            continue
        buffer.append(line)
        while "/" in " ".join(buffer):
            joined = " ".join(buffer)
            head, tail = joined.split("/", 1)
            if head.strip():
                records.append(head.strip() + " /")
            buffer = [tail.strip()] if tail.strip() else []
    return records


def first_quoted(record: str) -> str | None:
    match = re.search(r"'([^']+)'", record)
    return match.group(1) if match else None


def diagnose(root: Path) -> dict[str, object]:
    by_keyword: dict[str, set[str]] = defaultdict(set)
    welspec_records: list[str] = []
    compdat_records: list[str] = []
    fluid_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".data", ".inc"}
    )
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for keyword in WELL_KEYWORDS:
            records = records_for_keyword(text, keyword)
            if keyword == "WELSPECS":
                welspec_records.extend(records)
            elif keyword == "COMPDAT":
                compdat_records.extend(records)
            for record in records:
                name = first_quoted(record)
                if name and "*" not in name:
                    by_keyword[keyword].add(name)
        for keyword in FLUID_KEYWORDS:
            for record in records_for_keyword(text, keyword):
                fluid_records[keyword].append(
                    {"file": str(path.relative_to(root)), "record": record}
                )
    all_names = set().union(*by_keyword.values()) if by_keyword else set()
    declared = by_keyword.get("WELSPECS", set())
    return {
        "files_scanned": [str(p.relative_to(root)) for p in files],
        "counts": {key: len(by_keyword.get(key, set())) for key in WELL_KEYWORDS},
        "welspec_names": sorted(declared),
        "all_well_names": sorted(all_names),
        "all_well_count": len(all_names),
        "used_but_not_welspec": sorted(all_names - declared),
        "welspec_records": welspec_records,
        "compdat_records": compdat_records,
        "fluid_property_records": dict(fluid_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="aios-model-z-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(args.archive) as zf:
            zf.extractall(root)
        print(json.dumps(diagnose(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
