#!/usr/bin/env python3
"""Report actual gameasset HTTP 404s and propose bounded archive groups.

Reads diagnostic text and archive metadata only; never downloads copyrighted
DLC. The output is evidence for selecting future *small* private tests,
NOT a playable-offline or package-completeness assertion.
"""
import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from urllib.parse import unquote

HTTP = re.compile(r'\bGET\s+(\S+)\s+HTTP/1\.1[^\n]*?\b(200|206|404)\b')
SCORPIO_FAILURE = re.compile(
    r'ScorpioDLCIndex::AddPackageToDownload.*?file:\s+simpsons4:([^:\s]+):([^\s]+\.zip)',
    re.I
)


def archive_map(metadata):
    if not metadata:
        return {}
    rows = metadata.get("completed", metadata.get("files", []))
    return {record["name"]: {"bytes": int(record.get("bytes", record.get("size", 0))),
                              "sha1": record.get("sha1", "")}
            for record in rows if record["name"].startswith("cdn/")
            and record["name"].endswith(".7z")}


def audit(http_text, native_text="", metadata=None, budget=500_000_000):
    if budget < 0:
        raise ValueError("Download budget must not be negative")
    requests = OrderedDict()
    for match in HTTP.finditer(http_text):
        target, code = match.groups()
        if "/gameassets/" not in target:
            continue
        relative = unquote(target.rsplit("/gameassets/", 1)[1].split("?", 1)[0])
        if not relative or ".." in Path(relative).parts:
            continue
        entry = requests.setdefault(relative, {"path": relative, "http_200": 0,
                                                "http_206": 0, "http_404": 0})
        entry["http_" + code] += 1
    missing = [row for row in requests.values()
               if row["http_404"] and not row["http_200"] and not row["http_206"]]
    native = [{"directory": x.group(1), "filename": x.group(2)}
              for x in SCORPIO_FAILURE.finditer(native_text)]
    catalog = archive_map(metadata)
    candidates = OrderedDict()
    for row in missing:
        name = row["path"]
        if name.startswith("dlc/"):
            row["kind"] = "index"
            continue
        if "/" not in name:
            row["kind"] = "unknown"
            continue
        folder = name.split("/", 1)[0]
        item = "cdn/" + folder + ".7z"
        row["kind"] = "asset"
        row["archive"] = item
        row["archive_available"] = item in catalog
        if item in catalog:
            row["archive_bytes"] = catalog[item]["bytes"]
            if item not in candidates:
                candidates[item] = {"name": item, **catalog[item]}
    selected = []
    deferred = []
    used = 0
    for name, row in candidates.items():
        if used + row["bytes"] <= budget and row["bytes"] > 0:
            selected.append(row)
            used += row["bytes"]
        else:
            deferred.append(row)
    return {"missing_unique_paths": len(missing),
            "missing_requests": missing,
            "scorpio_package_failures": native,
            "budget_bytes": budget,
            "selected_archives": selected,
            "selected_total_bytes": used,
            "deferred_archives": deferred,
            "note": "Selection is not proof that DLC packages or Springfield can run."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("http_log", type=Path)
    p.add_argument("--native-log", type=Path)
    p.add_argument("--metadata-json", type=Path)
    p.add_argument("--max-archive-mb", type=float, default=500)
    p.add_argument("--output", type=Path, default=Path("reports/bootstrap-gaps.json"))
    a = p.parse_args()
    metadata = json.loads(a.metadata_json.read_text()) if a.metadata_json else None
    data = audit(a.http_log.read_text(errors="replace"),
                 a.native_log.read_text(errors="replace") if a.native_log else "",
                 metadata, int(a.max_archive_mb * 1_000_000))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps({"missing_unique_paths": data["missing_unique_paths"],
                      "selected_total_bytes": data["selected_total_bytes"],
                      "selected_archives": [row["name"] for row in data["selected_archives"]],
                      "scorpio_package_failures": data["scorpio_package_failures"]},
                     indent=2))


if __name__ == "__main__":
    main()
