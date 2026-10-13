#!/usr/bin/env python3
"""TSTO offline preflight: static APK and DLC checks. No EA assets are included."""
import argparse
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path
from xml.etree import ElementTree

KNOWN_SCORPIO_SHA256 = {
 "8137faf7a4349043e6e0b0516249c673eb402cf96012c87d805c969cdccae3c2",
 "5b6bf2e4ee386a825e58c3e12aae923d66db8b6fbdd21df05b81b4e1edeb44ae",
 "9eae61f7ae7ae3a8880809e7ffab33e5ae63582d6ad2d941d539d65639a6a47a",
 "7f4d355773c8a7328c9ee405f11b8795ca95fd2dbf7755fa686ea900593cbee3"
}

def elf_alignment(blob):
    """Parse ELF PT_LOAD alignment without executing native code."""
    if blob[:4] != b"\x7fELF" or len(blob) < 64 or blob[4] not in (1, 2) or blob[5] not in (1, 2):
        return []
    order = "<" if blob[5] == 1 else ">"
    if blob[4] == 2:
        phoff = struct.unpack_from(order + "Q", blob, 32)[0]
        entsize, count = struct.unpack_from(order + "HH", blob, 54)
        minsize, alignoffset, alignformat = 56, 48, "Q"
    else:
        phoff = struct.unpack_from(order + "I", blob, 28)[0]
        entsize, count = struct.unpack_from(order + "HH", blob, 42)
        minsize, alignoffset, alignformat = 32, 28, "I"
    if not minsize <= entsize <= 4096 or count > 8192 or phoff + entsize * count > len(blob):
        return []
    aligns = []
    for i in range(count):
        base = phoff + i * entsize
        if struct.unpack_from(order + "I", blob, base)[0] == 1:
            aligns.append(struct.unpack_from(order + alignformat, blob, base + alignoffset)[0])
    return aligns

def audit_apk(apk):
    path = Path(apk)
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    report = {"filename": path.name, "bytes": path.stat().st_size, "sha256": digest,
              "manifest": False, "arm64_engine": False, "libraries": [], "warnings": []}
    with zipfile.ZipFile(path) as z:
        report["manifest"] = "AndroidManifest.xml" in z.namelist()
        for entry in z.infolist():
            if not entry.filename.startswith("lib/") or not entry.filename.endswith(".so"):
                continue
            if entry.file_size > 150 * 1024 * 1024:
                report["warnings"].append("Oversized library skipped: " + entry.filename)
                continue
            data = z.read(entry)
            sha = hashlib.sha256(data).hexdigest()
            align = elf_alignment(data)
            engine = Path(entry.filename).name.startswith("libscorpio")
            if engine and entry.filename.startswith("lib/arm64-v8a/"):
                report["arm64_engine"] = True
            report["libraries"].append({
                "file": entry.filename, "sha256": sha, "load_alignment": align,
                "supports_16kb_elf": bool(align) and all(x >= 16384 for x in align),
                "patcher_support": sha in KNOWN_SCORPIO_SHA256 if engine else None
            })
            if engine and sha not in KNOWN_SCORPIO_SHA256:
                report["warnings"].append("Unknown Scorpio hash: " + entry.filename)
    if not report["manifest"]:
        report["warnings"].append("APK does not contain AndroidManifest.xml")
    if not report["arm64_engine"]:
        report["warnings"].append("No arm64 Scorpio engine; Pixel 10 support unknown")
    if any(not lib["supports_16kb_elf"] for lib in report["libraries"]):
        report["warnings"].append("Native 16-KiB memory-page compatibility needs device testing")
    return report

def parse_dlc_index(blob):
    if blob.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            xmls = [name for name in z.namelist() if name.lower().endswith(".xml")]
            if not xmls:
                raise ValueError("ZIP has no XML index")
            blob = z.read(xmls[0])
    root = ElementTree.fromstring(blob)
    out = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "Package":
            continue
        attrs = {c.tag.rsplit("}",1)[-1]: (c.get("val") or c.get("name") or (c.text or "").strip()) for c in node}
        name = attrs.get("FileName")
        if name:
            out.append({
                "path": name.replace(":", "/"),
                "language": attrs.get("Language", "").lower(),
                "tier": node.get("tier", "").lower(),
                "bytes_estimate": int(attrs.get("FileSize", "0") or 0)
            })
    return out

def plan_dlc(paths, languages={"ru", "en"}, tiers={"100", "retina", "mp3", "wav"}):
    selected = {}
    excluded = 0
    for path in paths:
        for item in parse_dlc_index(Path(path).read_bytes()):
            lang_ok = item["language"] in languages | {"", "all", "none", "common"}
            tier_ok = item["tier"] in tiers | {"", "all", "none", "common"}
            if lang_ok and tier_ok:
                selected[item["path"]] = item
            else:
                excluded += 1
    return {"languages": sorted(languages), "tiers": sorted(tiers),
            "selected_count": len(selected), "excluded_count": excluded,
            "compressed_bytes_estimate": sum(x["bytes_estimate"] for x in selected.values()),
            "selected": sorted(selected.values(), key=lambda x: x["path"]),
            "warning": "A planning manifest only: real client DLC requests must be validated before any deletion"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path)
    subs = p.add_subparsers(dest="command", required=True)
    a = subs.add_parser("audit-apk")
    a.add_argument("apk")
    d = subs.add_parser("plan-dlc")
    d.add_argument("indexes", nargs="+")
    d.add_argument("--languages", default="ru,en")
    d.add_argument("--tiers", default="100,retina,mp3,wav")
    x = p.parse_args()
    result = audit_apk(x.apk) if x.command == "audit-apk" else plan_dlc(
        x.indexes, set(x.languages.split(",")), set(x.tiers.split(",")))
    data = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if x.output:
        x.output.write_text(data, encoding="utf-8")
    print(data)

if __name__ == "__main__":
    main()
