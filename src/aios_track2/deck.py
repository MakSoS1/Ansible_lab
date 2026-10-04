from __future__ import annotations

import argparse
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Well:
    name: str
    group: str
    i: int
    j: int
    phase: str


@dataclass(frozen=True)
class DeckMetadata:
    dimensions: tuple[int, int, int]
    wells: tuple[Well, ...]
    source: str

    @property
    def well_count(self) -> int:
        return len(self.wells)


@dataclass(frozen=True)
class WellGraph:
    wells: tuple[Well, ...]
    edges: tuple[tuple[str, str], ...]
    weights: tuple[float, ...]
    clusters: tuple[tuple[str, ...], ...]


def _tokenize_records(text: str) -> list[str]:
    cleaned: list[str] = []
    for raw in text.splitlines():
        line = raw.split("--", 1)[0].strip()
        if line:
            cleaned.append(line)
    return cleaned


def _parse_dimens(lines: list[str]) -> tuple[int, int, int] | None:
    for index, line in enumerate(lines):
        if line.split()[0] == "DIMENS":
            buffer: list[str] = []
            for follow in lines[index + 1 :]:
                buffer.append(follow.replace("/", " "))
                if "/" in follow:
                    break
            numbers = [int(float(token)) for token in " ".join(buffer).split() if token]
            if len(numbers) >= 3:
                return numbers[0], numbers[1], numbers[2]
    return None


def _parse_welspecs(lines: list[str]) -> list[Well]:
    wells: list[Well] = []
    capturing = False
    for line in lines:
        token = line.split()[0].lstrip("'").rstrip("'")
        if token == "WELSPECS":
            capturing = True
            continue
        if capturing:
            if line.strip() == "/" or line.startswith("/"):
                capturing = False
                continue
            if token in {"COMPDAT", "WCONPROD", "WCONINJE", "DATES", "WELLIST", "END"}:
                capturing = False
                continue
            parts = [item.strip("'") for item in line.replace("/", " ").split() if item.strip("'")]
            if len(parts) >= 4:
                wells.append(
                    Well(
                        name=parts[0],
                        group=parts[1],
                        i=int(float(parts[2])),
                        j=int(float(parts[3])),
                        phase=parts[5] if len(parts) > 5 else "OIL",
                    )
                )
    unique: dict[str, Well] = {}
    order: list[str] = []
    for well in wells:
        if well.name not in unique:
            unique[well.name] = well
            order.append(well.name)
    return [unique[name] for name in order]


def _read_text_from_path(path: Path) -> str:
    return path.read_text(encoding="latin-1")


def _read_include(current: Path, target: str, seen: set[Path]) -> str:
    include_path = (current.parent / target.strip().strip("'")).resolve()
    if include_path in seen:
        raise ValueError(f"include cycle at {include_path}")
    if include_path.exists():
        return _expand_includes(_read_text_from_path(include_path), include_path, seen | {include_path})
    return ""


def _expand_includes(text: str, current: Path, seen: set[Path]) -> str:
    lines = text.splitlines()
    expanded: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].split("--", 1)[0].strip()
        if stripped.split()[:1] == ["INCLUDE"]:
            target = ""
            index += 1
            while index < len(lines):
                candidate = lines[index].split("--", 1)[0].strip().strip("'").rstrip("/").strip()
                if candidate:
                    target = candidate
                    break
                index += 1
            if target:
                expanded.append(_read_include(current, target, seen))
        else:
            expanded.append(lines[index])
        index += 1
    return "\n".join(expanded)


def _parse_text(text: str, source: str) -> DeckMetadata:
    lines = _tokenize_records(text)
    dimensions = _parse_dimens(lines) or (0, 0, 0)
    wells = tuple(_parse_welspecs(lines))
    return DeckMetadata(dimensions=dimensions, wells=wells, source=source)


def parse_deck(path: Path) -> DeckMetadata:
    path = path.resolve()
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            data_name = next(name for name in archive.namelist() if name.lower().endswith(".data"))
            sch_name = next((name for name in archive.namelist() if name.lower().endswith("_sch.inc")), None)
            chunks = [archive.read(data_name).decode("latin-1")]
            if sch_name:
                chunks.append(archive.read(sch_name).decode("latin-1"))
            return _parse_text("\n".join(chunks), str(path))
    text = _expand_includes(_read_text_from_path(path), path, {path})
    return _parse_text(text, str(path))


def build_well_graph(metadata: DeckMetadata, radius_m: float, cell_size_m: float = 100.0) -> WellGraph:
    wells = tuple(sorted(metadata.wells, key=lambda well: well.name))
    edges: list[tuple[str, str]] = []
    weights: list[float] = []
    for left in wells:
        for right in wells:
            if left.name == right.name:
                continue
            distance = math.hypot((left.i - right.i) * cell_size_m, (left.j - right.j) * cell_size_m)
            if distance <= radius_m:
                edges.append((left.name, right.name))
                weights.append(1.0 / max(distance, cell_size_m))
    clusters = _cluster_wells(wells, k=min(6, max(1, len(wells))))
    return WellGraph(wells=wells, edges=tuple(edges), weights=tuple(weights), clusters=tuple(clusters))


def _cluster_wells(wells: tuple[Well, ...], k: int) -> list[tuple[str, ...]]:
    if not wells:
        return []
    centroids: list[tuple[float, float]] = [
        (
            float(wells[index * (len(wells) - 1) // max(k - 1, 1)].i),
            float(wells[index * (len(wells) - 1) // max(k - 1, 1)].j),
        )
        for index in range(k)
    ]
    assignment = [0] * len(wells)
    for _ in range(8):
        buckets: list[list[Well]] = [[] for _ in range(k)]
        for well in wells:
            nearest = min(
                range(k),
                key=lambda idx: math.hypot(well.i - centroids[idx][0], well.j - centroids[idx][1]),
            )
            buckets[nearest].append(well)
        centroids = [
            (
                sum(well.i for well in bucket) / len(bucket) if bucket else centroids[idx][0],
                sum(well.j for well in bucket) / len(bucket) if bucket else centroids[idx][1],
            )
            for idx, bucket in enumerate(buckets)
        ]
        assignment = [
            min(range(k), key=lambda idx: math.hypot(well.i - centroids[idx][0], well.j - centroids[idx][1]))
            for well in wells
        ]
    grouped: dict[int, list[str]] = {idx: [] for idx in range(k)}
    for well, cluster in zip(wells, assignment, strict=True):
        grouped[cluster].append(well.name)
    return [tuple(names) for names in grouped.values() if names]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("deck", type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    metadata = parse_deck(args.deck)
    payload = {
        "dimensions": list(metadata.dimensions),
        "well_count": metadata.well_count,
        "wells": [well.name for well in metadata.wells],
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
