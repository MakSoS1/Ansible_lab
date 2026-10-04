from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Well:
    name: str
    i: int
    j: int
    phase: str


@dataclass(frozen=True, slots=True)
class DeckMetadata:
    dimensions: tuple[int, int, int] | None
    wells: tuple[Well, ...]


def _strip_comments(text: str) -> str:
    return re.sub(r"--[^\n]*", "", text)


def parse_deck_text(text: str) -> DeckMetadata:
    clean = _strip_comments(text)
    md = re.search(r"\bDIMENS\b\s*\n?\s*(\d+)\s+(\d+)\s+(\d+)\s*/", clean, re.I)
    dims = tuple(map(int, md.groups())) if md else None
    wells: dict[str, Well] = {}
    for block in re.finditer(r"\bWELSPECS\b(.*?)(?=\n\s*/|\n\s*[A-Z][A-Z0-9_]+\b|\Z)", clean, re.I | re.S):
        body = block.group(1)
        pattern = re.compile(r"'([^']+)'\s+'[^']*'\s+(\d+)\s+(\d+)\s+[^/]*?'([^']+)'\s*/", re.I)
        for m in pattern.finditer(body):
            name, i, j, phase = m.groups()
            wells[name] = Well(name, int(i), int(j), phase.upper())
    return DeckMetadata(dims, tuple(sorted(wells.values(), key=lambda w: w.name)))


def parse_deck(path: Path) -> DeckMetadata:
    visited: set[Path] = set()
    texts: list[str] = []

    def visit(p: Path) -> None:
        p = p.resolve()
        if p in visited:
            return
        visited.add(p)
        text = p.read_text(encoding="utf-8", errors="ignore")
        texts.append(text)
        for match in re.finditer(r"\bINCLUDE\b\s*\n?\s*'([^']+)'\s*/", text, re.I):
            child = (p.parent / match.group(1)).resolve()
            if child.exists():
                visit(child)
    visit(path)
    return parse_deck_text("\n".join(texts))
