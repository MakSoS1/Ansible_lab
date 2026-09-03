from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .benchmark import run_compute_benchmark
from .config import load_config
from .deck import parse_deck


def main() -> None:
    parser = argparse.ArgumentParser(prog="aios-track2")
    sub = parser.add_subparsers(dest="command", required=True)
    p_cfg = sub.add_parser("config")
    p_cfg.add_argument("path", type=Path, nargs="?", default=Path("configs/base.yaml"))
    p_deck = sub.add_parser("deck-summary")
    p_deck.add_argument("path", type=Path)
    sub.add_parser("benchmark")
    args = parser.parse_args()
    if args.command == "config":
        cfg = load_config(args.path)
        print(cfg.model_dump_json(indent=2))
    elif args.command == "deck-summary":
        meta = parse_deck(args.path)
        print(json.dumps({"dimensions": meta.dimensions, "well_count": len(meta.wells)}, indent=2))
    elif args.command == "benchmark":
        print(json.dumps(asdict(run_compute_benchmark()), indent=2, sort_keys=True))
