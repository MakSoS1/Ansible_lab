from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

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
    p_ui = sub.add_parser("ui", help="serve the Track 2 control room on verified submission artifacts")
    p_ui.add_argument("--submission", type=Path, default=Path("submission"))
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "config":
        cfg = load_config(args.path)
        print(cfg.model_dump_json(indent=2))
    elif args.command == "deck-summary":
        meta = parse_deck(args.path)
        print(json.dumps({"dimensions": meta.dimensions, "well_count": len(meta.wells)}, indent=2))
    elif args.command == "benchmark":
        from .benchmark import run_compute_benchmark

        print(json.dumps(asdict(run_compute_benchmark()), indent=2, sort_keys=True))
    elif args.command == "ui":
        from .api import create_app

        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit("install project with [api] extras to serve the UI") from exc
        app = create_app(args.submission)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
