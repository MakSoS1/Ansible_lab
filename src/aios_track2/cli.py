from __future__ import annotations

import argparse
import json
from pathlib import Path

from aios_track2.agents import run_pipeline
from aios_track2.api import write_package
from aios_track2.bakeoff import pick_winner, reports_to_dict, run_bakeoff
from aios_track2.device import resolve_device
from aios_track2.hfpub import publish_run
from aios_track2.schedule import write_schedule_inc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aios-track2")
    sub = parser.add_subparsers(dest="cmd", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--out", type=Path, default=Path("runs/local/smoke"))
    bake = sub.add_parser("bakeoff")
    bake.add_argument("--seed", type=int, default=42)
    bake.add_argument("--out", type=Path, default=Path("runs/local/bakeoff"))
    bake.add_argument("--preset", choices=["local", "m1"], default="local")
    bake.add_argument("--publish", action="store_true")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.cmd == "smoke":
        result = run_pipeline()
        write_package(result, args.out)
        write_schedule_inc(result.schedule, args.out / "wells_schedule.inc")
        print(json.dumps({"npv_mrub": result.npv_mrub, "sha": result.schedule_sha256, "backend": result.backend}))
        return 0
    if args.cmd == "bakeoff":
        device = resolve_device("auto")
        reports = run_bakeoff(seed=args.seed)
        winner = pick_winner(reports)
        args.out.mkdir(parents=True, exist_ok=True)
        payload = {
            "device": str(device),
            "winner": winner.name,
            "reports": reports_to_dict(reports),
        }
        (args.out / "bakeoff.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        result = run_pipeline()
        write_package(result, args.out / "package")
        print(json.dumps({"winner": winner.name, "npv_mrub": winner.npv_mrub, "device": str(device)}))
        if args.publish:
            publish_run(args.out)
        return 0
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("aios_track2.api:app", host=args.host, port=args.port, reload=False)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
