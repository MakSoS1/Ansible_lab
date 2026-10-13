#!/usr/bin/env python3
"""Configure a locally checked-out private TSTO server for device loopback."""
import argparse
import json
import secrets
from pathlib import Path

def configure(path: Path, host: str = "127.0.0.2", port: int = 9000) -> dict:
    if host not in ("127.0.0.1", "127.0.0.2"):
        raise ValueError("Only explicit loopback addresses are accepted")
    if not 1024 <= port <= 65535:
        raise ValueError("Only unprivileged ports are accepted")
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("secret_key", "client_secret"):
        if not data.get(key):
            data[key] = secrets.token_urlsafe(48)
    data.update({"debug": False, "http_debug": False, "server_debug_mode": False,
                 "listening_ip": host, "listening_port": port,
                 "server_url": f"http://{host}:{port}",
                 "reverse_proxy": False, "run_tutorial": True,
                 "dlc_dir": "dlc", "towns_dir": "towns", "active_town": "mytown"})
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--host", default="127.0.0.2")
    p.add_argument("--port", type=int, default=9000)
    args = p.parse_args()
    print("Configured local game server:", configure(args.config,args.host,args.port)["server_url"])
