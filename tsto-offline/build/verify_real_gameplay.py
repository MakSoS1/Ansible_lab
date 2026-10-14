#!/usr/bin/env python3
"""Strict acceptance test: a living process or successful login is not gameplay."""
from pathlib import Path
import hashlib, json, re, sys

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path("_upstream_server")
report=Path("reports");report.mkdir(exist_ok=True)
log=(report/"host-game-server.log").read_text(errors="replace") if (report/"host-game-server.log").exists() else ""
towns=root/"towns"
townfiles=[p for p in towns.glob("*") if p.is_file()] if towns.is_dir() else []
gameget=re.findall(r'"GET /mh/games/bg_gameserver_plugin/protoland/[^"]* HTTP/1.1" 200',log)
gamepost=re.findall(r'"POST /mh/games/bg_gameserver_plugin/protoland/[^"]* HTTP/1.1" 200',log)
town_data=[{"file":p.name,"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in townfiles]
out={"server_login":bool(re.search(r'"GET /mh/gameplayconfig HTTP/1.1" 200',log)),
     "town_loads_http200":len(gameget),"town_saves_http200":len(gamepost),
     "saved_town_files":town_data}
(report/"gameplay-acceptance.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
if not out["server_login"]:raise SystemExit("FAIL: no gameplay config")
if not gameget:raise SystemExit("FAIL: no town load; a splash screen is NOT playable gameplay")
if not gamepost:raise SystemExit("FAIL: no city save POST; persistence unverified")
if not any(t["bytes"]>500 for t in town_data):raise SystemExit("FAIL: no nontrivial serialized saved city")
print("PASS: gameplay server handled actual town GET/POST and persisted a city")
