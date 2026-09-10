#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SEED="${1:-42}"
OUT="${2:-runs/local/smoke}"
python -m aios_track2.cli smoke --seed "$SEED" --out "$OUT"
python -m aios_track2.cli smoke --seed "$SEED" --out "${OUT}-repeat"
python - "$OUT" "${OUT}-repeat" <<'PY'
import json, sys
from pathlib import Path
a = json.loads((Path(sys.argv[1]) / "npv.json").read_text())
b = json.loads((Path(sys.argv[2]) / "npv.json").read_text())
assert a["npv_mrub"] == b["npv_mrub"], (a, b)
print("smoke hashes match", a)
PY
