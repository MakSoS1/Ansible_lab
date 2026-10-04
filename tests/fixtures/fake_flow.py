import sys
from pathlib import Path

Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
Path(sys.argv[2], "CASE.UNSMRY").write_bytes(b"fixture")
print("OPM FLOW FIXTURE OK")
