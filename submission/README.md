# Verified Track 2 submission (MAPPO)

Source: GitHub Actions run `33925326455` (clean OPM rerun + later publish `33951759766`).

Do not edit these files by hand. Re-verify:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
raw = Path('wells_schedule.inc').read_bytes()
sha = hashlib.sha256(raw).hexdigest()
man = json.loads(Path('final-submission-manifest.json').read_text())
print(sha)
print(man['schedule_sha256'])
assert sha == man['schedule_sha256']
assert abs(man['clean_npv_mrub'] - 12475.954558553085) < 1e-9
print('ok')
PY
```
