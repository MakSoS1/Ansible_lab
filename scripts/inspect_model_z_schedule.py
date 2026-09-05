from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

from aios_track2.schedule_structure import inspect_schedule_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="aios-model-z-schedule-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(args.archive) as zf:
            zf.extractall(root)
        reports: list[dict[str, object]] = []
        for path in sorted(root.rglob("*.inc")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "WCONPROD" not in text.upper() and "WCONINJE" not in text.upper():
                continue
            info = inspect_schedule_text(text)
            reports.append(
                {
                    "path": str(path.relative_to(root)),
                    "explicit_date_count": len(info.explicit_dates),
                    "first_explicit_date": info.explicit_dates[0].isoformat() if info.explicit_dates else None,
                    "last_explicit_date": info.explicit_dates[-1].isoformat() if info.explicit_dates else None,
                    "tstep_count": len(info.tstep_days),
                    "tstep_total_days": sum(info.tstep_days),
                    "producer_blocks": info.producer_blocks,
                    "injector_blocks": info.injector_blocks,
                    "producer_records": info.producer_records,
                    "injector_records": info.injector_records,
                    "producer_modes": info.producer_modes,
                    "injector_modes": info.injector_modes,
                    "sample_dates": [d.isoformat() for d in info.explicit_dates[:20]],
                }
            )
        print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
