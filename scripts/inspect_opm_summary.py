from __future__ import annotations

import argparse
import json
from pathlib import Path

from opm.io.ecl import ESmry


def find_summary_case(root: Path) -> Path:
    esmry = sorted(root.rglob("*.ESMRY"))
    if esmry:
        return esmry[0]
    smspec = sorted(root.rglob("*.SMSPEC"))
    if smspec:
        return smspec[0]
    raise FileNotFoundError(f"no ESMRY/SMSPEC beneath {root}")


def inspect(case: Path) -> dict[str, object]:
    summary = ESmry(str(case))
    keys = list(summary.keys())
    well_keys = [key for key in keys if key.startswith("W") and ":" in key]
    well_names = sorted({key.split(":", 1)[1] for key in well_keys})
    wanted_prefixes = (
        "WOPR:", "WOPT:", "WWPR:", "WWPT:", "WWIR:", "WWIT:",
        "WLPR:", "WLPT:", "WOMR:", "WOMT:", "WBHP:", "WTHP:",
    )
    relevant = [key for key in keys if key.startswith(wanted_prefixes)]
    dates = summary.dates()
    return {
        "case": str(case),
        "start_date": summary.start_date.isoformat(),
        "end_date": summary.end_date.isoformat(),
        "report_steps": len(summary),
        "key_count": len(keys),
        "well_count_from_summary_keys": len(well_names),
        "well_names": well_names,
        "relevant_keys": relevant,
        "all_keys": keys,
        "first_report_date": dates[0].isoformat() if dates else None,
        "last_report_date": dates[-1].isoformat() if dates else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(find_summary_case(args.root))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
