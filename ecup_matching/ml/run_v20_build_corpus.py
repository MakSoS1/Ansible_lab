"""Materialize a source-aware v20 gold corpus from human, prepared weak and admitted generated rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .v20_corpus import balanced_sample, build_gold_corpus
from .v20_policy import V20Policy, policy_sha256


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--human", type=Path, required=True)
    p.add_argument("--historical-weak", type=Path, required=True)
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument("--forbidden-ids", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    human = pd.read_parquet(a.human)
    weak = pd.read_parquet(a.historical_weak)
    generated = pd.read_parquet(a.generated) if a.generated.exists() else pd.DataFrame()
    forbidden_frame = pd.read_parquet(a.forbidden_ids)
    forbidden_col = "id" if "id" in forbidden_frame else forbidden_frame.columns[0]
    gold, report = build_gold_corpus(
        human, weak, generated,
        forbidden_ids=set(forbidden_frame[forbidden_col].tolist()), seed=a.seed,
    )
    if a.max_rows > 0:
        gold = balanced_sample(gold, a.max_rows, seed=a.seed)
        report["capped_rows"] = int(len(gold))
        report["max_rows"] = int(a.max_rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(a.output, index=False)
    policy = V20Policy()
    report.update({
        "version": "v20-gold-corpus-v1", "policy_sha256": policy_sha256(policy),
        "sealed_gold_opened": False, "gold_rows_scored": 0,
    })
    a.output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
