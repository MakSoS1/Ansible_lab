"""Compare the item population we train on with the one we are scored on.

The retrieval audit returned a result nobody had measured before: the 711,304
items behind `matches.parquet` and the 12,384,610 items behind
`matches_llm.parquet` share **zero** ids. The two supervision sources do not
describe the same products, which is why weak label quality was never
verifiable against human truth and never could have been.

That leaves one question standing. `items.parquet` holds 13,397,761 items,
human labelling covers 711,304 of them, and every candidate so far was trained
and validated inside that 5.3% slice. If the rest of the universe is a
materially different population — sparser attributes, shorter names, a
different category mix — then a local Macro AP of `0.706` and a leaderboard of
`0.380` are not in conflict at all: they are two different populations, and
every architecture change was tuned on the wrong one.

This driver reads `items.parquet` in row-group batches and profiles three
disjoint populations — human-labelled, weak-labelled, and the remainder that
belongs to neither — on the axes a matcher actually consumes: category mix,
name length, attribute payload size and attribute key count.

No labels are read and no split is touched. This is description, not fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


_STARTED = time.perf_counter()


def _emit(phase: str, payload: dict[str, object]) -> None:
    text = json.dumps(
        {"phase": phase, "elapsed_seconds": round(time.perf_counter() - _STARTED, 2), **payload},
        ensure_ascii=False,
        sort_keys=True,
    )
    print(text[:6000], flush=True)


class _Accumulator:
    """Streaming profile of one population; never holds the whole slice."""

    def __init__(self, label: str):
        self.label = label
        self.rows = 0
        self.category_counts: dict[str, int] = {}
        self.name_chars = 0
        self.name_empty = 0
        self.attr_chars = 0
        self.attr_missing = 0
        self.attr_keys = 0
        self.attr_key_samples: list[int] = []
        self.name_length_samples: list[int] = []

    def add(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.rows += int(len(frame))
        counts = frame["category"].astype(str).value_counts()
        for category, count in counts.items():
            self.category_counts[str(category)] = self.category_counts.get(str(category), 0) + int(
                count
            )

        names = frame["name"].fillna("").astype(str)
        lengths = names.str.len().to_numpy()
        self.name_chars += int(lengths.sum())
        self.name_empty += int((lengths == 0).sum())

        attributes = frame["attributes"].fillna("").astype(str)
        attr_lengths = attributes.str.len().to_numpy()
        self.attr_chars += int(attr_lengths.sum())
        self.attr_missing += int((attr_lengths <= 2).sum())
        # The payload is JSON; counting separators is a cheap, allocation-free
        # proxy for key count that does not depend on the exact schema.
        keys = attributes.str.count('":').to_numpy()
        self.attr_keys += int(keys.sum())

        # Reservoir-free deterministic sampling: first N of each batch is
        # enough for quantiles at these population sizes.
        if len(self.name_length_samples) < 200_000:
            self.name_length_samples.extend(lengths[:2_000].tolist())
            self.attr_key_samples.extend(keys[:2_000].tolist())

    def report(self) -> dict[str, object]:
        rows = max(self.rows, 1)
        names = np.asarray(self.name_length_samples, dtype=np.float64)
        keys = np.asarray(self.attr_key_samples, dtype=np.float64)
        total_category = sum(self.category_counts.values()) or 1
        return {
            "label": self.label,
            "rows": self.rows,
            "name_chars_mean": self.name_chars / rows,
            "name_empty_fraction": self.name_empty / rows,
            "attribute_chars_mean": self.attr_chars / rows,
            "attribute_missing_fraction": self.attr_missing / rows,
            "attribute_keys_mean": self.attr_keys / rows,
            "name_length_median": float(np.median(names)) if names.size else None,
            "name_length_p10": float(np.percentile(names, 10)) if names.size else None,
            "name_length_p90": float(np.percentile(names, 90)) if names.size else None,
            "attribute_keys_median": float(np.median(keys)) if keys.size else None,
            "attribute_keys_p90": float(np.percentile(keys, 90)) if keys.size else None,
            "category_share": {
                category: count / total_category
                for category, count in sorted(
                    self.category_counts.items(), key=lambda kv: -kv[1]
                )
            },
        }


def run_population_audit(
    *,
    items_path: Path,
    human_items_path: Path,
    weak_matches_path: Path | None,
    output_path: Path,
) -> dict[str, object]:
    human_ids = np.unique(
        pd.read_parquet(human_items_path, columns=["id"])["id"].to_numpy().astype(np.int64)
    )
    _emit("human-ids", {"count": int(human_ids.shape[0])})

    if weak_matches_path is not None and Path(weak_matches_path).exists():
        weak = pd.read_parquet(weak_matches_path, columns=["id1", "id2"])
        weak_ids = np.union1d(
            weak["id1"].to_numpy().astype(np.int64), weak["id2"].to_numpy().astype(np.int64)
        )
        del weak
    else:
        weak_ids = np.array([], dtype=np.int64)
    _emit("weak-ids", {"count": int(weak_ids.shape[0])})

    accumulators = {
        "human": _Accumulator("human"),
        "weak": _Accumulator("weak"),
        "unlabelled": _Accumulator("unlabelled"),
    }

    handle = pq.ParquetFile(items_path)
    total_groups = handle.metadata.num_row_groups
    seen = 0
    for index in range(total_groups):
        frame = handle.read_row_group(
            index, columns=["id", "name", "attributes", "category"]
        ).to_pandas()
        ids = frame["id"].to_numpy().astype(np.int64)
        is_human = np.isin(ids, human_ids, assume_unique=False)
        is_weak = np.isin(ids, weak_ids, assume_unique=False)
        accumulators["human"].add(frame.loc[is_human])
        accumulators["weak"].add(frame.loc[is_weak & ~is_human])
        accumulators["unlabelled"].add(frame.loc[~is_weak & ~is_human])
        seen += int(len(frame))
        _emit(
            "items-scan",
            {
                "row_group": index + 1,
                "row_groups": total_groups,
                "rows_seen": seen,
                "human": accumulators["human"].rows,
                "weak": accumulators["weak"].rows,
                "unlabelled": accumulators["unlabelled"].rows,
            },
        )
        del frame

    report: dict[str, object] = {
        "items_rows": seen,
        "human_ids": int(human_ids.shape[0]),
        "weak_ids": int(weak_ids.shape[0]),
        "populations": {name: acc.report() for name, acc in accumulators.items()},
    }
    report["coverage"] = {
        "human_fraction_of_universe": accumulators["human"].rows / max(seen, 1),
        "weak_fraction_of_universe": accumulators["weak"].rows / max(seen, 1),
        "unlabelled_fraction_of_universe": accumulators["unlabelled"].rows / max(seen, 1),
    }
    _emit("coverage", report["coverage"])
    for name, acc in accumulators.items():
        _emit(f"population-{name}", acc.report())

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--human-items", type=Path, required=True)
    parser.add_argument("--weak-matches", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_population_audit(
        items_path=args.items,
        human_items_path=args.human_items,
        weak_matches_path=args.weak_matches,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
