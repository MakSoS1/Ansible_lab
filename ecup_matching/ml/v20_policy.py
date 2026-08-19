from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

import pandas as pd


PUBLIC_LB_ANCHORS = {
    "v14": 0.38032704703111925,
    "v12": 0.379811620418641,
    "v13B": 0.37837816527590995,
    "v7": 0.3655833314,
}


@dataclass(frozen=True)
class V20Policy:
    split_sha256: str = "aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b"
    positive_precision_lcb: float = 0.985
    negative_precision_lcb: float = 0.995
    category_precision_lcb: float = 0.970
    critical_precision_lcb: float = 0.950
    min_stratum_support: int = 100
    max_length: int = 256
    production_checkpoints: int = 1
    lambda_reason: float = 0.15
    lambda_consistency: float = 0.05
    phase_b_human_to_other: tuple[int, int] = (1, 2)
    phase_c_human_to_other: tuple[int, int] = (4, 1)
    phase_c_lr_multiplier: float = 0.35
    proxy_gain_min_strict: float = 0.005
    human_delta_min: float = -0.003
    audited_tail_delta_min: float = -0.02
    category_delta_min: float = -0.04
    scaled_mean_human_delta_min: float = 0.0
    seed: int = 2026


def policy_sha256(policy: V20Policy) -> str:
    raw = json.dumps(asdict(policy), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_fold_exclusion(frame: pd.DataFrame, forbidden_ids: Iterable[object]) -> int:
    forbidden = set(forbidden_ids)
    if not {"id1", "id2"}.issubset(frame.columns):
        raise ValueError("frame requires id1 and id2")
    leaked = frame[frame["id1"].isin(forbidden) | frame["id2"].isin(forbidden)]
    if len(leaked):
        raise ValueError(f"generated/training frame contains {len(leaked)} held-fold item rows")
    return 0


__all__ = ["PUBLIC_LB_ANCHORS", "V20Policy", "policy_sha256", "validate_fold_exclusion"]
