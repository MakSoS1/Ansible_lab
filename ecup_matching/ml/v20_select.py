from __future__ import annotations

from typing import Mapping

from .v20_promotion import evaluate_candidate


def _category_deltas(control: Mapping[str, object], candidate: Mapping[str, object]) -> dict[str, float]:
    base = dict(control["human_per_category_ap"])
    got = dict(candidate["human_per_category_ap"])
    common = sorted(set(base) & set(got))
    if not common:
        raise ValueError("candidate/control share no human categories")
    return {name: float(got[name]) - float(base[name]) for name in common}


def _gate(control, candidate, *, proxy_promotable: bool, best_anchor_proxy: float) -> dict[str, object]:
    candidate_proxy = float(candidate["proxy_metrics"]["macro_average_precision"])
    proxy_delta = candidate_proxy - float(control["proxy_metrics"]["macro_average_precision"])
    human_delta = float(candidate["human_macro_average_precision"]) - float(control["human_macro_average_precision"])
    tail_delta = float(candidate["proxy_metrics"]["tail_macro_average_precision"]) - float(control["proxy_metrics"]["tail_macro_average_precision"])
    gate = evaluate_candidate(
        proxy_delta=proxy_delta, human_delta=human_delta, audited_tail_delta=tail_delta,
        category_deltas=_category_deltas(control, candidate),
        proxy_axis_promotable=proxy_promotable,
    )
    anchor_gate = candidate_proxy > float(best_anchor_proxy) + 1e-12
    gate["anchor_gate"] = bool(anchor_gate)
    gate["best_anchor_proxy"] = float(best_anchor_proxy)
    gate["promote"] = bool(gate["promote"] and anchor_gate)
    gate["candidate"] = str(candidate["candidate"])
    gate["proxy_value"] = candidate_proxy
    gate["human_value"] = float(candidate["human_macro_average_precision"])
    return gate


def select_stage1(
    control: Mapping[str, object],
    candidates: Mapping[str, Mapping[str, object]],
    *,
    proxy_promotable: bool,
    best_anchor_proxy: float,
) -> dict[str, object]:
    expected = {"data-only", "rationale"}
    if set(candidates) != expected:
        raise ValueError(f"stage1 candidates must be exactly {sorted(expected)}")
    gates = {
        name: _gate(control, value, proxy_promotable=proxy_promotable, best_anchor_proxy=best_anchor_proxy)
        for name, value in candidates.items()
    }
    passing = [name for name, gate in gates.items() if gate["promote"]]
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda name: (
                float(candidates[name]["proxy_metrics"]["macro_average_precision"]),
                float(candidates[name]["human_macro_average_precision"]),
                name,
            ),
        )
    return {
        "version": "v20-stage1-selection-v2", "selected": selected,
        "promote": selected is not None, "gates": gates,
        "best_anchor_proxy": float(best_anchor_proxy),
        "replay_candidate": None if selected is None else ("replay-rationale" if selected == "rationale" else "replay-data"),
    }


def select_replay(
    control: Mapping[str, object],
    keeper: Mapping[str, object],
    replay: Mapping[str, object],
    *,
    proxy_promotable: bool,
    best_anchor_proxy: float,
) -> dict[str, object]:
    gate = _gate(control, replay, proxy_promotable=proxy_promotable, best_anchor_proxy=best_anchor_proxy)
    keeper_proxy = float(keeper["proxy_metrics"]["macro_average_precision"])
    replay_proxy = float(replay["proxy_metrics"]["macro_average_precision"])
    improves_keeper = replay_proxy > keeper_proxy + 1e-12
    selected = str(replay["candidate"]) if gate["promote"] and improves_keeper else str(keeper["candidate"])
    return {
        "version": "v20-replay-selection-v2", "selected": selected,
        "replay_gate": gate, "replay_improves_keeper_proxy": bool(improves_keeper),
        "best_anchor_proxy": float(best_anchor_proxy),
        "keeper_candidate": str(keeper["candidate"]), "keeper_proxy": keeper_proxy,
        "replay_proxy": replay_proxy,
    }


__all__ = ["select_stage1", "select_replay"]
