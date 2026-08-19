from __future__ import annotations

import math
from typing import Mapping


EXPECTED_PUBLIC_ORDER = ("v14", "v12", "v13B", "v7")


def _ordered(values: Mapping[str, float], *, higher_is_better: bool) -> list[str]:
    missing = set(EXPECTED_PUBLIC_ORDER) - set(values)
    if missing:
        raise ValueError(f"proxy axis missing anchors: {sorted(missing)}")
    for name in EXPECTED_PUBLIC_ORDER:
        if not math.isfinite(float(values[name])):
            raise ValueError(f"proxy value for {name} is not finite")
    return sorted(
        EXPECTED_PUBLIC_ORDER,
        key=lambda name: float(values[name]),
        reverse=bool(higher_is_better),
    )


def _strictly_separated(values: Mapping[str, float], order: list[str], *, higher_is_better: bool) -> bool:
    for left, right in zip(order, order[1:]):
        a, b = float(values[left]), float(values[right])
        if math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12):
            return False
        if higher_is_better and not a > b:
            return False
        if not higher_is_better and not a < b:
            return False
    return True


def calibrate_proxy_axes(axes: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    report: dict[str, object] = {}
    promotable: list[str] = []
    for name in sorted(axes):
        spec = axes[name]
        if "values" not in spec or "higher_is_better" not in spec:
            raise ValueError(f"proxy axis {name!r} requires values and higher_is_better")
        values = {str(k): float(v) for k, v in dict(spec["values"]).items()}
        higher = bool(spec["higher_is_better"])
        observed = _ordered(values, higher_is_better=higher)
        exact = observed == list(EXPECTED_PUBLIC_ORDER) and _strictly_separated(
            values, observed, higher_is_better=higher
        )
        report[name] = {
            "promotable": bool(exact),
            "observed_order": observed,
            "expected_order": list(EXPECTED_PUBLIC_ORDER),
            "higher_is_better": higher,
            "values": values,
        }
        if exact:
            promotable.append(name)
    return {"axes": report, "promotable_axes": promotable}


__all__ = ["EXPECTED_PUBLIC_ORDER", "calibrate_proxy_axes"]
