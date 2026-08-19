"""Progress heartbeat for every long-running phase.

A count-based interval is not enough on its own. ``build_features_v2_chunked``
over 365,654 pairs printed one line and then went silent for minutes, which
looks identical to a hung job or an idle GPU — that is exactly how a failed v7
production run presented itself.

``ProgressReporter`` therefore fires on whichever comes first: a number of
completed units, or a wall-clock interval. The time bound is what guarantees a
run can never appear dead while it is working.

Every tick carries what is needed to tell "working" from "stuck": phase,
done/total, percent, elapsed, rolling throughput, ETA, resident and peak RSS,
and CUDA memory when a device is in use.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any


def _rss_bytes() -> tuple[int, int]:
    """Current and peak resident set size, best effort and never raising."""
    current = 0
    peak = 0
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS reports bytes.
        peak = int(usage) * (1024 if os.uname().sysname == "Linux" else 1)
    except Exception:
        peak = 0
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
        current = pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        current = peak
    return current, peak


def _cuda_payload() -> dict[str, int]:
    try:
        import torch
    except Exception:
        return {}
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return {}
    try:
        return {
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
    except Exception:
        return {}


def format_eta(seconds: float) -> str:
    if seconds < 0 or seconds != seconds or seconds == float("inf"):
        return "unknown"
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class ProgressReporter:
    """Emit a bounded-silence heartbeat for one phase of work."""

    def __init__(
        self,
        phase: str,
        total: int,
        *,
        every_units: int = 10_000,
        every_seconds: float = 30.0,
        emit=print,
        clock=time.perf_counter,
    ):
        if not phase:
            raise ValueError("phase must be a non-empty name")
        if total < 0:
            raise ValueError("total must not be negative")
        if every_units <= 0 and every_seconds <= 0:
            raise ValueError("at least one of every_units or every_seconds must be positive")
        self.phase = str(phase)
        self.total = int(total)
        self._every_units = int(every_units)
        self._every_seconds = float(every_seconds)
        self._emit = emit
        self._clock = clock
        self._started = clock()
        self._last_tick = self._started
        self._last_done = 0
        self.ticks = 0

    def _payload(self, done: int, **extra: Any) -> dict[str, Any]:
        now = self._clock()
        elapsed = max(now - self._started, 1e-9)
        rate = done / elapsed
        window = max(now - self._last_tick, 1e-9)
        recent = (done - self._last_done) / window
        remaining = max(self.total - done, 0)
        eta = remaining / recent if recent > 0 else float("inf")
        current_rss, peak_rss = _rss_bytes()
        payload: dict[str, Any] = {
            "phase": self.phase,
            "done": int(done),
            "total": int(self.total),
            "percent": round(100.0 * done / self.total, 2) if self.total else 100.0,
            "elapsed_seconds": round(elapsed, 2),
            "units_per_second": round(rate, 2),
            "recent_units_per_second": round(recent, 2),
            "eta": format_eta(eta),
            "eta_seconds": None if eta == float("inf") else round(eta, 1),
            "rss_bytes": current_rss,
            "peak_rss_bytes": peak_rss,
            **_cuda_payload(),
            **extra,
        }
        return payload

    def update(self, done: int, **extra: Any) -> bool:
        """Report if enough units or enough time has passed. Returns True if emitted."""
        now = self._clock()
        by_units = self._every_units > 0 and (done - self._last_done) >= self._every_units
        by_time = self._every_seconds > 0 and (now - self._last_tick) >= self._every_seconds
        if not (by_units or by_time):
            return False
        payload = self._payload(done, **extra)
        self._last_tick = now
        self._last_done = done
        self.ticks += 1
        self._emit(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        return True

    def finish(self, done: int | None = None, **extra: Any) -> None:
        total = self.total if done is None else done
        payload = self._payload(total, final=True, **extra)
        self.ticks += 1
        self._emit(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
