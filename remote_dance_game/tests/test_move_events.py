from __future__ import annotations

import math

from services.move_events import build_move_markers, build_pictogram_markers


def _pose(i: int):
    pts = [{"x": 0.5, "y": 0.4, "z": 0.0, "v": 1.0} for _ in range(33)]
    phase = i / 30.0 * math.pi * 2.0
    pts[11].update(x=0.42, y=0.32)
    pts[12].update(x=0.58, y=0.32)
    pts[13].update(x=0.34, y=0.42)
    pts[14].update(x=0.66, y=0.42)
    pts[15].update(x=0.25 + 0.07 * math.sin(phase), y=0.48 - 0.16 * math.sin(phase))
    pts[16].update(x=0.75 - 0.07 * math.sin(phase), y=0.48 + 0.16 * math.sin(phase))
    pts[23].update(x=0.45, y=0.57)
    pts[24].update(x=0.55, y=0.57)
    pts[25].update(x=0.43, y=0.76)
    pts[26].update(x=0.57, y=0.76)
    pts[27].update(x=0.40 + 0.04 * math.sin(phase * 0.5), y=0.95)
    pts[28].update(x=0.60 - 0.04 * math.sin(phase * 0.5), y=0.95)
    return pts


def test_move_markers_are_dense_and_ordered():
    frames = [{"t_ms": i * 33, "landmarks": _pose(i)} for i in range(180)]
    timing = {"beat_ms": list(range(0, frames[-1]["t_ms"] + 1, 500))}
    markers = build_move_markers(frames, timing)

    assert len(markers) >= 8
    assert all(markers[i]["t_ms"] < markers[i + 1]["t_ms"] for i in range(len(markers) - 1))
    gaps = [markers[i + 1]["t_ms"] - markers[i]["t_ms"] for i in range(len(markers) - 1)]
    assert max(gaps) <= 700
    assert all(m["move_index"] == i for i, m in enumerate(markers))


def test_move_markers_reference_real_frames():
    frames = [{"t_ms": i * 40, "landmarks": _pose(i)} for i in range(100)]
    markers = build_move_markers(frames, {"beat_ms": [400, 800, 1200, 1600, 2000, 2400, 2800, 3200]})
    assert markers
    for marker in markers:
        idx = marker["frame_index"]
        assert 0 <= idx < len(frames)
        assert marker["t_ms"] == frames[idx]["t_ms"]


def test_pictograms_are_sparser_than_scoring_markers():
    frames = [{"t_ms": i * 33, "landmarks": _pose(i)} for i in range(240)]
    timing = {"beat_ms": list(range(0, frames[-1]["t_ms"] + 1, 500))}
    dense = build_move_markers(frames, timing)
    cues = build_pictogram_markers(frames, timing)
    assert cues
    assert len(cues) < len(dense)
    gaps = [cues[i + 1]["t_ms"] - cues[i]["t_ms"] for i in range(len(cues) - 1)]
    if gaps:
        assert min(gaps) >= 600
        assert max(gaps) <= 2200
    assert all(cue["kind"].startswith("pictogram") for cue in cues)
