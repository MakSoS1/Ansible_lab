from __future__ import annotations

from ecup_matching.ml.run_v19_weight_interp_probe import interpolation_specs


def test_interpolation_specs_cover_conservative_v18_and_v19_directions() -> None:
    specs = interpolation_specs()
    names = [spec["name"] for spec in specs]
    assert len(names) == len(set(names))
    assert any(name.startswith("v18-ema-rescue") for name in names)
    assert any(name.startswith("v19-sharpen") for name in names)
    for spec in specs:
        assert 0.0 < float(spec["alpha"]) < 1.0
        assert spec["left"] in {"base", "v18", "r005"}
        assert spec["right"] in {"v18", "r005", "r010"}
