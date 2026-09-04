from datetime import date
from pathlib import Path

from aios_track2.config import load_config


def test_base_config_has_track2_contract() -> None:
    cfg = load_config(Path("configs/base.yaml"))
    assert cfg.seed == 42
    assert cfg.economic_start == date(2007, 1, 1)
    assert cfg.max_wlpr_m3_day == 500.0
    assert cfg.hf_dataset_id == "<HF_TOKEN_OWNER>/aios-track2-runs"
