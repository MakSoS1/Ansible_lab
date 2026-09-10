from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    economic_start: date = date(2007, 1, 1)
    max_wlpr_m3_day: float = 500.0
    hf_dataset_id: str = "MakSoS1/aios-track2-runs"
    model_z_archive: Path = Path("aios-track2/materials/41_Model_Z_final_OPM.zip")
    work_dir: Path = Path("runs")
    control_step_months: int = Field(default=3, ge=1, le=12)


def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)
