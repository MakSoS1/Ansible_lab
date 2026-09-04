from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deck: Path
    work_dir: Path
    materials: Path = Path("aios-track2/materials")


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int
    economic_start: date
    max_wlpr_m3_day: float
    hf_dataset_id: str
    device: str = "auto"
    paths: PathsConfig = Field(...)

    def resolve(self, root: Path | None = None) -> AppConfig:
        base = root or Path.cwd()
        paths = PathsConfig(
            deck=(base / self.paths.deck).resolve() if not self.paths.deck.is_absolute() else self.paths.deck,
            work_dir=(base / self.paths.work_dir).resolve()
            if not self.paths.work_dir.is_absolute()
            else self.paths.work_dir,
            materials=(base / self.paths.materials).resolve()
            if not self.paths.materials.is_absolute()
            else self.paths.materials,
        )
        return self.model_copy(update={"paths": paths})


def load_config(path: Path) -> AppConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(payload)
