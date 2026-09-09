from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

import pandas as pd
from PIL import Image

CLASS_NAME_TO_ID = {
    "dark": 0,
    "normal": 1,
    "bright": 2,
    "0": 0,
    "1": 1,
    "2": 2,
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    train_csv: Path
    test_csv: Path
    sample_submission_csv: Path
    train_dir: Path
    test_dir: Path


def discover_layout(root: Path | str) -> DatasetLayout:
    root = Path(root).resolve()
    candidates = [root, root / "data"]
    for candidate in candidates:
        if (
            (candidate / "train.csv").is_file()
            and (candidate / "test.csv").is_file()
            and (candidate / "sample_submission.csv").is_file()
            and (candidate / "train").is_dir()
            and (candidate / "test").is_dir()
        ):
            return DatasetLayout(
                root=candidate,
                train_csv=candidate / "train.csv",
                test_csv=candidate / "test.csv",
                sample_submission_csv=candidate / "sample_submission.csv",
                train_dir=candidate / "train",
                test_dir=candidate / "test",
            )
    raise FileNotFoundError(f"Could not discover dataset layout under {root}")


def _image_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _id_map(paths: Iterable[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        image_id = path.stem
        if image_id in result:
            raise ValueError(f"duplicate image id {image_id!r}: {result[image_id]} and {path}")
        result[image_id] = path
    return result


def load_manifests(layout: DatasetLayout) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(layout.train_csv)
    test_df = pd.read_csv(layout.test_csv)
    sample_df = pd.read_csv(layout.sample_submission_csv)

    if list(train_df.columns) != ["id", "label"]:
        raise ValueError(f"train.csv columns must be ['id', 'label'], got {list(train_df.columns)}")
    valid_test_columns = list(test_df.columns) == ["id"] or (
        list(test_df.columns) == ["id", "label"] and test_df["label"].isna().all()
    )
    if not valid_test_columns:
        raise ValueError(
            "test.csv must contain id or id plus an entirely empty label column, "
            f"got {list(test_df.columns)}"
        )
    if "label" in test_df.columns:
        test_df = test_df[["id"]].copy()
    if list(sample_df.columns) != ["id", "label"]:
        raise ValueError(
            "sample_submission.csv columns must be ['id', 'label'], "
            f"got {list(sample_df.columns)}"
        )

    train_df = train_df.copy()
    test_df = test_df.copy()
    sample_df = sample_df.copy()
    train_df["id"] = train_df["id"].astype(str)
    test_df["id"] = test_df["id"].astype(str)
    sample_df["id"] = sample_df["id"].astype(str)
    train_df["label"] = train_df["label"].astype(int)

    if not set(train_df["label"].unique()).issubset({0, 1, 2}):
        raise ValueError("train labels must be a subset of {0, 1, 2}")
    if train_df["id"].duplicated().any() or test_df["id"].duplicated().any():
        raise ValueError("manifest ids must be unique")
    if sample_df["id"].tolist() != test_df["id"].tolist():
        raise ValueError("sample submission id order must match test.csv")
    return train_df, test_df, sample_df


def resolve_image_paths(layout: DatasetLayout) -> tuple[list[Path], list[Path]]:
    train_df, test_df, _ = load_manifests(layout)
    train_map = _id_map(_image_files(layout.train_dir))
    test_map = _id_map(_image_files(layout.test_dir))
    train_paths = [train_map[str(image_id)] for image_id in train_df["id"]]
    test_paths = [test_map[str(image_id)] for image_id in test_df["id"]]
    return train_paths, test_paths


def validate_dataset(
    layout: DatasetLayout,
    expected_train: int | None = 1500,
    expected_test: int | None = 300,
    expected_per_class: int | None = 500,
) -> dict:
    train_df, test_df, sample_df = load_manifests(layout)
    train_files = _image_files(layout.train_dir)
    test_files = _image_files(layout.test_dir)
    train_map = _id_map(train_files)
    test_map = _id_map(test_files)

    train_ids = set(train_df["id"].astype(str))
    test_ids = set(test_df["id"].astype(str))
    train_image_ids = set(train_map)
    test_image_ids = set(test_map)

    if train_ids != train_image_ids:
        missing = sorted(train_ids - train_image_ids)[:10]
        extra = sorted(train_image_ids - train_ids)[:10]
        raise ValueError(f"train manifest/image mismatch: missing={missing}, extra={extra}")
    if test_ids != test_image_ids:
        missing = sorted(test_ids - test_image_ids)[:10]
        extra = sorted(test_image_ids - test_ids)[:10]
        raise ValueError(f"test manifest/image mismatch: missing={missing}, extra={extra}")

    if expected_train is not None and len(train_df) != expected_train:
        raise ValueError(f"expected {expected_train} train rows, got {len(train_df)}")
    if expected_test is not None and len(test_df) != expected_test:
        raise ValueError(f"expected {expected_test} test rows, got {len(test_df)}")

    dist = {int(k): int(v) for k, v in train_df["label"].value_counts().sort_index().items()}
    if expected_per_class is not None and dist != {0: expected_per_class, 1: expected_per_class, 2: expected_per_class}:
        raise ValueError(
            f"expected balanced train classes of {expected_per_class}, got {dist}"
        )

    shape_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    for path in train_files[: min(50, len(train_files))] + test_files[: min(20, len(test_files))]:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            key = f"{image.width}x{image.height}"
            shape_counts[key] = shape_counts.get(key, 0) + 1
            mode_counts[image.mode] = mode_counts.get(image.mode, 0) + 1

    return {
        "root": str(layout.root),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_images": int(len(train_files)),
        "test_images": int(len(test_files)),
        "class_distribution": dist,
        "sample_submission_rows": int(len(sample_df)),
        "sampled_shape_counts": shape_counts,
        "sampled_mode_counts": mode_counts,
    }


def normalize_extracted_tree(root: Path | str) -> DatasetLayout:
    root = Path(root).resolve()
    try:
        return discover_layout(root)
    except FileNotFoundError:
        pass

    candidates: list[Path] = []
    for train_csv in root.rglob("train.csv"):
        candidate = train_csv.parent
        if (
            (candidate / "test.csv").is_file()
            and (candidate / "sample_submission.csv").is_file()
            and (candidate / "train").is_dir()
            and (candidate / "test").is_dir()
        ):
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f"No valid extracted dataset found under {root}")
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return discover_layout(candidates[0])


def write_validation_report(report: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
