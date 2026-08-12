from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


GRAPH_CONFIG = {"rb": 0.0, "rt": 0.0, "ep": 0.02, "ap": 0.01}
MAX_UNCOMPRESSED_BYTES = 5 * 1024**3
MAX_MEMBERS = 20_000


RUN_PY = r'''from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecup_matching.submission.predict_v5 import predict_to_csv_v5
from ecup_matching.ml.v8_submission_graph import apply_graph_to_prediction

GRAPH_CONFIG = {'rb': 0.0, 'rt': 0.0, 'ep': 0.02, 'ap': 0.01}


def submission_root(run_file: Path) -> Path:
    return Path(run_file).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--items_path', type=Path, required=True)
    parser.add_argument('--matches_path', type=Path, required=True)
    args = parser.parse_args()

    root = submission_root(Path(__file__))
    base = predict_to_csv_v5(
        items_path=args.items_path,
        matches_path=args.matches_path,
        structured_model_path=root / 'model_v5_structured.joblib',
        contrastive_model_dir=root / 'model_v5_contrastive',
        teacher_model_dir=root / 'model_v5_teacher',
        category_model_path=root / 'model_v5_category_shrunk.json',
        hgb_model_path=root / 'model_v5_hgb_meta.joblib',
        runtime_root=root,
        output_path=args.output_path,
    ).reset_index(drop=True)

    matches = pd.read_parquet(args.matches_path, columns=['id1', 'id2']).reset_index(drop=True)
    if len(base) != len(matches):
        raise RuntimeError(f'v8 base prediction row mismatch: {len(base)} != {len(matches)}')
    if not np.array_equal(base['id1'].to_numpy(), matches['id1'].to_numpy()) or not np.array_equal(
        base['id2'].to_numpy(), matches['id2'].to_numpy()
    ):
        raise RuntimeError('v8 base prediction pair order mismatch')

    test = matches.copy()
    test.insert(0, 'id', np.arange(len(test), dtype=np.int64))
    prediction = base[['predict']].copy()
    prediction.insert(0, 'id', np.arange(len(prediction), dtype=np.int64))
    items = pd.read_parquet(args.items_path, columns=['id', 'category'])
    rescored = apply_graph_to_prediction(test, items, prediction, GRAPH_CONFIG)

    final = base[['id1', 'id2']].copy()
    final['predict'] = rescored['predict'].to_numpy(dtype=np.float64)
    if len(final) != len(matches) or not np.isfinite(final['predict'].to_numpy(float)).all():
        raise RuntimeError('v8 graph prediction is incomplete or non-finite')
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output_path, index=False)
    print(f'[v8] graph postprocess complete rows={len(final):,}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
'''


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_member(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename
    if not name or "\\" in name:
        raise ValueError(f"unsafe archive member: {name!r}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe archive member: {name!r}")
    mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_IFMT(mode) == stat.S_IFLNK:
        raise ValueError(f"symlink archive member is forbidden: {name!r}")
    return pure


def safe_extract_zip(archive: Path, destination: Path) -> Path:
    archive = Path(archive).resolve(strict=True)
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    destination.mkdir(parents=True, mode=0o700)
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            if not infos or len(infos) > MAX_MEMBERS:
                raise ValueError(f"invalid archive member count: {len(infos)}")
            total = sum(int(info.file_size) for info in infos)
            if total <= 0 or total > MAX_UNCOMPRESSED_BYTES:
                raise ValueError(f"invalid uncompressed archive size: {total}")
            for info in infos:
                pure = _safe_member(info)
                target = destination.joinpath(*pure.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
        return destination
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _validate_source_v5(root: Path) -> None:
    required = [
        "run.py",
        "ecup_matching/submission/predict_v5.py",
        "ecup_matching/ml/__init__.py",
        "model_v5_structured.joblib",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"source v5 archive missing predict_v5 runtime contract: {missing}")
    predictor = (root / "ecup_matching/submission/predict_v5.py").read_text(encoding="utf-8")
    if "predict_to_csv_v5" not in predictor:
        raise ValueError("source v5 predict_v5 runtime contract lacks predict_to_csv_v5")


def _write_zip(root: Path, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"output archive already exists: {output}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for path in files:
            if path.is_symlink():
                raise ValueError(f"refusing to package symlink: {path}")
            arcname = path.relative_to(root).as_posix()
            if PurePosixPath(arcname).is_absolute() or ".." in PurePosixPath(arcname).parts:
                raise ValueError(f"unsafe package path: {arcname}")
            zf.write(path, arcname)


def build_v8_from_v5_zip(
    source_v5_zip: Path,
    output_zip: Path,
    *,
    v8_graph_source: Path,
    v8_submission_graph_source: Path,
    source_v5_metric: float,
    graph_oof_delta: float,
    source_commit: str,
) -> dict[str, object]:
    source_v5_zip = Path(source_v5_zip).resolve(strict=True)
    output_zip = Path(output_zip)
    if len(source_commit) != 40 or any(ch not in "0123456789abcdef" for ch in source_commit.lower()):
        raise ValueError("source_commit must be an exact 40-character hex SHA")
    for path in (Path(v8_graph_source), Path(v8_submission_graph_source)):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"graph runtime source must be a regular file: {path}")

    with tempfile.TemporaryDirectory(prefix="ecup-v8-build-") as tmp:
        root = Path(tmp) / "submission"
        safe_extract_zip(source_v5_zip, root)
        _validate_source_v5(root)

        ml = root / "ecup_matching/ml"
        ml.mkdir(parents=True, exist_ok=True)
        init = ml / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
        shutil.copyfile(v8_graph_source, ml / "v8_graph.py")
        shutil.copyfile(v8_submission_graph_source, ml / "v8_submission_graph.py")
        (root / "run.py").write_text(RUN_PY, encoding="utf-8")

        metadata = {
            "version": "v8-v5-best-plus-graph",
            "source_commit": source_commit,
            "source_v5_zip_sha256": _sha256(source_v5_zip),
            "base": {
                "version": "v5-category-hgb-fusion",
                "strict_oof_macro_ap": float(source_v5_metric),
                "score": "equal_rank_fusion_score",
            },
            "graph": {
                "config": dict(GRAPH_CONFIG),
                "strict_oof_full_delta": float(graph_oof_delta),
                "selection_folds": [0, 1],
                "confirmatory_folds": [2, 3, 4],
                "confirmatory_delta": 0.0006224282109241752,
                "target_free": True,
            },
            "sealed_gold_opened": False,
            "gold_rows_scored": 0,
            "true_test_prevalence_claimed": False,
            "notes": "Graph config was selected without sealed gold; LLM prevalence diagnostics are diagnostic-only.",
        }
        (root / "v8_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_zip(root, output_zip)

    archive_bytes = output_zip.stat().st_size
    if archive_bytes >= 5 * 1024**3:
        output_zip.unlink(missing_ok=True)
        raise ValueError(f"v8 archive exceeds 5 GiB: {archive_bytes}")
    return {
        "output": str(output_zip),
        "archive_bytes": int(archive_bytes),
        "archive_sha256": _sha256(output_zip),
        "source_v5_zip_sha256": metadata["source_v5_zip_sha256"],
        "graph_config": dict(GRAPH_CONFIG),
    }


__all__ = ["GRAPH_CONFIG", "build_v8_from_v5_zip", "safe_extract_zip"]
