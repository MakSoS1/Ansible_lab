"""Package the v7 cross-encoder submission.

The archive is deliberately small compared to v5/v6: one model directory, the
entrypoint, and the eight-module runtime import closure. There is no structured
bundle, no TF-IDF vectorizer and no meta stack, because v7 scores every pair
with a single cross-encoder forward pass.

The metadata written here records the fold-0 diagnostic under its own key and
leaves ``strict_oof_macro_average_precision`` null until the five-fold outer OOF
driver produces one. A model refit on every development row cannot be scored by
any fold it already saw, so no diagnostic may be promoted into that field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from ecup_matching.ci.runtime_closure import V7_ENTRYPOINTS, copy_runtime_closure


MAX_ZIP_BYTES = 5 * 1024**3
METADATA = {
    "image": "odsai/ecup26-matching-baseline:1.0",
    "entry_point": "python -u run.py",
}
_FORBIDDEN_SUFFIXES = {".parquet", ".csv", ".pcap", ".env"}
_CREDENTIAL_BASENAMES = {
    "token", "token.txt", "hf_token", "hf_token.txt",
    "secret", "secrets", "credentials.json", ".netrc",
}


def build_model_metadata(
    *,
    diagnostic_fold0: float | None,
    strict_oof: float | None,
    production_metrics: dict | None,
    max_length: int = 256,
    max_chars: int = 900,
    inference_batch_size: int = 64,
) -> dict:
    """Describe the packaged model without overstating what has been measured."""
    if strict_oof is not None and diagnostic_fold0 is not None:
        if float(strict_oof) == float(diagnostic_fold0):
            raise ValueError(
                "strict_oof_macro_average_precision must not repeat the fold-0 "
                "diagnostic; a single held fold is not an out-of-fold score"
            )
    payload: dict = {
        "version": "v7-crossencoder",
        "base_model": "ai-forever/ruBert-base",
        "max_length": int(max_length),
        "max_chars": int(max_chars),
        "inference_batch_size": int(inference_batch_size),
        "diagnostic_fold0_macro_average_precision": (
            None if diagnostic_fold0 is None else float(diagnostic_fold0)
        ),
        "diagnostic_fold0_is_not_out_of_fold": True,
        "strict_oof_macro_average_precision": (
            None if strict_oof is None else float(strict_oof)
        ),
        "is_production_refit": True,
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
    }
    if production_metrics:
        for key in (
            "split_sha256",
            "training_rows",
            "development_rows",
            "sealed_gold_rows",
            "cross_split_item_overlap",
            "human_epochs",
            "weak_epochs",
            "base_model_revision",
        ):
            if key in production_metrics:
                payload[key] = production_metrics[key]
    return payload


def _assert_safe_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"submission must not contain symlinks: {path}")
        if not path.is_file():
            continue
        if path.suffix.lower() in _FORBIDDEN_SUFFIXES:
            raise ValueError(f"submission must not contain competition data: {path}")
        if path.name.lower() in _CREDENTIAL_BASENAMES:
            raise ValueError(f"submission must not contain credentials: {path}")


def build_submission_v7(
    *,
    model_dir: Path,
    output_zip: Path,
    diagnostic_fold0: float | None = None,
    strict_oof: float | None = None,
    production_metrics_path: Path | None = None,
) -> dict:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory is missing: {model_dir}")
    if not (model_dir / "config.json").is_file():
        raise ValueError("model directory is missing config.json")
    if not any(model_dir.glob("*.safetensors")):
        raise ValueError("model directory has no safetensors weights")

    production_metrics = None
    if production_metrics_path is not None:
        production_metrics = json.loads(
            Path(production_metrics_path).read_text(encoding="utf-8")
        )
        if production_metrics.get("validation_metric_reported") is not False:
            raise ValueError("production metrics must not report a validation score")

    metadata = build_model_metadata(
        diagnostic_fold0=diagnostic_fold0,
        strict_oof=strict_oof,
        production_metrics=production_metrics,
    )

    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw) / "submission"
        root.mkdir()
        (root / "metadata.json").write_text(
            json.dumps(METADATA, ensure_ascii=False, indent=4) + "\n", encoding="utf-8"
        )
        packaged = copy_runtime_closure(root, V7_ENTRYPOINTS)
        shutil.copy2(root / "ecup_matching" / "submission" / "run_v7.py", root / "run.py")
        (root / "model_v7_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        target_model = root / "model_v7_teacher"
        target_model.mkdir()
        for path in sorted(model_dir.iterdir()):
            if path.is_file():
                shutil.copy2(path, target_model / path.name)
        _assert_safe_tree(root)

        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())

    size = output_zip.stat().st_size
    if size > MAX_ZIP_BYTES:
        raise ValueError(f"submission archive is {size} bytes, over the 5GB limit")
    return {
        "zip": str(output_zip),
        "bytes": size,
        "packaged_modules": packaged,
        "model_metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--diagnostic-fold0", type=float, default=None)
    parser.add_argument("--strict-oof", type=float, default=None)
    parser.add_argument("--production-metrics", type=Path, default=None)
    args = parser.parse_args()
    report = build_submission_v7(
        model_dir=args.model_dir,
        output_zip=args.output_zip,
        diagnostic_fold0=args.diagnostic_fold0,
        strict_oof=args.strict_oof,
        production_metrics_path=args.production_metrics,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
