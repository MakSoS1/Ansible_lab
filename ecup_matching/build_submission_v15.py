from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile

MAX_ZIP_BYTES = 5 * 1024**3
METADATA = {"image": "odsai/ecup26-matching-baseline:1.0", "entry_point": "python -u run.py"}
RUNTIME_FILES = (
    "ecup_matching/__init__.py",
    "ecup_matching/v15_fields.py",
    "ecup_matching/v15_pair_features.py",
    "ecup_matching/v15_model.py",
    "ecup_matching/submission/__init__.py",
    "ecup_matching/submission/predict_v15.py",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_symlink():
            raise RuntimeError(f"symlink forbidden in submission: {p}")
        if p.is_file() and p.suffix.lower() in {".parquet", ".csv", ".env", ".pcap"}:
            raise RuntimeError(f"data/secret-like file forbidden in submission: {p}")


def build(*, source_root: Path, checkpoint: Path, base_model_dir: Path, output_zip: Path,
          fold0_macro_ap: float, variant: str, source_sha: str, training_rows: int) -> dict:
    import torch
    ckpt = torch.load(checkpoint, map_location="cpu")
    if str(ckpt.get("variant")) != str(variant):
        raise RuntimeError("checkpoint variant mismatch")
    if ckpt.get("base_model_weights_sha256") != "f3ea88b230492811046145513710e76b4cc8c2ad49e8708da0e7247e548903be":
        raise RuntimeError("unexpected Granite base weights provenance")
    if int(ckpt.get("max_length", -1)) != 128:
        raise RuntimeError("v15 checkpoint max_length must be 128")

    metadata = {
        "version": "v15-field-aware-crossencoder",
        "variant": str(variant),
        "diagnostic_fold0_macro_average_precision": float(fold0_macro_ap),
        "diagnostic_fold0_is_not_strict_oof": True,
        "training_rows": int(training_rows),
        "development_rows": 285210,
        "sealed_gold_rows": 80444,
        "gold_metric_opened": False,
        "gold_rows_scored": 0,
        "split_sha256": "aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b",
        "canonical_rowmap_sha256": "00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f",
        "public_source_sha": source_sha,
        "base_model_revision": ckpt.get("base_model_revision"),
        "base_model_weights_sha256": ckpt.get("base_model_weights_sha256"),
        "max_length": 128,
        "inference_batch_size": 64,
        "single_transformer_checkpoint": True,
        "include_attributes": bool(ckpt.get("include_attributes")),
        "use_typed_features": bool(ckpt.get("use_typed_features")),
        "use_category_head": bool(ckpt.get("use_category_head")),
        "macro_balanced_training": bool(ckpt.get("macro_balanced")),
    }

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "submission"; root.mkdir()
        (root / "metadata.json").write_text(json.dumps(METADATA, indent=2) + "\n", encoding="utf-8")
        (root / "model_v15_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.copy2(source_root / "ecup_matching/submission/run_v15.py", root / "run.py")
        for rel in RUNTIME_FILES:
            src = source_root / rel
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        shutil.copy2(checkpoint, root / "v15_model.pt")
        cfg_dir = root / "v15_base_config"; cfg_dir.mkdir()
        copied = []
        for src in sorted(base_model_dir.iterdir()):
            if not src.is_file():
                continue
            lower = src.name.lower()
            if lower.endswith((".safetensors", ".bin", ".pt", ".pth")):
                continue
            shutil.copy2(src, cfg_dir / src.name); copied.append(src.name)
        if "config.json" not in copied:
            raise RuntimeError("base config.json missing")
        if not any(x.startswith("tokenizer") for x in copied):
            raise RuntimeError("tokenizer files missing")
        _safe(root)
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(root).as_posix())
    if output_zip.stat().st_size >= MAX_ZIP_BYTES:
        raise RuntimeError("v15 submission exceeds 5 GiB")
    with zipfile.ZipFile(output_zip) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"zip integrity failure at {bad}")
        if "run.py" not in z.namelist() or "v15_model.pt" not in z.namelist():
            raise RuntimeError("required v15 runtime files missing")
    return {"archive": output_zip.name, "archive_bytes": output_zip.stat().st_size,
            "archive_sha256": sha256(output_zip), "metadata": metadata}


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--source-root",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--base-model-dir",type=Path,required=True); p.add_argument("--output-zip",type=Path,required=True); p.add_argument("--fold0",type=float,required=True)
    p.add_argument("--variant",required=True); p.add_argument("--source-sha",required=True); p.add_argument("--training-rows",type=int,required=True)
    a=p.parse_args(); report=build(source_root=a.source_root,checkpoint=a.checkpoint,base_model_dir=a.base_model_dir,output_zip=a.output_zip,fold0_macro_ap=a.fold0,variant=a.variant,source_sha=a.source_sha,training_rows=a.training_rows)
    print("V15_PACKAGE="+json.dumps(report,sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
