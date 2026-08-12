from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from .build_submission_v8_graph import GRAPH_CONFIG, safe_extract_zip


RUN_PY = r'''from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecup_matching.submission.predict_v6 import predict_to_csv_v6
from ecup_matching.ml.v8_submission_graph import apply_graph_to_prediction

GRAPH_CONFIG = {'rb': 0.0, 'rt': 0.0, 'ep': 0.02, 'ap': 0.01}
MIN_STRICT_OOF_MACRO_AP = 0.60


def submission_root(run_file: Path) -> Path:
    return Path(run_file).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--items_path', type=Path, required=True)
    parser.add_argument('--matches_path', type=Path, required=True)
    args = parser.parse_args()

    root = submission_root(Path(__file__))
    metadata = json.loads((root / 'model_v6_gate_metadata.json').read_text(encoding='utf-8'))
    selected_oof = float(metadata['strict_selected_oof_macro_ap'])
    if selected_oof < MIN_STRICT_OOF_MACRO_AP:
        raise RuntimeError(f'packaged v6 candidate violates strict OOF gate: {selected_oof}')
    if metadata.get('selection_gold_metric_opened') is not False or int(
        metadata.get('selection_gold_rows_scored', -1)
    ) != 0:
        raise RuntimeError('packaged v6 metadata violates sealed-gold selection contract')

    coverage = float(metadata['coverage'])
    base = predict_to_csv_v6(
        coverage=coverage,
        items_path=args.items_path,
        matches_path=args.matches_path,
        structured_model_path=root / 'model_v5_structured.joblib',
        contrastive_model_dir=root / 'model_v5_contrastive',
        teacher_model_dir=root / 'model_v5_teacher',
        category_model_path=root / 'model_v6_category_shrunk.json',
        hgb_model_path=root / 'model_v6_hgb_meta.joblib',
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
    print(f'[v8-fast] v6 gate + graph complete rows={len(final):,} coverage={coverage:.3f}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
'''


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _validate_source_v6(root: Path, expected_metric: float) -> dict[str, object]:
    required = [
        'run.py',
        'ecup_matching/submission/predict_v6.py',
        'model_v5_structured.joblib',
        'model_v6_category_shrunk.json',
        'model_v6_hgb_meta.joblib',
        'model_v6_gate_metadata.json',
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f'source v6 archive missing optimized runtime contract: {missing}')
    if not (root / 'model_v5_contrastive').is_dir() or not (root / 'model_v5_teacher').is_dir():
        raise ValueError('source v6 archive missing neural model directories')
    predictor = (root / 'ecup_matching/submission/predict_v6.py').read_text(encoding='utf-8')
    if 'predict_to_csv_v6' not in predictor:
        raise ValueError('source v6 predict_v6 runtime contract lacks predict_to_csv_v6')
    metadata = json.loads((root / 'model_v6_gate_metadata.json').read_text(encoding='utf-8'))
    if abs(float(metadata['strict_selected_oof_macro_ap']) - float(expected_metric)) > 1e-12:
        raise ValueError('source v6 strict OOF metric mismatch')
    if abs(float(metadata['coverage']) - 0.95) > 1e-12:
        raise ValueError('source v6 must be exact gate95 package')
    if metadata.get('selection_gold_metric_opened') is not False or int(
        metadata.get('selection_gold_rows_scored', -1)
    ) != 0:
        raise ValueError('source v6 violates sealed-gold contract')
    return metadata


def _write_zip(root: Path, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f'output archive already exists: {output}')
    files = sorted(path for path in root.rglob('*') if path.is_file())
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for path in files:
            if path.is_symlink():
                raise ValueError(f'refusing to package symlink: {path}')
            arcname = path.relative_to(root).as_posix()
            pure = PurePosixPath(arcname)
            if pure.is_absolute() or '..' in pure.parts:
                raise ValueError(f'unsafe package path: {arcname}')
            zf.write(path, arcname)


def build_v8_from_v6_zip(
    source_v6_zip: Path,
    output_zip: Path,
    *,
    v8_graph_source: Path,
    v8_submission_graph_source: Path,
    source_v6_metric: float,
    graph_oof_delta: float,
    source_commit: str,
) -> dict[str, object]:
    source_v6_zip = Path(source_v6_zip).resolve(strict=True)
    output_zip = Path(output_zip)
    if len(source_commit) != 40 or any(ch not in '0123456789abcdef' for ch in source_commit.lower()):
        raise ValueError('source_commit must be an exact 40-character hex SHA')
    for path in (Path(v8_graph_source), Path(v8_submission_graph_source)):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'graph runtime source must be a regular file: {path}')

    with tempfile.TemporaryDirectory(prefix='ecup-v8-fast-build-') as tmp:
        root = Path(tmp) / 'submission'
        safe_extract_zip(source_v6_zip, root)
        source_meta = _validate_source_v6(root, source_v6_metric)

        ml = root / 'ecup_matching/ml'
        ml.mkdir(parents=True, exist_ok=True)
        init = ml / '__init__.py'
        if not init.exists():
            init.write_text('', encoding='utf-8')
        shutil.copyfile(v8_graph_source, ml / 'v8_graph.py')
        shutil.copyfile(v8_submission_graph_source, ml / 'v8_submission_graph.py')
        (root / 'run.py').write_text(RUN_PY, encoding='utf-8')

        metadata = {
            'version': 'v8-v6-fast-gate95-plus-graph',
            'source_commit': source_commit,
            'source_v6_zip_sha256': _sha256(source_v6_zip),
            'base': {
                'version': 'v6-fast-gate95',
                'strict_oof_macro_ap': float(source_v6_metric),
                'teacher_coverage': float(source_meta['coverage']),
                'runtime_path': 'predict_to_csv_v6',
            },
            'graph': {
                'config': dict(GRAPH_CONFIG),
                'strict_oof_full_delta': float(graph_oof_delta),
                'target_free': True,
            },
            'sealed_gold_opened': False,
            'gold_rows_scored': 0,
            'true_test_prevalence_claimed': False,
            'notes': 'Uses optimized v6 inference path; graph is target-free and selected without sealed gold.',
        }
        (root / 'v8_metadata.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        _write_zip(root, output_zip)

    archive_bytes = output_zip.stat().st_size
    if archive_bytes >= 5 * 1024**3:
        output_zip.unlink(missing_ok=True)
        raise ValueError(f'v8 archive exceeds 5 GiB: {archive_bytes}')
    return {
        'output': str(output_zip),
        'archive_bytes': int(archive_bytes),
        'archive_sha256': _sha256(output_zip),
        'source_v6_zip_sha256': metadata['source_v6_zip_sha256'],
        'graph_config': dict(GRAPH_CONFIG),
    }


__all__ = ['build_v8_from_v6_zip']
