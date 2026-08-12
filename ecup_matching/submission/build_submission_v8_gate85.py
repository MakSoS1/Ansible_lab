from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from ecup_matching.ci.runtime_closure import copy_runtime_closure, runtime_import_closure
from .build_submission_v8_graph import safe_extract_zip
from .build_submission_v8_v6graph import _validate_source_v6, _write_zip


SOURCE_V6_METRIC = 0.6006003614522999
EXPECTED_COVERAGE = 0.85
EXPECTED_BASE_OOF = 0.5999300791828578
EXPECTED_GRAPH_OOF = 0.6021573018691804
EXPECTED_SPLIT_SHA = "aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b"
GRAPH_CONFIG = {"rb": 0.0, "rt": 0.0, "ep": 0.02, "ap": 0.01}


RUN_PY = r'''from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecup_matching.submission.predict_v6 import predict_to_csv_v6
from ecup_matching.ml.v8_submission_graph import apply_graph_to_prediction

EXPECTED_COVERAGE = 0.85
EXPECTED_BASE_OOF = 0.5999300791828578
EXPECTED_GRAPH_OOF = 0.6021573018691804
EXPECTED_SPLIT_SHA = "aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b"
GRAPH_CONFIG = {"rb": 0.0, "rt": 0.0, "ep": 0.02, "ap": 0.01}


def _close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= 1e-12


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--items_path', type=Path, required=True)
    parser.add_argument('--matches_path', type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    metadata = json.loads((root / 'model_v6_gate_metadata.json').read_text(encoding='utf-8'))
    if not _close(metadata['coverage'], EXPECTED_COVERAGE):
        raise RuntimeError(f"wrong packaged teacher coverage: {metadata.get('coverage')}")
    if not _close(metadata['base_strict_oof_macro_ap'], EXPECTED_BASE_OOF):
        raise RuntimeError('wrong packaged gate85 base OOF evidence')
    if not _close(metadata['fold_local_graph_strict_oof_macro_ap'], EXPECTED_GRAPH_OOF):
        raise RuntimeError('wrong packaged fold-local graph OOF evidence')
    if float(metadata['fold_local_graph_strict_oof_macro_ap']) < 0.60:
        raise RuntimeError('packaged candidate violates >=0.60 graph quality gate')
    if metadata.get('split_sha256') != EXPECTED_SPLIT_SHA:
        raise RuntimeError('packaged split SHA mismatch')
    if metadata.get('graph_config') != GRAPH_CONFIG:
        raise RuntimeError('packaged graph config does not match validated graph config')
    if metadata.get('selection_gold_metric_opened') is not False or int(metadata.get('selection_gold_rows_scored', -1)) != 0:
        raise RuntimeError('packaged candidate violates sealed-gold selection contract')

    base = predict_to_csv_v6(
        coverage=EXPECTED_COVERAGE,
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
        raise RuntimeError(f'base prediction row mismatch: {len(base)} != {len(matches)}')
    if not np.array_equal(base['id1'].to_numpy(), matches['id1'].to_numpy()) or not np.array_equal(base['id2'].to_numpy(), matches['id2'].to_numpy()):
        raise RuntimeError('base prediction pair order mismatch')

    test = matches.copy()
    test.insert(0, 'id', np.arange(len(test), dtype=np.int64))
    prediction = base[['predict']].copy()
    prediction.insert(0, 'id', np.arange(len(prediction), dtype=np.int64))
    items = pd.read_parquet(args.items_path, columns=['id', 'category'])
    rescored = apply_graph_to_prediction(test, items, prediction, GRAPH_CONFIG)

    final = base[['id1', 'id2']].copy()
    final['predict'] = rescored['predict'].to_numpy(dtype=np.float64)
    score = final['predict'].to_numpy(dtype=np.float64)
    if len(final) != len(matches) or not np.isfinite(score).all() or len(np.unique(score)) <= 1:
        raise RuntimeError('final gate85+graph prediction is invalid')
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output_path, index=False)
    print(f'[v8-gate85] complete rows={len(final):,} coverage={EXPECTED_COVERAGE:.2f}', flush=True)
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


def _load_gate85_metadata(path: Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError('gate85 metadata must be a regular file')
    meta = json.loads(path.read_text(encoding='utf-8'))
    checks = (
        (abs(float(meta.get('coverage', -1.0)) - EXPECTED_COVERAGE) <= 1e-12, 'coverage'),
        (abs(float(meta.get('base_strict_oof_macro_ap', -1.0)) - EXPECTED_BASE_OOF) <= 1e-12, 'base OOF'),
        (abs(float(meta.get('fold_local_graph_strict_oof_macro_ap', -1.0)) - EXPECTED_GRAPH_OOF) <= 1e-12, 'graph OOF'),
        (meta.get('split_sha256') == EXPECTED_SPLIT_SHA, 'split SHA'),
        (meta.get('graph_config') == GRAPH_CONFIG, 'graph config'),
        (meta.get('selection_gold_metric_opened') is False, 'sealed gold flag'),
        (int(meta.get('selection_gold_rows_scored', -1)) == 0, 'sealed gold rows'),
    )
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise ValueError(f'invalid gate85 production metadata: {failed}')
    if float(meta['fold_local_graph_strict_oof_macro_ap']) < 0.60:
        raise ValueError('gate85 graph OOF does not clear 0.60 quality gate')
    return meta


def build_v8_gate85(
    *,
    source_v6_zip: Path,
    gate85_category_path: Path,
    gate85_hgb_path: Path,
    gate85_metadata_path: Path,
    output_zip: Path,
    v8_graph_source: Path,
    v8_submission_graph_source: Path,
    source_commit: str,
) -> dict[str, object]:
    source_v6_zip = Path(source_v6_zip).resolve(strict=True)
    output_zip = Path(output_zip)
    if len(source_commit) != 40 or any(ch not in '0123456789abcdef' for ch in source_commit.lower()):
        raise ValueError('source_commit must be an exact 40-character hex SHA')
    for path in (gate85_category_path, gate85_hgb_path, v8_graph_source, v8_submission_graph_source):
        p = Path(path)
        if not p.is_file() or p.is_symlink():
            raise ValueError(f'package input must be a regular file: {p}')
    gate_meta = _load_gate85_metadata(Path(gate85_metadata_path))

    with tempfile.TemporaryDirectory(prefix='ecup-v8-gate85-build-') as tmp:
        root = Path(tmp) / 'submission'
        safe_extract_zip(source_v6_zip, root)
        _validate_source_v6(root, SOURCE_V6_METRIC)

        # Keep only the verified common heavy weights from the exact gate95 ZIP.
        # All reachable first-party Python is overlaid from this source commit.
        copied = copy_runtime_closure(root)
        ml = root / 'ecup_matching' / 'ml'
        ml.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(v8_graph_source, ml / 'v8_graph.py')
        shutil.copyfile(v8_submission_graph_source, ml / 'v8_submission_graph.py')

        # Coverage-specific production meta MUST be gate85, never stale gate95.
        shutil.copyfile(gate85_category_path, root / 'model_v6_category_shrunk.json')
        shutil.copyfile(gate85_hgb_path, root / 'model_v6_hgb_meta.joblib')
        shutil.copyfile(gate85_metadata_path, root / 'model_v6_gate_metadata.json')
        (root / 'run.py').write_text(RUN_PY, encoding='utf-8')

        gaps = [rel for rel in runtime_import_closure() if not (root / rel).is_file()]
        if gaps:
            raise RuntimeError(f'final runtime closure incomplete: {gaps}')
        runtime_hashes = {
            rel: _sha256(root / rel)
            for rel in runtime_import_closure()
            if (root / rel).is_file()
        }
        runtime_predict = root / 'ecup_matching/submission/predict_v6.py'
        if runtime_predict.is_file():
            text = runtime_predict.read_text(encoding='utf-8')
            for marker in ('_structured_scores_streaming', 'run_structured_chunks', 'torch_autocast', 'build_dual_text_cache'):
                if marker not in text:
                    raise RuntimeError(f'final predict_v6 missing optimized marker: {marker}')

        final_meta = {
            'version': 'v8-gate85-fp16-dualcache-graph',
            'source_commit': source_commit,
            'source_v6_zip_sha256': _sha256(source_v6_zip),
            'base': {
                'teacher_coverage': EXPECTED_COVERAGE,
                'base_strict_oof_macro_ap': EXPECTED_BASE_OOF,
                'fold_local_graph_strict_oof_macro_ap': EXPECTED_GRAPH_OOF,
                'split_sha256': EXPECTED_SPLIT_SHA,
            },
            'graph': {
                'config': dict(GRAPH_CONFIG),
                'fold_local_validation': True,
                'all_folds_positive': True,
                'target_free': True,
            },
            'runtime': {
                'path': 'predict_to_csv_v6',
                'structured_worker_cap': 8,
                'cuda_autocast': 'float16',
                'contrastive_batch_rtx2060': 256,
                'teacher_batch_rtx2060': 96,
                'dual_text_cache_exact': True,
                'dual_text_cache_workers': 'same safe CPU worker count as structured',
                'closure_files': list(copied),
                'closure_sha256': runtime_hashes,
            },
            'gate85_metadata_sha256': _sha256(Path(gate85_metadata_path)),
            'gate85_category_sha256': _sha256(Path(gate85_category_path)),
            'gate85_hgb_sha256': _sha256(Path(gate85_hgb_path)),
            'sealed_gold_opened': False,
            'gold_rows_scored': 0,
            'quality_gate_macro_ap': 0.60,
            'quality_gate_basis': 'strict leakage-free fold-local graph OOF',
        }
        # Preserve exact evidence from the refit inside the archive as the source of truth.
        if abs(float(gate_meta['fold_local_graph_strict_oof_macro_ap']) - final_meta['base']['fold_local_graph_strict_oof_macro_ap']) > 1e-12:
            raise RuntimeError('gate85 evidence changed during packaging')
        (root / 'v8_metadata.json').write_text(
            json.dumps(final_meta, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        _write_zip(root, output_zip)

    archive_bytes = output_zip.stat().st_size
    if archive_bytes >= 5 * 1024**3:
        output_zip.unlink(missing_ok=True)
        raise ValueError(f'v8 gate85 archive exceeds 5 GiB: {archive_bytes}')
    return {
        'output': str(output_zip),
        'archive_bytes': int(archive_bytes),
        'archive_sha256': _sha256(output_zip),
        'coverage': EXPECTED_COVERAGE,
        'base_strict_oof_macro_ap': EXPECTED_BASE_OOF,
        'fold_local_graph_strict_oof_macro_ap': EXPECTED_GRAPH_OOF,
        'graph_config': dict(GRAPH_CONFIG),
    }


__all__ = ['build_v8_gate85']
