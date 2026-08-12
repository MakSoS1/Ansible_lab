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
EXPECTED_COVERAGE = 0.40
EXPECTED_BASE_OOF = 0.595505427416499
EXPECTED_GRAPH_OOF = 0.597005931143384
EXPECTED_TARGET_STRESS = 0.4515676235464289
EXPECTED_STRESS_RATIO = 0.566880890615799
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

EXPECTED_COVERAGE = 0.40
EXPECTED_BASE_OOF = 0.595505427416499
EXPECTED_GRAPH_OOF = 0.597005931143384
EXPECTED_TARGET_STRESS = 0.4515676235464289
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
    if metadata.get('version') != 'v9-gate40-production-refit':
        raise RuntimeError(f"wrong packaged v9 metadata version: {metadata.get('version')}")
    if not _close(metadata['coverage'], EXPECTED_COVERAGE):
        raise RuntimeError(f"wrong packaged teacher coverage: {metadata.get('coverage')}")
    if not _close(metadata['base_strict_oof_macro_ap'], EXPECTED_BASE_OOF):
        raise RuntimeError('wrong packaged gate40 base OOF evidence')
    if not _close(metadata['fold_local_graph_strict_oof_macro_ap'], EXPECTED_GRAPH_OOF):
        raise RuntimeError('wrong packaged fold-local graph OOF evidence')
    if not _close(metadata['target_stress_mean'], EXPECTED_TARGET_STRESS):
        raise RuntimeError('wrong packaged target-stress evidence')
    if metadata.get('split_sha256') != EXPECTED_SPLIT_SHA:
        raise RuntimeError('packaged split SHA mismatch')
    if metadata.get('graph_config') != GRAPH_CONFIG:
        raise RuntimeError('packaged graph config does not match validated graph config')
    if metadata.get('leaderboard_anchor_used_for_fitting') is not False:
        raise RuntimeError('leaderboard anchor leaked into fitting contract')
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
        raise RuntimeError('final v9 gate40+graph prediction is invalid')
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output_path, index=False)
    print(f'[v9-gate40] complete rows={len(final):,} coverage={EXPECTED_COVERAGE:.2f}', flush=True)
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


def _load_gate40_metadata(path: Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError('gate40 metadata must be a regular file')
    meta = json.loads(path.read_text(encoding='utf-8'))
    checks = (
        (meta.get('version') == 'v9-gate40-production-refit', 'version'),
        (abs(float(meta.get('coverage', -1.0)) - EXPECTED_COVERAGE) <= 1e-12, 'coverage'),
        (abs(float(meta.get('base_strict_oof_macro_ap', -1.0)) - EXPECTED_BASE_OOF) <= 1e-12, 'base OOF'),
        (abs(float(meta.get('fold_local_graph_strict_oof_macro_ap', -1.0)) - EXPECTED_GRAPH_OOF) <= 1e-12, 'graph OOF'),
        (abs(float(meta.get('target_stress_mean', -1.0)) - EXPECTED_TARGET_STRESS) <= 1e-12, 'target stress'),
        (meta.get('split_sha256') == EXPECTED_SPLIT_SHA, 'split SHA'),
        (meta.get('graph_config') == GRAPH_CONFIG, 'graph config'),
        (meta.get('leaderboard_anchor_used_for_fitting') is False, 'leaderboard anchor fitting flag'),
        (meta.get('selection_gold_metric_opened') is False, 'sealed gold flag'),
        (int(meta.get('selection_gold_rows_scored', -1)) == 0, 'sealed gold rows'),
    )
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise ValueError(f'invalid gate40 v9 production metadata: {failed}')
    return meta


def build_v9_gate40(
    *,
    source_v6_zip: Path,
    gate40_category_path: Path,
    gate40_hgb_path: Path,
    gate40_metadata_path: Path,
    output_zip: Path,
    v8_graph_source: Path,
    v8_submission_graph_source: Path,
    source_commit: str,
) -> dict[str, object]:
    source_v6_zip = Path(source_v6_zip).resolve(strict=True)
    output_zip = Path(output_zip)
    if len(source_commit) != 40 or any(ch not in '0123456789abcdef' for ch in source_commit.lower()):
        raise ValueError('source_commit must be an exact 40-character hex SHA')
    for path in (gate40_category_path, gate40_hgb_path, v8_graph_source, v8_submission_graph_source):
        p = Path(path)
        if not p.is_file() or p.is_symlink():
            raise ValueError(f'package input must be a regular file: {p}')
    gate_meta = _load_gate40_metadata(Path(gate40_metadata_path))

    with tempfile.TemporaryDirectory(prefix='ecup-v9-gate40-build-') as tmp:
        root = Path(tmp) / 'submission'
        safe_extract_zip(source_v6_zip, root)
        _validate_source_v6(root, SOURCE_V6_METRIC)

        copied = copy_runtime_closure(root)
        ml = root / 'ecup_matching' / 'ml'
        ml.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(v8_graph_source, ml / 'v8_graph.py')
        shutil.copyfile(v8_submission_graph_source, ml / 'v8_submission_graph.py')

        shutil.copyfile(gate40_category_path, root / 'model_v6_category_shrunk.json')
        shutil.copyfile(gate40_hgb_path, root / 'model_v6_hgb_meta.joblib')
        shutil.copyfile(gate40_metadata_path, root / 'model_v6_gate_metadata.json')
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
            for marker in ('_structured_scores_streaming', 'run_structured_chunks', 'torch_autocast'):
                if marker not in text:
                    raise RuntimeError(f'final predict_v6 missing optimized marker: {marker}')

        final_meta = {
            'version': 'v9-gate40-fp16-graph',
            'source_commit': source_commit,
            'source_v6_zip_sha256': _sha256(source_v6_zip),
            'validation': {
                'teacher_coverage': EXPECTED_COVERAGE,
                'base_strict_oof_macro_ap': EXPECTED_BASE_OOF,
                'fold_local_graph_strict_oof_macro_ap': EXPECTED_GRAPH_OOF,
                'target_stress_mean': EXPECTED_TARGET_STRESS,
                'target_stress_ratio': EXPECTED_STRESS_RATIO,
                'split_sha256': EXPECTED_SPLIT_SHA,
                'leaderboard_anchor_v7_observed_by_owner': 0.36,
                'leaderboard_anchor_used_for_fitting': False,
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
                'outer_wall_acceptance_seconds': 700.0,
                'watchdog_seconds': 720.0,
                'closure_files': list(copied),
                'closure_sha256': runtime_hashes,
            },
            'gate40_metadata_sha256': _sha256(Path(gate40_metadata_path)),
            'gate40_category_sha256': _sha256(Path(gate40_category_path)),
            'gate40_hgb_sha256': _sha256(Path(gate40_hgb_path)),
            'selection_basis': 'gate40 Pareto winner over gate25 before final runtime; runtime has veto power',
            'sealed_gold_opened': False,
            'gold_rows_scored': 0,
            'leaderboard_score_claimed': False,
        }
        if abs(float(gate_meta['fold_local_graph_strict_oof_macro_ap']) - final_meta['validation']['fold_local_graph_strict_oof_macro_ap']) > 1e-12:
            raise RuntimeError('v9 evidence changed during packaging')
        (root / 'v9_metadata.json').write_text(
            json.dumps(final_meta, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        _write_zip(root, output_zip)

    archive_bytes = output_zip.stat().st_size
    if archive_bytes >= 5 * 1024**3:
        output_zip.unlink(missing_ok=True)
        raise ValueError(f'v9 gate40 archive exceeds 5 GiB: {archive_bytes}')
    return {
        'output': str(output_zip),
        'archive_bytes': int(archive_bytes),
        'archive_sha256': _sha256(output_zip),
        'coverage': EXPECTED_COVERAGE,
        'base_strict_oof_macro_ap': EXPECTED_BASE_OOF,
        'fold_local_graph_strict_oof_macro_ap': EXPECTED_GRAPH_OOF,
        'target_stress_mean': EXPECTED_TARGET_STRESS,
        'graph_config': dict(GRAPH_CONFIG),
    }


__all__ = ['build_v9_gate40']
