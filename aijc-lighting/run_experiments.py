from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import discover_layout, load_manifests, resolve_image_paths, validate_dataset, write_validation_report
from src.ensemble import select_best, write_submission
from src.features import extract_feature_table
from src.leakage import audit_leakage, build_duplicate_groups, write_audit
from src.solutions import run_solution_1, run_solution_2, run_solution_3, run_solution_4, run_solution_5


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _feature_cache(paths, mode: str, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache)
    frame = extract_feature_table(paths, mode=mode)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--vision-mode', choices=['none', 'embeddings', 'full'], default='full')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fast-tabular', action='store_true')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    layout = discover_layout(args.data_dir)
    validation_report = validate_dataset(layout)
    write_validation_report(validation_report, args.output_dir / 'dataset_validation.json')
    train_df, test_df, _ = load_manifests(layout)
    train_paths, test_paths = resolve_image_paths(layout)
    y = train_df['label'].to_numpy(dtype=int)

    print('Dataset:', validation_report, flush=True)
    print('Running leakage audit...', flush=True)
    leakage = audit_leakage(train_df, test_df, train_paths, test_paths, cv_splits=min(5, args.folds))
    write_audit(leakage, args.output_dir / 'leakage_audit.json')
    print(json.dumps(leakage, indent=2, ensure_ascii=False), flush=True)

    groups = build_duplicate_groups(train_paths, max_hamming=4)
    counts = Counter(groups.tolist())
    duplicate_samples = sum(v for v in counts.values() if v > 1)
    use_groups = groups if duplicate_samples > 0 else None
    print(f'Perceptual groups={len(counts)}, samples in non-singletons={duplicate_samples}', flush=True)

    print('Extracting/caching image physics...', flush=True)
    global_train = _feature_cache(train_paths, 'global', cache_dir / 'train_global.csv')
    global_test = _feature_cache(test_paths, 'global', cache_dir / 'test_global.csv')
    spatial_train = _feature_cache(train_paths, 'spatial', cache_dir / 'train_spatial.csv')
    spatial_test = _feature_cache(test_paths, 'spatial', cache_dir / 'test_spatial.csv')
    all_train = pd.concat([global_train, spatial_train], axis=1)
    all_test = pd.concat([global_test, spatial_test], axis=1)
    all_train = all_train.loc[:, ~all_train.columns.duplicated()]
    all_test = all_test[all_train.columns]

    results = []
    durations = {}

    def timed(name, fn):
        t0 = time.time()
        result = fn()
        durations[name] = time.time() - t0
        results.append(result)
        print(f'{name}: OOF={result.oof_accuracy:.6f} model={result.name} time={durations[name]:.1f}s', flush=True)
        return result

    timed('solution_1', lambda: run_solution_1(
        global_train, global_test, y, groups=use_groups, n_splits=args.folds, seed=args.seed))
    timed('solution_2', lambda: run_solution_2(
        all_train, all_test, y, groups=use_groups, n_splits=args.folds, seed=args.seed, fast=args.fast_tabular))
    timed('solution_3', lambda: run_solution_3(
        spatial_train, spatial_test, y, groups=use_groups, n_splits=args.folds, seed=args.seed, fast=args.fast_tabular))

    if args.vision_mode in {'embeddings', 'full'}:
        timed('solution_4', lambda: run_solution_4(
            train_paths,
            test_paths,
            y,
            physics_train=global_train,
            physics_test=global_test,
            groups=use_groups,
            n_splits=args.folds,
            seed=args.seed,
            cache_dir=cache_dir,
            backbone=os.getenv('AIJC_EMBED_BACKBONE', 'efficientnet_b0'),
        ))
    else:
        print('Solution 4 skipped by --vision-mode none', flush=True)

    if args.vision_mode == 'full':
        timed('solution_5', lambda: run_solution_5(
            train_paths,
            test_paths,
            y,
            groups=use_groups,
            n_splits=int(os.getenv('AIJC_CNN_FOLDS', '3')),
            seed=args.seed,
            backbone=os.getenv('AIJC_CNN_BACKBONE', 'efficientnet_b0'),
            epochs=int(os.getenv('AIJC_CNN_EPOCHS', '8')),
            batch_size=int(os.getenv('AIJC_CNN_BATCH', '16')),
        ))
    else:
        print('Solution 5 skipped unless --vision-mode full', flush=True)

    if len(results) < 3:
        raise RuntimeError('not enough completed candidates')

    selection = select_best(results, y)
    best = selection['result']
    print(f'BEST: {best.name} OOF={best.oof_accuracy:.6f}', flush=True)

    test_ids = test_df['id'].astype(str).tolist()
    for idx, result in enumerate(results, 1):
        write_submission(test_ids, result.test_probs, args.output_dir / f'solution_{idx}.csv')
    write_submission(test_ids, best.test_probs, args.output_dir / 'submission_best.csv')

    arrays = {}
    metrics_candidates = []
    for i, result in enumerate(results, 1):
        arrays[f'oof_{i}'] = result.oof_probs
        arrays[f'test_{i}'] = result.test_probs
        pred = result.oof_probs.argmax(1)
        metrics_candidates.append({
            'slot': i,
            'name': result.name,
            'oof_accuracy': float(result.oof_accuracy),
            'fold_scores': [float(x) for x in result.fold_scores],
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            'test_distribution': {str(int(k)): int(v) for k, v in Counter(result.test_probs.argmax(1)).items()},
            'metadata': _jsonable(result.metadata),
            'duration_seconds': float(durations.get(f'solution_{i}', 0.0)),
        })
    arrays['oof_best'] = best.oof_probs
    arrays['test_best'] = best.test_probs
    np.savez_compressed(args.output_dir / 'oof_predictions.npz', **arrays)

    metrics = {
        'validation': validation_report,
        'grouping': {
            'used_group_aware_cv': use_groups is not None,
            'n_groups': int(len(counts)),
            'samples_in_duplicate_groups': int(duplicate_samples),
        },
        'candidates': metrics_candidates,
        'selection': {
            'base_best': selection['base_best'],
            'base_best_accuracy': selection['base_best_accuracy'],
            'selected_name': selection['selected_name'],
            'selected_accuracy': selection['selected_accuracy'],
            'selected_test_distribution': {
                str(int(k)): int(v) for k, v in Counter(best.test_probs.argmax(1)).items()
            },
        },
    }
    (args.output_dir / 'metrics.json').write_text(
        json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False), encoding='utf-8')

    print('\nScore table:', flush=True)
    for row in sorted(metrics_candidates, key=lambda x: x['oof_accuracy'], reverse=True):
        print(f"  {row['slot']}: {row['oof_accuracy']:.6f} {row['name']}", flush=True)
    print(f"  selected: {best.oof_accuracy:.6f} {best.name}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
