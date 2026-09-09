from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import uuid
import zlib

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

COMMON_SEEDS = [0, 1, 7, 13, 42, 99, 2024, 2025, 2026]


def phash64(path: Path | str) -> int:
    with Image.open(path) as image:
        gray = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    dct = cv2.dct(gray)
    low = dct[:8, :8].copy()
    values = low.flatten()[1:]
    median = float(np.median(values))
    bits = (low.flatten() >= median).astype(np.uint8)
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def _hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def build_duplicate_groups(paths: list[Path | str], max_hamming: int = 4) -> np.ndarray:
    hashes = [phash64(p) for p in paths]
    exact_hashes = [hashlib.sha256(Path(p).read_bytes()).digest() for p in paths]
    parent = list(range(len(paths)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if max_hamming == 0:
        exact_buckets: dict[bytes, list[int]] = {}
        for i, h in enumerate(exact_hashes):
            for j in exact_buckets.get(h, []):
                union(i, j)
            exact_buckets.setdefault(h, []).append(i)
    else:
        for i in range(len(hashes)):
            hi = hashes[i]
            for j in range(i):
                if _hamming(hi, hashes[j]) <= max_hamming:
                    union(i, j)

    root_to_group: dict[int, int] = {}
    groups = np.empty(len(paths), dtype=int)
    next_group = 0
    for i in range(len(paths)):
        root = find(i)
        if root not in root_to_group:
            root_to_group[root] = next_group
            next_group += 1
        groups[i] = root_to_group[root]
    return groups


def _uuid_features(ids: pd.Series) -> np.ndarray:
    rows = []
    for raw in ids.astype(str):
        compact = raw.replace('-', '')
        try:
            u = uuid.UUID(raw)
            value = u.int
            chunks = [
                (value >> 96) & 0xFFFFFFFF,
                (value >> 64) & 0xFFFFFFFF,
                (value >> 32) & 0xFFFFFFFF,
                value & 0xFFFFFFFF,
                int(u.version or 0),
                int(u.variant == uuid.RFC_4122),
            ]
        except Exception:
            crc = zlib.crc32(raw.encode('utf-8')) & 0xFFFFFFFF
            chunks = [crc, len(raw), sum(map(ord, raw))]
        try:
            raw_bytes = bytes.fromhex(compact[:32].ljust(32, '0'))
            prefix_bytes = list(raw_bytes[:16])
        except ValueError:
            prefix_bytes = [ord(c) % 256 for c in raw[:16].ljust(16, '\0')]
        rows.append(chunks + prefix_bytes)
    max_len = max(len(r) for r in rows)
    return np.array([r + [0] * (max_len - len(r)) for r in rows], dtype=np.float64)


def _metadata_features(paths: list[Path | str]) -> np.ndarray:
    rows = []
    for path in paths:
        path = Path(path)
        stat = path.stat()
        data = path.read_bytes()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            fmt = image.format or ''
            info_keys = sorted(image.info.keys())
        entropy = 0.0
        if data:
            counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(float)
            p = counts[counts > 0] / len(data)
            entropy = float(-(p * np.log2(p)).sum())
        rows.append([
            stat.st_size,
            width,
            height,
            width / max(height, 1),
            zlib.crc32(mode.encode()) & 0xFFFF,
            zlib.crc32(fmt.encode()) & 0xFFFF,
            zlib.crc32('|'.join(info_keys).encode()) & 0xFFFF,
            entropy,
            zlib.crc32(data[:4096]) & 0xFFFFFFFF,
            zlib.crc32(data[-4096:]) & 0xFFFFFFFF,
        ])
    return np.asarray(rows, dtype=np.float64)


def _safe_cv_accuracy(X: np.ndarray, y: np.ndarray, splits: int) -> float:
    counts = np.bincount(y, minlength=3)
    splits = int(max(2, min(splits, counts[counts > 0].min())))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    model = ExtraTreesClassifier(
        n_estimators=160,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
    )
    return float(cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=1).mean())


def _ordering_report(train_df: pd.DataFrame) -> dict:
    labels = train_df['label'].astype(int).to_numpy()
    transitions = int(np.sum(labels[1:] != labels[:-1])) if len(labels) > 1 else 0
    sorted_fraction_by_class = {}
    sorted_prefix_by_class = {}
    for label in sorted(np.unique(labels)):
        ids = train_df.loc[train_df['label'] == label, 'id'].astype(str).tolist()
        if len(ids) <= 1:
            sorted_fraction = 1.0
            prefix = len(ids)
        else:
            sorted_fraction = float(np.mean([a <= b for a, b in zip(ids, ids[1:])]))
            prefix = 1
            while prefix < len(ids) and ids[prefix - 1] <= ids[prefix]:
                prefix += 1
        sorted_fraction_by_class[str(int(label))] = sorted_fraction
        sorted_prefix_by_class[str(int(label))] = int(prefix)
    return {
        'label_transitions': transitions,
        'class_blocked': bool(transitions <= 2),
        'sorted_adjacent_fraction_by_class': sorted_fraction_by_class,
        'sorted_prefix_by_class': sorted_prefix_by_class,
    }


def _split_reconstruction_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_counts = train_df['label'].astype(int).value_counts().sort_index().to_dict()
    n_train, n_test = len(train_df), len(test_df)
    candidates = []
    if set(train_counts.keys()) == {0, 1, 2} and len(set(train_counts.values())) == 1:
        per_train = next(iter(train_counts.values()))
        if n_test % 3 == 0:
            per_test = n_test // 3
            candidates.append({
                'hypothesis': 'balanced_per_class_total',
                'full_counts': {0: per_train + per_test, 1: per_train + per_test, 2: per_train + per_test},
                'implied_test_counts': {0: per_test, 1: per_test, 2: per_test},
                'compatible_with_counts': True,
            })
    return {
        'n_total': n_train + n_test,
        'train_counts': {str(int(k)): int(v) for k, v in train_counts.items()},
        'candidate_common_random_seeds': COMMON_SEEDS,
        'candidates': candidates,
        'note': (
            'Counts alone cannot recover test labels because UUIDs do not expose the original pre-split index. '
            'A reconstruction is only actionable if ordering or another reversible index is discovered.'
        ),
    }


def audit_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_paths: list[Path | str],
    test_paths: list[Path | str],
    cv_splits: int = 5,
) -> dict:
    y = train_df['label'].astype(int).to_numpy()
    uuid_X = _uuid_features(train_df['id'])
    metadata_X = _metadata_features(train_paths)

    train_hashes = [phash64(p) for p in train_paths]
    test_hashes = [phash64(p) for p in test_paths]
    groups = build_duplicate_groups(train_paths, max_hamming=4)
    group_counts = Counter(groups.tolist())
    duplicate_group_sizes = sorted([int(v) for v in group_counts.values() if v > 1], reverse=True)

    nearest_test_distances = []
    if train_hashes:
        for h in test_hashes:
            nearest_test_distances.append(min(_hamming(h, th) for th in train_hashes))

    return {
        'uuid': {
            'cv_accuracy': _safe_cv_accuracy(uuid_X, y, cv_splits),
            'n_features': int(uuid_X.shape[1]),
        },
        'metadata': {
            'cv_accuracy': _safe_cv_accuracy(metadata_X, y, cv_splits),
            'n_features': int(metadata_X.shape[1]),
        },
        'duplicates': {
            'train_groups': int(len(np.unique(groups))),
            'duplicate_group_sizes': duplicate_group_sizes[:50],
            'max_group_size': int(max(group_counts.values(), default=1)),
            'test_nearest_phash_min': int(min(nearest_test_distances)) if nearest_test_distances else None,
            'test_nearest_phash_median': float(np.median(nearest_test_distances)) if nearest_test_distances else None,
            'test_exact_or_phash_near_4': int(sum(d <= 4 for d in nearest_test_distances)),
        },
        'ordering': _ordering_report(train_df),
        'split_reconstruction': _split_reconstruction_report(train_df, test_df),
    }


def write_audit(report: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return path
