from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.leakage import phash64, build_duplicate_groups, audit_leakage


def _save(path: Path, value: int, tweak: bool = False):
    arr = np.full((32, 32, 3), value, dtype=np.uint8)
    if tweak:
        arr[0, 0] = min(value + 1, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, 'RGB').save(path)


def test_duplicate_groups_join_identical_images(tmp_path: Path):
    a = tmp_path / 'a.png'
    b = tmp_path / 'b.png'
    c = tmp_path / 'c.png'
    _save(a, 50)
    _save(b, 50)
    _save(c, 220)
    groups = build_duplicate_groups([a, b, c], max_hamming=0)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]
    assert phash64(a) == phash64(b)


def test_audit_has_required_sections(tmp_path: Path):
    train_paths = []
    ids = []
    labels = []
    for label in range(3):
        for i in range(4):
            image_id = f'{label:01d}{i:02d}00000-0000-4000-8000-00000000000{i}'
            p = tmp_path / 'train' / f'{image_id}.png'
            _save(p, 40 + 80 * label, tweak=bool(i % 2))
            train_paths.append(p)
            ids.append(image_id)
            labels.append(label)
    test_paths = []
    test_ids = []
    for i in range(3):
        image_id = f'9000000{i}-0000-4000-8000-00000000000{i}'
        p = tmp_path / 'test' / f'{image_id}.png'
        _save(p, 100 + 20 * i)
        test_paths.append(p)
        test_ids.append(image_id)
    train_df = pd.DataFrame({'id': ids, 'label': labels})
    test_df = pd.DataFrame({'id': test_ids})
    report = audit_leakage(train_df, test_df, train_paths, test_paths, cv_splits=3)
    for key in ['uuid', 'metadata', 'duplicates', 'ordering', 'split_reconstruction']:
        assert key in report
    assert 0.0 <= report['uuid']['cv_accuracy'] <= 1.0
    assert 0.0 <= report['metadata']['cv_accuracy'] <= 1.0
