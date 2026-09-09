from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

from src.data import discover_layout, validate_dataset, normalize_extracted_tree


def _png(path: Path, value: int = 128):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (8, 8), (value, value, value)).save(path)


def _mini_dataset(root: Path):
    ids = []
    labels = []
    for label, cls in enumerate(['dark', 'normal', 'bright']):
        image_id = f'{cls}-1'
        ids.append(image_id)
        labels.append(label)
        _png(root / 'train' / cls / f'{image_id}.png', 30 + label * 100)
    test_ids = ['test-1', 'test-2']
    for i, image_id in enumerate(test_ids):
        _png(root / 'test' / f'{image_id}.png', 80 + i * 100)
    pd.DataFrame({'id': ids, 'label': labels}).to_csv(root / 'train.csv', index=False)
    pd.DataFrame({'id': test_ids}).to_csv(root / 'test.csv', index=False)
    pd.DataFrame({'id': test_ids, 'label': [0, 0]}).to_csv(root / 'sample_submission.csv', index=False)


def test_discover_and_validate_matching_dataset(tmp_path: Path):
    _mini_dataset(tmp_path)
    layout = discover_layout(tmp_path)
    report = validate_dataset(layout, expected_train=None, expected_test=None, expected_per_class=None)
    assert report['train_images'] == 3
    assert report['test_images'] == 2
    assert report['class_distribution'] == {0: 1, 1: 1, 2: 1}


def test_validate_rejects_missing_manifest_image(tmp_path: Path):
    _mini_dataset(tmp_path)
    (tmp_path / 'test' / 'test-2.png').unlink()
    with pytest.raises(ValueError, match='test manifest/image mismatch'):
        validate_dataset(discover_layout(tmp_path), expected_train=None, expected_test=None, expected_per_class=None)


def test_normalize_extracted_tree_finds_nested_data(tmp_path: Path):
    nested = tmp_path / 'wrapper' / 'dataset'
    nested.mkdir(parents=True)
    _mini_dataset(nested)
    layout = normalize_extracted_tree(tmp_path)
    assert layout.root == nested


def test_test_manifest_may_have_empty_label_column(tmp_path: Path):
    _mini_dataset(tmp_path)
    test_df = pd.read_csv(tmp_path / 'test.csv')
    test_df['label'] = float('nan')
    test_df.to_csv(tmp_path / 'test.csv', index=False)
    report = validate_dataset(discover_layout(tmp_path), expected_train=None, expected_test=None, expected_per_class=None)
    assert report['test_rows'] == 2
