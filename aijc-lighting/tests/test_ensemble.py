from pathlib import Path

import numpy as np
import pandas as pd

from src.ensemble import select_best, write_submission
from src.solutions import CVResult


def _result(name, probs, test_probs):
    y = np.array([0, 1, 2, 0, 1, 2])
    return CVResult(
        name=name,
        oof_probs=np.asarray(probs, dtype=float),
        test_probs=np.asarray(test_probs, dtype=float),
        fold_scores=[],
        oof_accuracy=float((np.asarray(probs).argmax(1) == y).mean()),
    )


def test_select_best_prefers_higher_oof_accuracy():
    y = np.array([0, 1, 2, 0, 1, 2])
    perfect = np.eye(3)[y] * 0.9 + 0.1 / 3
    weak = np.roll(perfect, 1, axis=1)
    a = _result('a', perfect, np.tile([[0.8, 0.1, 0.1]], (4, 1)))
    b = _result('b', weak, np.tile([[0.1, 0.8, 0.1]], (4, 1)))
    chosen = select_best([a, b], y)
    assert chosen['result'].oof_accuracy == 1.0
    assert chosen['result'].name == 'a'


def test_write_submission_preserves_id_order_and_schema(tmp_path: Path):
    ids = [f'id-{i:03d}' for i in range(300)]
    probs = np.zeros((300, 3), dtype=float)
    probs[np.arange(300), np.arange(300) % 3] = 1.0
    out = write_submission(ids, probs, tmp_path / 'submission.csv', expected_rows=300)
    df = pd.read_csv(out)
    assert df.columns.tolist() == ['id', 'label']
    assert df['id'].tolist() == ids
    assert set(df['label']) == {0, 1, 2}
    assert len(df) == 300
