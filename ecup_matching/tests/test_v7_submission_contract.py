"""The v7 submission must satisfy the organizer contract end to end.

v7 scores pairs with the cross-encoder alone, so there is no structured phase
and no meta fusion: the archive contract is much smaller than v5/v6, but the
output schema and CLI are identical and are what the platform validates.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _items(count: int) -> pd.DataFrame:
    brands = ["samsung", "xiaomi", "bosch"]
    return pd.DataFrame(
        {
            "id": np.arange(count, dtype=np.int64),
            "name": [f"смартфон {brands[i % 3]} sm-s{900 + i}b {128 * (1 + i % 3)} гб" for i in range(count)],
            "attributes": [
                json.dumps(
                    {
                        "Бренд": [brands[i % 3]],
                        "Объем встроенной памяти": [f"{128 * (1 + i % 3)} ГБ"],
                        "Цвет товара": ["черный" if i % 2 else "белый"],
                    },
                    ensure_ascii=False,
                )
                for i in range(count)
            ],
            "category": ["Электроника" for _ in range(count)],
        }
    )


def _pairs(item_count: int, count: int) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "id1": rng.integers(0, item_count, count).astype(np.int64),
            "id2": rng.integers(0, item_count, count).astype(np.int64),
        }
    )


def _corpus_vocabulary() -> list[str]:
    """Cover every word the fixtures actually produce.

    A vocabulary that maps everything to [UNK] would make the model emit one
    constant score, which would hide exactly the failure these tests check for.
    """
    import re

    from ecup_matching.ml.textnorm import normalize_item
    from ecup_matching.ml.v7_item_text import serialize_item_v7

    items = _items(80)
    words: set[str] = set()
    for row in items.itertuples(index=False):
        norm = normalize_item(row.id, row.name, row.attributes, row.category)
        text = f"[CAT] {norm.category}\n{serialize_item_v7(norm, max_chars=900)}"
        words.update(re.findall(r"\w+", text.lower()))
    return sorted(words)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A real but minimal sequence-classification checkpoint on disk."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoModelForSequenceClassification, BertConfig, BertTokenizerFast

    path = tmp_path_factory.mktemp("v7model")
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *_corpus_vocabulary()]
    vocab = path / "vocab.txt"
    vocab.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True).save_pretrained(path)

    config = BertConfig(
        vocab_size=len(tokens),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
        num_labels=1,
    )
    AutoModelForSequenceClassification.from_config(config).save_pretrained(
        path, safe_serialization=True
    )
    return path


def test_predict_to_csv_v7_writes_the_organizer_schema(tmp_path, tiny_model):
    pytest.importorskip("torch")
    from ecup_matching.submission.predict_v7 import predict_to_csv_v7

    items = _items(40)
    pairs = _pairs(40, 25)
    items_path = tmp_path / "items.parquet"
    matches_path = tmp_path / "matches.parquet"
    output_path = tmp_path / "submit.csv"
    items.to_parquet(items_path, index=False)
    pairs.to_parquet(matches_path, index=False)

    result = predict_to_csv_v7(
        items_path=items_path,
        matches_path=matches_path,
        model_dir=tiny_model,
        output_path=output_path,
        batch_size=8,
    )

    assert output_path.is_file()
    written = pd.read_csv(output_path)
    assert list(written.columns) == ["id1", "id2", "predict"]
    assert len(written) == len(pairs)
    assert written["id1"].tolist() == pairs["id1"].tolist()
    assert written["id2"].tolist() == pairs["id2"].tolist()
    assert np.isfinite(written["predict"]).all()
    assert written["predict"].between(0.0, 1.0).all()
    assert result["predict"].nunique() > 1, "a constant score would fail the result stage"


def test_predict_to_csv_v7_preserves_input_pair_order(tmp_path, tiny_model):
    pytest.importorskip("torch")
    from ecup_matching.submission.predict_v7 import predict_to_csv_v7

    items = _items(30)
    pairs = _pairs(30, 40)
    items_path = tmp_path / "items.parquet"
    matches_path = tmp_path / "matches.parquet"
    items.to_parquet(items_path, index=False)
    pairs.to_parquet(matches_path, index=False)

    full = predict_to_csv_v7(
        items_path=items_path,
        matches_path=matches_path,
        model_dir=tiny_model,
        output_path=tmp_path / "a.csv",
        batch_size=7,
    )
    rebatched = predict_to_csv_v7(
        items_path=items_path,
        matches_path=matches_path,
        model_dir=tiny_model,
        output_path=tmp_path / "b.csv",
        batch_size=16,
    )
    np.testing.assert_allclose(
        full["predict"].to_numpy(), rebatched["predict"].to_numpy(), rtol=0, atol=1e-6
    )


def test_missing_item_is_reported_not_silently_scored(tmp_path, tiny_model):
    pytest.importorskip("torch")
    from ecup_matching.submission.predict_v7 import predict_to_csv_v7

    items = _items(10)
    items_path = tmp_path / "items.parquet"
    matches_path = tmp_path / "matches.parquet"
    items.to_parquet(items_path, index=False)
    pd.DataFrame({"id1": [0], "id2": [999]}).to_parquet(matches_path, index=False)

    with pytest.raises(KeyError, match="missing"):
        predict_to_csv_v7(
            items_path=items_path,
            matches_path=matches_path,
            model_dir=tiny_model,
            output_path=tmp_path / "out.csv",
        )


def test_run_v7_entrypoint_accepts_the_organizer_cli(tmp_path, tiny_model):
    pytest.importorskip("torch")
    items = _items(20)
    pairs = _pairs(20, 12)
    items_path = tmp_path / "items.parquet"
    matches_path = tmp_path / "matches.parquet"
    output_path = tmp_path / "submit.csv"
    items.to_parquet(items_path, index=False)
    pairs.to_parquet(matches_path, index=False)

    submission = tmp_path / "submission"
    (submission / "model_v7_teacher").mkdir(parents=True)
    for file in tiny_model.iterdir():
        (submission / "model_v7_teacher" / file.name).write_bytes(file.read_bytes())
    (submission / "model_v7_metadata.json").write_text(
        json.dumps(
            {
                "candidate": "v7-identity-first-rubert-base",
                "max_length": 256,
                "max_chars": 900,
                "inference_batch_size": 8,
                "diagnostic_fold0_macro_average_precision": 0.6791967999009738,
                "strict_oof_macro_average_precision": None,
                "gold_metric_opened": False,
                "gold_rows_scored": 0,
            }
        ),
        encoding="utf-8",
    )
    run_py = submission / "run.py"
    run_py.write_bytes((REPO_ROOT / "ecup_matching" / "submission" / "run_v7.py").read_bytes())

    env = {"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            str(run_py),
            "--items_path",
            str(items_path),
            "--matches_path",
            str(matches_path),
            "--output_path",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    written = pd.read_csv(output_path)
    assert list(written.columns) == ["id1", "id2", "predict"]
    assert len(written) == len(pairs)


def test_metadata_never_advertises_a_diagnostic_score_as_strict_oof():
    """A fold-0 probe number must not be recorded as validated OOF."""
    from ecup_matching.submission.predict_v7 import validate_v7_metadata

    with pytest.raises(ValueError, match="strict_oof"):
        validate_v7_metadata(
            {
                "strict_oof_macro_average_precision": 0.6791967999009738,
                "diagnostic_fold0_macro_average_precision": 0.6791967999009738,
                "gold_metric_opened": False,
                "gold_rows_scored": 0,
            }
        )

    ok = validate_v7_metadata(
        {
            "strict_oof_macro_average_precision": None,
            "diagnostic_fold0_macro_average_precision": 0.6791967999009738,
            "gold_metric_opened": False,
            "gold_rows_scored": 0,
        }
    )
    assert ok["diagnostic_fold0_macro_average_precision"] == 0.6791967999009738


def test_metadata_rejects_an_opened_sealed_gold():
    from ecup_matching.submission.predict_v7 import validate_v7_metadata

    with pytest.raises(ValueError, match="sealed gold"):
        validate_v7_metadata(
            {
                "strict_oof_macro_average_precision": None,
                "diagnostic_fold0_macro_average_precision": 0.5,
                "gold_metric_opened": True,
                "gold_rows_scored": 10,
            }
        )
