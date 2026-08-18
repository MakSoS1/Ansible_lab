import pandas as pd

from ecup_matching.ml.v20_corpus import build_gold_corpus, source_reliability_weight


def test_source_weights_never_outrank_human():
    assert source_reliability_weight("human", 1.0, admitted=True) == 1.0
    assert 0 < source_reliability_weight("historical_weak", 1.0, admitted=True) < 1.0
    assert 0 < source_reliability_weight("generated_llm", 1.0, admitted=True) < 1.0
    assert source_reliability_weight("generated_llm", 1.0, admitted=False) == 0.0
    assert source_reliability_weight("uncertain", 1.0, admitted=False) == 0.0


def test_gold_corpus_excludes_forbidden_and_zero_weight_rows():
    human = pd.DataFrame([
        {"id1": 1, "id2": 2, "target": 1, "category": "x", "reason_code": "SAME_MODEL"},
    ])
    weak = pd.DataFrame([
        {"id1": 3, "id2": 4, "target": 0.1, "category": "x", "reason_code": "MODEL_CONFLICT", "weak_weight": 0.7, "stratum_reliability": 0.99},
        {"id1": 5, "id2": 6, "target": 0.9, "category": "x", "reason_code": "SAME_MODEL", "weak_weight": 0.7, "stratum_reliability": 0.99},
    ])
    generated = pd.DataFrame([
        {"id1": 7, "id2": 8, "target": 1, "category": "x", "reason_code": "SAME_MODEL", "admitted": True, "stratum_reliability": 0.999},
        {"id1": 9, "id2": 10, "target": 0, "category": "x", "reason_code": "MODEL_CONFLICT", "admitted": False, "stratum_reliability": 1.0},
    ])
    gold, report = build_gold_corpus(human, weak, generated, forbidden_ids={6}, seed=2026)
    assert 6 not in set(gold.id1) | set(gold.id2)
    assert 9 not in set(gold.id1) | set(gold.id2)
    assert set(gold.source) == {"human", "historical_weak", "generated_llm"}
    assert gold.loc[gold.source == "human", "match_weight"].iloc[0] == 1.0
    assert report["forbidden_rows_removed"] == 1


def test_gold_has_sampling_key_for_category_class_reason_balance():
    human = pd.DataFrame([
        {"id1": 1, "id2": 2, "target": 1, "category": "x", "reason_code": "SAME_MODEL"},
        {"id1": 3, "id2": 4, "target": 0, "category": "x", "reason_code": "MODEL_CONFLICT"},
    ])
    gold, _ = build_gold_corpus(human, pd.DataFrame(), pd.DataFrame(), forbidden_ids=set(), seed=1)
    assert "balance_key" in gold
    assert set(gold.balance_key) == {"x|1|SAME_MODEL", "x|0|MODEL_CONFLICT"}
