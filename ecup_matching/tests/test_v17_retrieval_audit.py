"""The audit decides whether five iterations were selected on the wrong axis.

If it mislabels the proxy the conclusion is worse than no measurement, so the
component closure, the covered-endpoint restriction and the weak-label
agreement counts are pinned on a hand-checkable fixture.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ecup_matching.ml.run_v17_retrieval_audit import run_retrieval_audit


CATEGORY = "Электроника"


@pytest.fixture()
def fixture_root(tmp_path):
    items = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5, 6, 7, 8], "category": [CATEGORY] * 8}
    )
    items.to_parquet(tmp_path / "items_human.parquet", index=False)

    # Components {1,2} and {3,4}; 5 and 6 are covered but only by a negative.
    human = pd.DataFrame(
        {
            "id1": [1, 3, 2, 5],
            "id2": [2, 4, 3, 6],
            "target": [1, 1, 0, 0],
        }
    )
    human.to_parquet(tmp_path / "matches.parquet", index=False)

    # (7,8) has uncovered endpoints and must not reach the proxy.
    weak = pd.DataFrame(
        {
            "id1": [1, 1, 2, 1, 7],
            "id2": [3, 4, 4, 2, 8],
            "target": [0.9, 0.2, 0.8, 0.95, 0.5],
        }
    )
    weak.to_parquet(tmp_path / "matches_llm.parquet", index=False)
    return tmp_path


def _run(root, **kwargs):
    return run_retrieval_audit(
        human_matches_path=root / "matches.parquet",
        human_items_path=root / "items_human.parquet",
        weak_matches_path=root / "matches_llm.parquet",
        output_path=root / "audit.json",
        proxy_path=root / "proxy.parquet",
        **kwargs,
    )


def test_proxy_labels_come_from_human_component_closure(fixture_root):
    report = _run(fixture_root)
    proxy = pd.read_parquet(fixture_root / "proxy.parquet")

    # Only pairs whose endpoints human labelling covers may be judged.
    assert report["proxy_candidates_both_covered"] == 4
    assert set(zip(proxy["id1"], proxy["id2"])) == {(1, 3), (1, 4), (2, 4), (1, 2)}

    label = dict(zip(zip(proxy["id1"], proxy["id2"]), proxy["target"]))
    assert label[(1, 2)] == 1  # same component
    assert label[(1, 3)] == 0  # {1,2} vs {3,4}
    assert label[(1, 4)] == 0
    assert label[(2, 4)] == 0
    assert report["proxy"]["prevalence"] == pytest.approx(0.25)


def test_exact_pair_overlap_is_counted_order_independently(fixture_root):
    report = _run(fixture_root)
    # Human holds (1,2); the weak pool holds it too. Nothing else coincides.
    assert report["overlap"]["exact_canonical_pair_overlap"] == 1
    assert report["overlap"]["shared_items"] == 4


def test_weak_label_audit_reports_agreement_against_human_truth(fixture_root):
    audit = _run(fixture_root)["proxy"]["weak_label_audit"]
    # hard positives are (1,3), (2,4), (1,2); only (1,2) is truly a match.
    assert audit["weak_hard_precision"] == pytest.approx(1.0 / 3.0)
    assert audit["weak_hard_recall"] == pytest.approx(1.0)
    assert audit["weak_soft_mean_on_true_positive"] == pytest.approx(0.95)
    assert audit["weak_soft_mean_on_true_negative"] == pytest.approx(
        (0.9 + 0.2 + 0.8) / 3.0
    )


def test_human_degree_separates_all_edges_from_positive_edges(fixture_root):
    human = _run(fixture_root)["human"]
    assert human["rows"] == 4
    assert human["prevalence"] == pytest.approx(0.5)
    assert human["components"]["count"] == 2
    assert human["components"]["items_in_components"] == 4
    # Every item in this fixture carries exactly one positive edge.
    assert human["degree_positive_edges"]["fraction_degree_1"] == pytest.approx(1.0)


def test_missing_weak_pool_still_produces_the_human_half(fixture_root):
    report = run_retrieval_audit(
        human_matches_path=fixture_root / "matches.parquet",
        human_items_path=fixture_root / "items_human.parquet",
        weak_matches_path=fixture_root / "absent.parquet",
        output_path=fixture_root / "audit-nohweak.json",
    )
    assert report["weak"] == {"exists": False}
    assert report["human"]["rows"] == 4
