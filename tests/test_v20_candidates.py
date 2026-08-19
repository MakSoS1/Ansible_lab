import pandas as pd

from ecup_matching.ml.v20_candidates import generate_candidate_pairs


def test_candidate_generation_is_target_free_and_canonical():
    items = pd.DataFrame([
        {"id": 1, "name": "iPhone 15 Pro 256 GB", "attributes": '{"brand":"Apple"}', "category": "Электроника"},
        {"id": 2, "name": "Apple iPhone 15 Pro 512 GB", "attributes": '{"brand":"Apple"}', "category": "Электроника"},
        {"id": 3, "name": "Чехол Apple iPhone 15 Pro", "attributes": '{"brand":"Apple"}', "category": "Электроника"},
    ])
    out, report = generate_candidate_pairs(items, forbidden_ids=set(), max_degree=3, max_pairs_per_reason=20)
    assert "target" not in out.columns
    assert all(str(a) < str(b) for a, b in out[["id1", "id2"]].itertuples(index=False, name=None))
    assert len(out.drop_duplicates(["id1", "id2"])) == len(out)
    assert report["forbidden_rows"] == 0


def test_forbidden_endpoint_never_appears():
    items = pd.DataFrame([
        {"id": 1, "name": "Sony WH-1000XM5", "attributes": "{}", "category": "Электроника"},
        {"id": 2, "name": "Sony WH1000XM5", "attributes": "{}", "category": "Электроника"},
        {"id": 3, "name": "Sony WH-1000XM4", "attributes": "{}", "category": "Электроника"},
    ])
    out, _ = generate_candidate_pairs(items, forbidden_ids={2}, max_degree=5, max_pairs_per_reason=20)
    assert 2 not in set(out.id1) | set(out.id2)


def test_degree_cap_is_enforced():
    items = pd.DataFrame([
        {"id": i, "name": f"Brand ModelX variant {i}", "attributes": '{"brand":"Brand"}', "category": "x"}
        for i in range(1, 8)
    ])
    out, _ = generate_candidate_pairs(items, forbidden_ids=set(), max_degree=2, max_pairs_per_reason=100)
    degree = pd.concat([out.id1, out.id2]).value_counts()
    assert degree.max() <= 2
