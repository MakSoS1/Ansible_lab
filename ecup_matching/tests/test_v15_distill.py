import pandas as pd

from ecup_matching.v15_distill import select_unlabelled_candidates


class RecordingReader:
    def __init__(self, frame):
        self.frame = frame
        self.columns = None

    def read(self, columns):
        self.columns = tuple(columns)
        if "target" in columns:
            raise AssertionError("legacy target must not be requested")
        return self.frame.loc[:, list(columns)].copy()


def test_v15_distill_candidate_selection_never_reads_legacy_target():
    source = pd.DataFrame({"id1": [1, 2, 3], "id2": [4, 5, 6], "target": [0.99, 0.01, 0.5]})
    reader = RecordingReader(source)
    selected = select_unlabelled_candidates(reader, excluded_item_ids=set(), limit=10)
    assert reader.columns == ("id1", "id2")
    assert list(selected.columns) == ["id1", "id2"]


def test_v15_distill_excludes_every_protected_item_endpoint():
    source = pd.DataFrame({"id1": [1, 2, 3, 7], "id2": [4, 5, 6, 8], "target": [0, 0, 0, 0]})
    reader = RecordingReader(source)
    selected = select_unlabelled_candidates(reader, excluded_item_ids={1, 5, 8}, limit=10)
    assert set(selected["id1"]).isdisjoint({1, 5, 8})
    assert set(selected["id2"]).isdisjoint({1, 5, 8})
    assert {(int(r.id1), int(r.id2)) for r in selected.itertuples()} == {(3, 6)}
