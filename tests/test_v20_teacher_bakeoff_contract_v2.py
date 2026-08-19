import pandas as pd

from ecup_matching.ml.v20_teacher_bakeoff import score_pair, select_teacher_pair


def _teacher_report(name: str, family: str):
    return {
        "model_id": name,
        "revision": "a" * 40,
        "family": family,
        "backend": "test",
        "quantization": "none",
        "eligible": True,
        "rows_per_second": 10.0,
    }


def test_score_pair_output_can_be_selected_without_schema_rewrite():
    truth = pd.DataFrame([
        {"id1": i, "id2": i + 10000, "target": i % 2, "reason_code": "OTHER", "category": "c"}
        for i in range(300)
    ])
    first = truth[["id1", "id2"]].copy()
    first["pred"] = truth["target"]
    first["valid"] = True
    first["uncertain"] = False
    first["reason_code"] = "OTHER"
    second = first.copy()

    qwen = _teacher_report("qwen", "qwen")
    euro = _teacher_report("euro", "eurollm")
    pair = score_pair(truth, first, second, qwen, euro)

    assert pair["teachers"] == ["qwen", "euro"]
    pair["eligible"] = True
    selection = select_teacher_pair({"qwen": qwen, "euro": euro}, [pair])
    assert selection["selected"] == ["qwen", "euro"]
