from ecup_matching.ml.v20_proxy import calibrate_proxy_axes


def test_proxy_rejects_human_fold_like_misranking():
    axes = {
        "human_fold0": {
            "higher_is_better": True,
            "values": {"v7": 0.7023, "v12": 0.7059, "v13B": 0.7086, "v14": 0.7065},
        }
    }
    report = calibrate_proxy_axes(axes)
    assert report["axes"]["human_fold0"]["promotable"] is False


def test_proxy_accepts_exact_public_order_and_lower_is_better_axis():
    axes = {
        "weak_ap": {
            "higher_is_better": True,
            "values": {"v14": 0.60, "v12": 0.59, "v13B": 0.58, "v7": 0.50},
        },
        "brier": {
            "higher_is_better": False,
            "values": {"v14": 0.10, "v12": 0.11, "v13B": 0.12, "v7": 0.20},
        },
    }
    report = calibrate_proxy_axes(axes)
    assert report["axes"]["weak_ap"]["promotable"] is True
    assert report["axes"]["brier"]["promotable"] is True
    assert set(report["promotable_axes"]) == {"weak_ap", "brier"}
