import numpy as np


def test_replay_balanced_three_class_split_matches_known_seed42_prefix():
    from src.structural_split import replay_balanced_test_labels

    labels = replay_balanced_test_labels(seed=42, per_class_total=600, test_size=300)
    assert labels.shape == (300,)
    assert np.bincount(labels, minlength=3).tolist() == [100, 100, 100]
    assert labels[:12].tolist() == [0, 0, 1, 2, 1, 2, 0, 1, 0, 0, 2, 0]


def test_seed_scan_recovers_synthetic_generator_seed():
    from src.structural_split import replay_balanced_test_labels, rank_candidate_seeds

    target = replay_balanced_test_labels(seed=7, per_class_total=600, test_size=300)
    ranked = rank_candidate_seeds(target, seeds=[0, 1, 7, 13, 42], per_class_total=600, test_size=300)
    assert ranked[0]["seed"] == 7
    assert ranked[0]["agreement"] == 1.0


def test_seed_scan_accepts_multiple_proxy_predictions():
    from src.structural_split import replay_balanced_test_labels, rank_candidate_seeds

    target = replay_balanced_test_labels(seed=13, per_class_total=600, test_size=300)
    proxy_a = target.copy()
    proxy_b = target.copy()
    proxy_b[:30] = (proxy_b[:30] + 1) % 3
    ranked = rank_candidate_seeds(
        [proxy_a, proxy_b],
        seeds=[7, 13, 42],
        per_class_total=600,
        test_size=300,
    )
    assert ranked[0]["seed"] == 13
    assert ranked[0]["mean_agreement"] > 0.9


def test_original_manifest_permutation_is_complete_and_matches_uploaded_prefix():
    from src.original_manifest_order import ORIGINAL_TEST_SORTED_INDEX, restore_original_test_order

    assert len(ORIGINAL_TEST_SORTED_INDEX) == 300
    assert sorted(ORIGINAL_TEST_SORTED_INDEX) == list(range(300))
    sorted_ids = [f"id-{i:03d}" for i in range(300)]
    restored = restore_original_test_order(sorted_ids)
    assert restored[:5] == ["id-054", "id-087", "id-090", "id-178", "id-074"]
