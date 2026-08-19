import numpy as np

from ecup_matching.ml.v5_embeddings import build_embedding_pair_features


def test_embedding_pair_features_are_symmetric_and_finite():
    a = np.array([[1.0, 0.0, 2.0], [0.0, 2.0, 1.0]])
    b = np.array([[0.5, 0.5, 2.0], [1.0, 2.0, 0.0]])

    forward = build_embedding_pair_features(a, b)
    reverse = build_embedding_pair_features(b, a)

    assert forward.shape == (2, 7)
    assert np.isfinite(forward).all()
    assert np.allclose(forward, reverse)


def test_identical_embedding_has_cosine_one_and_zero_distances():
    a = np.array([[1.0, 2.0, 3.0]], dtype=float)
    features = build_embedding_pair_features(a, a)

    assert features[0, 0] == 1.0
    assert features[0, 1] == 0.0
    assert features[0, 2] == 0.0
    assert features[0, 3] == 0.0
