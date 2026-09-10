import numpy as np


def test_cosine_knn_excludes_self_and_predicts_nearest_label():
    from src.semantic_neighbors import cosine_knn_probabilities

    x = np.array([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]], dtype=np.float32)
    y = np.array([0, 0, 2, 2])
    probs = cosine_knn_probabilities(x, y, x, k=1, exclude_self=True)
    assert probs.shape == (4, 3)
    assert probs.argmax(1).tolist() == y.tolist()


def test_multiview_similarity_is_average_of_normalized_views():
    from src.semantic_neighbors import fuse_normalized_views

    a = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 2.0], [0.0, 1.0]], dtype=np.float32)
    fused = fuse_normalized_views([a, b])
    assert fused.shape == (2, 4)
    norms = np.linalg.norm(fused, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_neighbor_margin_detects_unambiguous_match():
    from src.semantic_neighbors import nearest_neighbor_margin

    similarities = np.array([[0.95, 0.70, 0.20], [0.80, 0.79, 0.10]], dtype=np.float32)
    margin = nearest_neighbor_margin(similarities)
    assert np.allclose(margin, [0.25, 0.01], atol=1e-6)


def test_model_input_size_uses_backbone_contract_instead_of_fixed_224():
    from src.semantic_neighbors import model_input_size

    class FakePatchEmbed:
        img_size = (518, 518)

    class FakeModel:
        patch_embed = FakePatchEmbed()

    assert model_input_size(FakeModel()) == 518
