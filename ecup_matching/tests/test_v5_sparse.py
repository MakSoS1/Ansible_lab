import numpy as np
import pandas as pd

from ecup_matching.ml.v5_sparse import fit_sparse_item_encoder, transform_sparse_pairs


def _items():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": [
                "Samsung Galaxy S24 SM-S921B 256GB black",
                "Samsung Galaxy S24 SM-S921B 256 GB черный",
                "Samsung Galaxy S24 SM-S921B 128GB black",
                "Apple iPhone 15 256GB black",
            ],
            "attributes": [
                '{"brand":"Samsung","model":"SM-S921B","memory":"256 GB"}',
                '{"brand":"Samsung","model":"SM-S921B","memory":"256 GB"}',
                '{"brand":"Samsung","model":"SM-S921B","memory":"128 GB"}',
                '{"brand":"Apple","model":"iPhone 15","memory":"256 GB"}',
            ],
            "category": ["electronics"] * 4,
        }
    )


def test_sparse_item_encoder_is_unseen_item_capable_and_pair_features_are_symmetric():
    items = _items()
    encoder = fit_sparse_item_encoder(items.iloc[:3], max_char_features=5000, max_word_features=2000)
    pairs = pd.DataFrame({"id1": [1, 1, 4], "id2": [2, 3, 1]})
    forward = transform_sparse_pairs(encoder, items, pairs)
    reverse = transform_sparse_pairs(
        encoder,
        items,
        pairs.rename(columns={"id1": "id2", "id2": "id1"})[["id1", "id2"]],
    )

    assert forward.shape == (3, 4)
    assert np.isfinite(forward.to_numpy()).all()
    assert np.allclose(forward.to_numpy(), reverse.to_numpy())


def test_sparse_similarity_recognizes_product_family_without_claiming_variant_identity():
    items = _items()
    encoder = fit_sparse_item_encoder(items.iloc[:3], max_char_features=5000, max_word_features=2000)
    pairs = pd.DataFrame({"id1": [1, 1, 1], "id2": [2, 3, 4]})
    features = transform_sparse_pairs(encoder, items, pairs)

    # Sparse lexical similarity is a family/semantic signal. It must not be
    # forced to override structured memory/SKU conflict features: translated
    # color words can make the true same variant lexically less similar than a
    # wrong-memory sibling. Both Samsung siblings should still rank above the
    # unrelated Apple product on the name representation.
    unrelated = features.loc[2, "name_char_tfidf_cosine"]
    assert features.loc[0, "name_char_tfidf_cosine"] > unrelated
    assert features.loc[1, "name_char_tfidf_cosine"] > unrelated
    assert features.loc[0, "full_word_tfidf_cosine"] > features.loc[2, "full_word_tfidf_cosine"]
