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
    assert ((forward.to_numpy() >= -1e-7) & (forward.to_numpy() <= 1.0 + 1e-7)).all()


def test_sparse_similarity_has_identity_sanity_without_encoding_variant_policy():
    items = _items()
    encoder = fit_sparse_item_encoder(items.iloc[:3], max_char_features=5000, max_word_features=2000)
    pairs = pd.DataFrame({"id1": [1, 1, 1], "id2": [1, 2, 4]})
    features = transform_sparse_pairs(encoder, items, pairs)

    # TF-IDF is only a lexical representation. Unknown vocabulary and translated
    # colors mean it must not be unit-tested as an identity oracle; structured
    # SKU/numeric conflicts and the learned residual decide that. The invariant
    # here is simply that an item is maximally similar to itself and non-identical
    # descriptions are not more similar than self-similarity.
    assert np.allclose(features.loc[0].to_numpy(), np.ones(4), atol=1e-6)
    assert (features.loc[1:].to_numpy() <= 1.0 + 1e-7).all()
    assert (features.loc[1:].to_numpy() < 1.0 - 1e-6).any()
