from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

import src.solutions as solutions


def _save(path: Path, value: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (24, 24), (value, value, value)).save(path)


class TinyEmbed(nn.Module):
    def forward(self, x):
        return torch.stack([
            x.mean(dim=(1, 2, 3)),
            x.std(dim=(1, 2, 3)),
        ], dim=1)


def tiny_preprocess(image):
    arr = np.asarray(image.resize((8, 8)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def test_embedding_extraction_preserves_order_and_cache(tmp_path: Path, monkeypatch):
    paths = []
    for i, value in enumerate([20, 80, 140, 220]):
        p = tmp_path / f'{i}.png'
        _save(p, value)
        paths.append(p)
    monkeypatch.setattr(
        solutions,
        '_build_embedding_backbone',
        lambda backbone, device: (TinyEmbed().to(device), tiny_preprocess),
    )
    cache = tmp_path / 'embeddings.npy'
    emb1 = solutions.extract_pretrained_embeddings(
        paths, backbone='tiny', batch_size=2, cache_path=cache)
    assert emb1.shape == (4, 2)
    assert np.all(np.diff(emb1[:, 0]) > 0)
    monkeypatch.setattr(
        solutions,
        '_build_embedding_backbone',
        lambda backbone, device: (_ for _ in ()).throw(RuntimeError('cache not used')),
    )
    emb2 = solutions.extract_pretrained_embeddings(
        paths, backbone='tiny', batch_size=2, cache_path=cache)
    assert np.allclose(emb1, emb2)


def test_cnn_helpers_keep_three_outputs_and_eval_is_deterministic():
    model = solutions.build_cnn_model(num_classes=3, pretrained=False, backbone='resnet18')
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 3)

    tf = solutions.build_eval_transform(224)
    image = Image.new('RGB', (256, 256), (128, 128, 128))
    a = tf(image)
    b = tf(image)
    assert torch.equal(a, b)
