from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import os

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from .validation import make_splits


@dataclass
class CVResult:
    name: str
    oof_probs: np.ndarray
    test_probs: np.ndarray
    fold_scores: list[float]
    oof_accuracy: float
    metadata: dict = field(default_factory=dict)


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-12, None)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def _cv_sklearn_model(
    name: str,
    model_factory: Callable[[], object],
    X: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    n_splits: int,
    seed: int,
) -> CVResult:
    X_arr = np.asarray(X, dtype=np.float64)
    Xt_arr = np.asarray(X_test, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    oof = np.zeros((len(y), 3), dtype=np.float64)
    test_accum = np.zeros((len(Xt_arr), 3), dtype=np.float64)
    fold_scores: list[float] = []
    splits = list(make_splits(y, groups=groups, n_splits=n_splits, seed=seed))

    for tr_idx, va_idx in splits:
        model = model_factory()
        model.fit(X_arr[tr_idx], y[tr_idx])
        va_probs = _normalize_probs(model.predict_proba(X_arr[va_idx]))
        te_probs = _normalize_probs(model.predict_proba(Xt_arr))
        oof[va_idx] = va_probs
        test_accum += te_probs / len(splits)
        fold_scores.append(float(accuracy_score(y[va_idx], va_probs.argmax(axis=1))))

    return CVResult(
        name=name,
        oof_probs=oof,
        test_probs=_normalize_probs(test_accum),
        fold_scores=fold_scores,
        oof_accuracy=float(accuracy_score(y, oof.argmax(axis=1))),
    )


def _choose_best(results: list[CVResult], y: np.ndarray, allow_blend: bool = True) -> CVResult:
    y = np.asarray(y, dtype=int)
    best = max(results, key=lambda r: (r.oof_accuracy, r.name))
    best_meta = dict(best.metadata)
    best_meta['candidate_scores'] = {r.name: r.oof_accuracy for r in results}

    if allow_blend and len(results) >= 2:
        ranked = sorted(results, key=lambda r: r.oof_accuracy, reverse=True)[:3]
        blend_best = best
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                a, b = ranked[i], ranked[j]
                for alpha in np.arange(0.2, 0.81, 0.1):
                    oof = _normalize_probs(alpha * a.oof_probs + (1 - alpha) * b.oof_probs)
                    acc = float(accuracy_score(y, oof.argmax(axis=1)))
                    if acc > blend_best.oof_accuracy:
                        test = _normalize_probs(alpha * a.test_probs + (1 - alpha) * b.test_probs)
                        blend_best = CVResult(
                            name=f'blend({a.name},{b.name},alpha={alpha:.1f})',
                            oof_probs=oof,
                            test_probs=test,
                            fold_scores=[],
                            oof_accuracy=acc,
                            metadata={'blend_components': [a.name, b.name], 'alpha': float(alpha)},
                        )
        best = blend_best
    best.metadata = {**best_meta, **best.metadata}
    return best


def run_solution_1(
    X: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> CVResult:
    candidates = [
        _cv_sklearn_model(
            's1_logreg',
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=3000, random_state=seed),
            ),
            X, X_test, y, groups, n_splits, seed,
        ),
        _cv_sklearn_model(
            's1_lda',
            lambda: make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
            ),
            X, X_test, y, groups, n_splits, seed,
        ),
        _cv_sklearn_model(
            's1_linear_svm',
            lambda: make_pipeline(
                StandardScaler(),
                SVC(C=1.0, kernel='linear', probability=True, random_state=seed),
            ),
            X, X_test, y, groups, n_splits, seed,
        ),
    ]
    result = _choose_best(candidates, y, allow_blend=True)
    result.name = 'solution_1_global_' + result.name
    return result


def run_solution_2(
    X: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42,
    fast: bool = False,
) -> CVResult:
    n_estimators = 180 if fast else 650
    cat_iterations = 180 if fast else 700
    candidates = [
        _cv_sklearn_model(
            's2_catboost',
            lambda: CatBoostClassifier(
                iterations=cat_iterations,
                depth=5,
                learning_rate=0.04,
                l2_leaf_reg=7.0,
                random_strength=0.4,
                loss_function='MultiClass',
                eval_metric='Accuracy',
                verbose=False,
                allow_writing_files=False,
                random_seed=seed,
                thread_count=-1,
            ),
            X, X_test, y, groups, n_splits, seed,
        ),
        _cv_sklearn_model(
            's2_lightgbm',
            lambda: LGBMClassifier(
                objective='multiclass',
                num_class=3,
                n_estimators=n_estimators,
                learning_rate=0.035,
                num_leaves=15,
                max_depth=5,
                min_child_samples=25,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=5.0,
                random_state=seed,
                verbosity=-1,
                n_jobs=-1,
            ),
            X, X_test, y, groups, n_splits, seed,
        ),
        _cv_sklearn_model(
            's2_extratrees',
            lambda: ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                min_samples_leaf=2,
                max_features=0.8,
                class_weight='balanced',
                random_state=seed,
                n_jobs=-1,
            ),
            X, X_test, y, groups, n_splits, seed,
        ),
    ]
    result = _choose_best(candidates, y, allow_blend=True)
    result.name = 'solution_2_physics_' + result.name
    return result


def _cv_ordinal_model(
    X: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    n_splits: int,
    seed: int,
) -> CVResult:
    X_arr = np.asarray(X, dtype=np.float64)
    Xt_arr = np.asarray(X_test, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    oof = np.zeros((len(y), 3), dtype=np.float64)
    test_accum = np.zeros((len(Xt_arr), 3), dtype=np.float64)
    scores = []
    splits = list(make_splits(y, groups=groups, n_splits=n_splits, seed=seed))
    for tr, va in splits:
        m_gt0 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2500, random_state=seed))
        m_gt1 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2500, random_state=seed + 1))
        m_gt0.fit(X_arr[tr], (y[tr] > 0).astype(int))
        m_gt1.fit(X_arr[tr], (y[tr] > 1).astype(int))

        def probs(arr):
            p_gt0 = m_gt0.predict_proba(arr)[:, 1]
            p_gt1 = m_gt1.predict_proba(arr)[:, 1]
            p_gt1 = np.minimum(p_gt1, p_gt0)
            return _normalize_probs(np.column_stack([1 - p_gt0, p_gt0 - p_gt1, p_gt1]))

        pv = probs(X_arr[va])
        pt = probs(Xt_arr)
        oof[va] = pv
        test_accum += pt / len(splits)
        scores.append(float(accuracy_score(y[va], pv.argmax(axis=1))))
    return CVResult(
        name='s3_ordinal_logreg',
        oof_probs=oof,
        test_probs=_normalize_probs(test_accum),
        fold_scores=scores,
        oof_accuracy=float(accuracy_score(y, oof.argmax(axis=1))),
    )


def run_solution_3(
    X: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42,
    fast: bool = False,
) -> CVResult:
    trees = 180 if fast else 650
    hgb_iter = 120 if fast else 300
    candidates = [
        _cv_sklearn_model(
            's3_extratrees',
            lambda: ExtraTreesClassifier(
                n_estimators=trees,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=seed + 13,
                n_jobs=-1,
            ),
            X, X_test, y, groups, n_splits, seed,
        ),
        _cv_sklearn_model(
            's3_histgb',
            lambda: HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=hgb_iter,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=2.0,
                random_state=seed,
            ),
            X, X_test, y, groups, n_splits, seed,
        ),
        _cv_sklearn_model(
            's3_rbf_svm',
            lambda: make_pipeline(
                StandardScaler(),
                SVC(C=3.0, gamma='scale', kernel='rbf', probability=True, random_state=seed),
            ),
            X, X_test, y, groups, n_splits, seed,
        ),
        _cv_ordinal_model(X, X_test, y, groups, n_splits, seed),
    ]
    result = _choose_best(candidates, y, allow_blend=True)
    result.name = 'solution_3_spatial_' + result.name
    return result


# -----------------------------------------------------------------------------
# Vision solutions
# -----------------------------------------------------------------------------

def _vision_device():
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _build_embedding_backbone(backbone: str, device):
    """Return a frozen feature extractor and its deterministic preprocessing."""
    try:
        import timm
        model = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool='avg')
        config = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**config, is_training=False)
        model.to(device).eval()
        return model, preprocess
    except Exception as timm_error:
        from torchvision import models
        mapping = {
            'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 'classifier'),
            'mobilenetv3_small_100': (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT, 'classifier'),
            'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT, 'fc'),
        }
        key = backbone if backbone in mapping else 'efficientnet_b0'
        ctor, weights, head = mapping[key]
        try:
            model = ctor(weights=weights)
        except Exception:
            if key != 'resnet18':
                ctor, weights, head = mapping['resnet18']
                model = ctor(weights=weights)
            else:
                raise timm_error
        import torch.nn as nn
        setattr(model, head, nn.Identity())
        preprocess = weights.transforms()
        model.to(device).eval()
        return model, preprocess


def extract_pretrained_embeddings(
    paths,
    backbone: str = 'efficientnet_b0',
    batch_size: int = 32,
    cache_path: Path | str | None = None,
):
    import torch
    from PIL import Image

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return np.load(cache_path)

    device = _vision_device()
    model, preprocess = _build_embedding_backbone(backbone, device)
    rows = []
    batch = []
    with torch.no_grad():
        for i, path in enumerate(paths):
            with Image.open(path) as image:
                batch.append(preprocess(image.convert('RGB')))
            if len(batch) == batch_size or i == len(paths) - 1:
                tensor = torch.stack(batch).to(device)
                out = model(tensor)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if out.ndim > 2:
                    out = out.flatten(2).mean(-1)
                rows.append(out.detach().cpu().numpy())
                batch = []
    embeddings = np.concatenate(rows, axis=0).astype(np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
    return embeddings


def run_solution_4(
    train_paths,
    test_paths,
    y: np.ndarray,
    physics_train: pd.DataFrame | np.ndarray | None = None,
    physics_test: pd.DataFrame | np.ndarray | None = None,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42,
    cache_dir: Path | str | None = None,
    backbone: str = 'efficientnet_b0',
) -> CVResult:
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    train_cache = cache_dir / f'train_{backbone}.npy' if cache_dir else None
    test_cache = cache_dir / f'test_{backbone}.npy' if cache_dir else None
    E = extract_pretrained_embeddings(train_paths, backbone=backbone, cache_path=train_cache)
    Et = extract_pretrained_embeddings(test_paths, backbone=backbone, cache_path=test_cache)
    matrices = [('emb', E, Et)]
    if physics_train is not None and physics_test is not None:
        matrices.append((
            'emb_phys',
            np.hstack([E, np.asarray(physics_train, dtype=np.float32)]),
            np.hstack([Et, np.asarray(physics_test, dtype=np.float32)]),
        ))

    results = []
    for suffix, X, Xt in matrices:
        results.extend([
            _cv_sklearn_model(
                f's4_{suffix}_logreg',
                lambda: make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=0.25, max_iter=3000, random_state=seed),
                ),
                X, Xt, y, groups, n_splits, seed,
            ),
            _cv_sklearn_model(
                f's4_{suffix}_linear_svm',
                lambda: make_pipeline(
                    StandardScaler(),
                    SVC(C=0.5, kernel='linear', probability=True, random_state=seed),
                ),
                X, Xt, y, groups, n_splits, seed,
            ),
            _cv_sklearn_model(
                f's4_{suffix}_lgbm',
                lambda: LGBMClassifier(
                    objective='multiclass', num_class=3,
                    n_estimators=450, learning_rate=0.03,
                    num_leaves=15, max_depth=5, min_child_samples=30,
                    colsample_bytree=0.35, reg_alpha=1.0, reg_lambda=6.0,
                    random_state=seed, verbosity=-1, n_jobs=-1,
                ),
                X, Xt, y, groups, n_splits, seed,
            ),
        ])
    result = _choose_best(results, y, allow_blend=True)
    result.name = 'solution_4_embeddings_' + result.name
    result.metadata['backbone'] = backbone
    return result


def build_eval_transform(image_size: int = 224):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_train_transform(image_size: int = 224):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.86, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(6),
        transforms.ColorJitter(brightness=0.06, contrast=0.10, saturation=0.05, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_cnn_model(num_classes: int = 3, pretrained: bool = True, backbone: str = 'efficientnet_b0'):
    import torch.nn as nn
    from torchvision import models

    if backbone == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if backbone == 'mobilenet_v3_small':
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _cnn_head_and_backbone_params(model, backbone: str):
    if backbone == 'resnet18':
        head = list(model.fc.parameters())
        head_ids = {id(p) for p in head}
    else:
        head = list(model.classifier.parameters())
        head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    return body, head


def run_solution_5(
    train_paths,
    test_paths,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 3,
    seed: int = 42,
    backbone: str = 'efficientnet_b0',
    image_size: int = 224,
    epochs: int | None = None,
    batch_size: int = 16,
) -> CVResult:
    import random
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    if epochs is None:
        epochs = int(os.getenv('AIJC_CNN_EPOCHS', '8'))
    n_splits = int(os.getenv('AIJC_CNN_FOLDS', str(n_splits)))
    device = _vision_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    class ImgDataset(Dataset):
        def __init__(self, paths, labels=None, transform=None):
            self.paths = list(paths)
            self.labels = None if labels is None else np.asarray(labels, dtype=int)
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            with Image.open(self.paths[idx]) as im:
                image = im.convert('RGB')
                x = self.transform(image) if self.transform else image
            if self.labels is None:
                return x
            return x, int(self.labels[idx])

    train_tf = build_train_transform(image_size)
    eval_tf = build_eval_transform(image_size)
    flip_tf = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    y = np.asarray(y, dtype=int)
    oof = np.zeros((len(y), 3), dtype=np.float64)
    test_accum = np.zeros((len(test_paths), 3), dtype=np.float64)
    fold_scores = []
    splits = list(make_splits(y, groups=groups, n_splits=n_splits, seed=seed))

    def infer(model, paths, tf, labels=None):
        ds = ImgDataset(paths, labels, tf)
        dl = DataLoader(ds, batch_size=max(batch_size, 24), shuffle=False, num_workers=2)
        probs = []
        model.eval()
        with torch.no_grad():
            for batch in dl:
                x = batch[0] if labels is not None else batch
                logits = model(x.to(device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs)

    for fold, (tr, va) in enumerate(splits):
        model = build_cnn_model(3, pretrained=True, backbone=backbone).to(device)
        body, head = _cnn_head_and_backbone_params(model, backbone)
        optimizer = torch.optim.AdamW([
            {'params': body, 'lr': 1e-5},
            {'params': head, 'lr': 3e-4},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=1e-7)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        tr_paths = [train_paths[i] for i in tr]
        va_paths = [train_paths[i] for i in va]
        tr_ds = ImgDataset(tr_paths, y[tr], train_tf)
        va_ds = ImgDataset(va_paths, y[va], eval_tf)
        tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        va_dl = DataLoader(va_ds, batch_size=max(batch_size, 24), shuffle=False, num_workers=2)

        best_acc = -1.0
        best_state = None
        patience = 3
        no_improve = 0
        for _epoch in range(epochs):
            model.train()
            for x, target in tr_dl:
                x = x.to(device)
                target = target.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, target in va_dl:
                    logits = model(x.to(device))
                    pred = logits.argmax(1).cpu()
                    correct += int((pred == target).sum())
                    total += len(target)
            acc = correct / max(total, 1)
            if acc > best_acc + 1e-9:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        val_probs = infer(model, va_paths, eval_tf)
        test_a = infer(model, test_paths, eval_tf)
        test_b = infer(model, test_paths, flip_tf)
        test_probs = _normalize_probs((test_a + test_b) / 2.0)
        oof[va] = val_probs
        test_accum += test_probs / len(splits)
        fold_scores.append(float(accuracy_score(y[va], val_probs.argmax(1))))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return CVResult(
        name=f'solution_5_cnn_{backbone}',
        oof_probs=_normalize_probs(oof),
        test_probs=_normalize_probs(test_accum),
        fold_scores=fold_scores,
        oof_accuracy=float(accuracy_score(y, oof.argmax(1))),
        metadata={
            'backbone': backbone,
            'epochs': epochs,
            'n_splits': n_splits,
            'device': str(device),
        },
    )
