from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score


OFFICIAL_CATEGORIES: tuple[str, ...] = (
    "Автотовары",
    "Аптека",
    "Бытовая техника",
    "Бытовая химия",
    "Галантерея и аксессуары",
    "Детские товары",
    "Дом и сад",
    "Канцелярские товары",
    "Красота и гигиена",
    "Мебель",
    "Музыкальные инструменты",
    "Обувь",
    "Одежда",
    "Продукты питания",
    "Спорт и отдых",
    "Строительство и ремонт",
    "Товары для животных",
    "Хобби и творчество",
    "Электроника",
    "Ювелирные изделия",
)


def macro_average_precision(
    y_true,
    y_score,
    categories,
    *,
    expected_categories: Sequence[str] | None = None,
    require_both_classes: bool = False,
) -> tuple[float, dict[str, float]]:
    """Compute the competition metric: mean sklearn average precision over categories.

    The default remains generic for unit tests and reusable diagnostics. Callers
    that evaluate the official full development/test distribution can provide
    ``expected_categories`` and ``require_both_classes=True`` to fail loudly on
    incomplete inputs instead of silently averaging a different problem.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    categories = np.asarray(categories).astype(str)

    if not (len(y_true) == len(y_score) == len(categories)):
        raise ValueError("y_true, y_score and categories must have equal lengths")
    if len(y_true) == 0:
        raise ValueError("metric input must not be empty")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score contains NaN or infinity")

    present_categories = sorted(np.unique(categories).tolist())
    if expected_categories is not None:
        expected = {str(category) for category in expected_categories}
        present = set(present_categories)
        if present != expected:
            missing = sorted(expected - present)
            unexpected = sorted(present - expected)
            raise ValueError(
                f"metric category set mismatch: missing={missing}, unexpected={unexpected}"
            )

    per_category: dict[str, float] = {}
    for category in present_categories:
        mask = categories == category
        category_targets = np.unique(y_true[mask])
        if require_both_classes and set(category_targets.tolist()) != {0, 1}:
            raise ValueError(
                f"category {category!r} must contain both target classes 0 and 1; "
                f"observed={category_targets.tolist()}"
            )
        per_category[category] = float(average_precision_score(y_true[mask], y_score[mask]))

    return float(np.mean(list(per_category.values()))), per_category
