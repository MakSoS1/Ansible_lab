"""Single-backbone field-aware pair matcher for E-CUP v15."""

from __future__ import annotations

import torch
from torch import nn


class V15Matcher(nn.Module):
    """One Transformer backbone plus lightweight deterministic-feature heads.

    The class intentionally accepts a constructed backbone so unit tests can use
    a tiny fake module while production loads the pinned offline Transformer.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        typed_feature_dim: int,
        num_categories: int,
        use_typed_features: bool = False,
        use_category_head: bool = False,
        typed_projection_dim: int = 32,
        category_embedding_dim: int = 16,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.use_typed_features = bool(use_typed_features)
        self.use_category_head = bool(use_category_head)
        self.typed_feature_dim = int(typed_feature_dim)
        self.num_categories = int(num_categories)

        hidden_size = int(getattr(backbone.config, "hidden_size"))
        if hidden_size <= 0:
            raise ValueError("backbone hidden_size must be positive")
        if self.use_typed_features and self.typed_feature_dim <= 0:
            raise ValueError("typed_feature_dim must be positive when typed features are enabled")
        if self.use_category_head and self.num_categories <= 0:
            raise ValueError("num_categories must be positive when category head is enabled")

        fused_dim = hidden_size
        if self.use_typed_features:
            self.typed_projection = nn.Sequential(
                nn.LayerNorm(self.typed_feature_dim),
                nn.Linear(self.typed_feature_dim, typed_projection_dim),
                nn.GELU(),
            )
            fused_dim += typed_projection_dim
        else:
            self.typed_projection = None

        if self.use_category_head:
            self.category_embedding = nn.Embedding(self.num_categories, category_embedding_dim)
            fused_dim += category_embedding_dim
        else:
            self.category_embedding = None

        head_hidden = max(64, hidden_size // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    @staticmethod
    def _pool(outputs, attention_mask: torch.Tensor | None) -> torch.Tensor:
        pooler = getattr(outputs, "pooler_output", None)
        if pooler is not None:
            return pooler
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise ValueError("backbone output must expose pooler_output or last_hidden_state")
        # CLS is the stable historical CrossEncoder representation. Keeping this
        # simple also avoids a second learned sequence aggregation path.
        return hidden[:, 0]

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        typed_features: torch.Tensor | None = None,
        category_ids: torch.Tensor | None = None,
        **backbone_kwargs,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **backbone_kwargs,
        )
        pooled = self._pool(outputs, attention_mask)
        parts = [pooled]

        if self.use_typed_features:
            if typed_features is None:
                raise ValueError("typed_features are required when typed feature fusion is enabled")
            if typed_features.ndim != 2 or typed_features.shape[1] != self.typed_feature_dim:
                raise ValueError("typed_features shape does not match typed_feature_dim")
            parts.append(self.typed_projection(typed_features.to(dtype=pooled.dtype)))

        if self.use_category_head:
            if category_ids is None:
                raise ValueError("category_ids are required when category head is enabled")
            if category_ids.ndim != 1 or category_ids.shape[0] != pooled.shape[0]:
                raise ValueError("category_ids must contain one id per pair")
            if torch.any(category_ids < 0) or torch.any(category_ids >= self.num_categories):
                raise ValueError("category_ids outside configured category range")
            parts.append(self.category_embedding(category_ids))

        fused = torch.cat(parts, dim=-1)
        return self.head(fused).squeeze(-1)
