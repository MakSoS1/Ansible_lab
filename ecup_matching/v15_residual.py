"""Zero-safe category-conditioned residual head for v15."""
from __future__ import annotations

import torch
from torch import nn


class CategoryResidualHead(nn.Module):
    """Add a bounded category-specific correction to a frozen teacher logit.

    The expert projection and bias are exactly zero-initialized, therefore a new
    head returns the teacher logit exactly before optimization.  A shared feature
    trunk keeps the parameter count small; category-specific expert vectors let
    the same typed conflict mean different things in different product domains.
    """
    def __init__(self, *, feature_dim: int, num_categories: int, hidden_dim: int=64, dropout: float=0.05, residual_limit: float=3.0):
        super().__init__()
        if feature_dim<=0 or num_categories<=0 or hidden_dim<=0: raise ValueError("dimensions must be positive")
        self.feature_dim=int(feature_dim); self.num_categories=int(num_categories); self.hidden_dim=int(hidden_dim); self.residual_limit=float(residual_limit)
        self.trunk=nn.Sequential(nn.LayerNorm(self.feature_dim),nn.Linear(self.feature_dim,self.hidden_dim),nn.GELU(),nn.Dropout(float(dropout)))
        self.expert_weight=nn.Parameter(torch.zeros(self.num_categories,self.hidden_dim))
        self.expert_bias=nn.Parameter(torch.zeros(self.num_categories))

    def residual(self, features: torch.Tensor, category_ids: torch.Tensor) -> torch.Tensor:
        if features.ndim!=2 or features.shape[1]!=self.feature_dim: raise ValueError("bad feature shape")
        if category_ids.ndim!=1 or category_ids.shape[0]!=features.shape[0]: raise ValueError("bad category shape")
        if torch.any(category_ids<0) or torch.any(category_ids>=self.num_categories): raise ValueError("category id out of range")
        h=self.trunk(features)
        raw=(h*self.expert_weight[category_ids]).sum(-1)+self.expert_bias[category_ids]
        lim=self.residual_limit
        return lim*torch.tanh(raw/lim) if lim>0 else raw

    def forward(self, teacher_logit: torch.Tensor, features: torch.Tensor, category_ids: torch.Tensor) -> torch.Tensor:
        if teacher_logit.ndim!=1 or teacher_logit.shape[0]!=features.shape[0]: raise ValueError("teacher logit must be [B]")
        return teacher_logit+self.residual(features,category_ids)
