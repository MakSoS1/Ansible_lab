from __future__ import annotations

import numpy as np


def source_loss_weights(source, weak_confidence, *, human_weight: float=1.0, weak_scale: float=0.1)->np.ndarray:
    src=np.asarray(source);conf=np.asarray(weak_confidence,dtype=float)
    if src.shape!=conf.shape: raise ValueError('source and confidence shapes differ')
    if human_weight<=0 or weak_scale<0: raise ValueError('invalid weights')
    if not np.isfinite(conf).all() or (conf<0).any(): raise ValueError('invalid confidence')
    return np.where(src=='human',float(human_weight),float(weak_scale)*conf).astype(float)


def pairwise_category_ranking_loss_numpy(logits,target,category)->float:
    z=np.asarray(logits,dtype=float);y=np.asarray(target,dtype=float);cat=np.asarray(category)
    if not (z.shape==y.shape==cat.shape): raise ValueError('shape mismatch')
    losses=[]
    for c in np.unique(cat):
        mask=cat==c;pos=z[mask & (y>=0.5)];neg=z[mask & (y<0.5)]
        if len(pos)==0 or len(neg)==0: continue
        diff=pos[:,None]-neg[None,:]
        losses.append(np.logaddexp(0.0,-diff).mean())
    return float(np.mean(losses)) if losses else 0.0


def torch_category_ranking_loss(logits,target,category_ids):
    import torch
    losses=[]
    for c in torch.unique(category_ids):
        mask=category_ids==c;pos=logits[mask & (target>=0.5)];neg=logits[mask & (target<0.5)]
        if pos.numel()==0 or neg.numel()==0: continue
        losses.append(torch.nn.functional.softplus(-(pos[:,None]-neg[None,:])).mean())
    return torch.stack(losses).mean() if losses else logits.sum()*0.0
