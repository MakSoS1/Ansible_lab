from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import numpy as np
import pandas as pd

from .category_attrs import _leaf_key
from .features import normalize_items
from .textnorm import ItemNorm


def _leaf_values(item: ItemNorm) -> dict[str, frozenset[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for key, value in item.attrs.items():
        if value:
            values[_leaf_key(key)].add(str(value))
    return {key: frozenset(v) for key, v in values.items()}


def learn_explicit_attribute_keys(
    items: pd.DataFrame,
    train_pairs: pd.DataFrame,
    *,
    max_keys_per_category: int = 40,
    min_support: int = 30,
    item_cache: Mapping[object, ItemNorm] | None = None,
) -> dict[str, list[str]]:
    if max_keys_per_category <= 0 or min_support <= 0:
        raise ValueError("key limits must be positive")
    required = {"id1", "id2", "target", "category"}
    missing = required - set(train_pairs.columns)
    if missing:
        raise ValueError(f"train_pairs missing columns: {sorted(missing)}")
    cache = dict(item_cache) if item_cache is not None else normalize_items(items)
    leaf_cache = {item_id: _leaf_values(item) for item_id, item in cache.items()}
    stats = defaultdict(lambda: defaultdict(lambda: {"support":0,"pos_eq":0,"pos_n":0,"neg_eq":0,"neg_n":0}))
    for id1,id2,target,category in train_pairs[["id1","id2","target","category"]].itertuples(index=False,name=None):
        if id1 not in leaf_cache or id2 not in leaf_cache:
            continue
        a,b=leaf_cache[id1],leaf_cache[id2]
        for key in set(a)&set(b):
            eq=bool(a[key]&b[key])
            entry=stats[str(category)][key];entry["support"]+=1
            if float(target)>=0.5:
                entry["pos_n"]+=1;entry["pos_eq"]+=int(eq)
            else:
                entry["neg_n"]+=1;entry["neg_eq"]+=int(eq)
    result={}
    for category,by_key in stats.items():
        ranked=[]
        for key,s in by_key.items():
            if s["support"]<min_support: continue
            pos=(s["pos_eq"]+1.0)/(s["pos_n"]+2.0)
            neg=(s["neg_eq"]+1.0)/(s["neg_n"]+2.0)
            discrimination=abs(pos-neg)
            score=discrimination*np.log1p(float(s["support"]))
            ranked.append((score,s["support"],key))
        ranked.sort(key=lambda x:(-x[0],-x[1],x[2]))
        if ranked: result[category]=[key for _,_,key in ranked[:max_keys_per_category]]
    return result


def build_explicit_attribute_features(
    items: pd.DataFrame,
    pairs: pd.DataFrame,
    key_spec: Mapping[str, list[str]],
    *,
    item_cache: Mapping[object, ItemNorm] | None = None,
    category: str | None = None,
) -> pd.DataFrame:
    if not {"id1","id2","category"}.issubset(pairs.columns):
        raise ValueError("pairs must contain id1,id2,category")
    cache=dict(item_cache) if item_cache is not None else normalize_items(items)
    leaf_cache={item_id:_leaf_values(item) for item_id,item in cache.items()}
    if category is None:
        categories=sorted(set(pairs["category"].astype(str).tolist()))
        keys=sorted({key for cat in categories for key in key_spec.get(cat,[])})
    else:
        keys=list(key_spec.get(str(category),[]))
    columns=[]
    for key in keys:
        columns.extend([f"attr_eq::{key}",f"attr_conflict::{key}",f"attr_missing::{key}"])
    rows=[]
    for id1,id2,cat in pairs[["id1","id2","category"]].itertuples(index=False,name=None):
        if id1 not in leaf_cache or id2 not in leaf_cache:
            raise KeyError("pair references missing item")
        a,b=leaf_cache[id1],leaf_cache[id2];row={}
        active=set(key_spec.get(str(cat),[]))
        for key in keys:
            va,vb=a.get(key),b.get(key)
            if key not in active:
                eq=conflict=missing=0.0
            elif va is None or vb is None:
                eq=conflict=0.0;missing=1.0
            else:
                eq=float(bool(va&vb));conflict=float(not bool(va&vb));missing=0.0
            row[f"attr_eq::{key}"]=eq;row[f"attr_conflict::{key}"]=conflict;row[f"attr_missing::{key}"]=missing
        rows.append(row)
    return pd.DataFrame(rows,columns=columns,dtype=np.float32)
