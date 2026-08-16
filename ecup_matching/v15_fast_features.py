"""Fast deterministic typed pair features for the v15 residual ranker.

The implementation is deliberately independent of the legacy feature stack: one
JSON parse per item, then O(1) set/dict operations per candidate pair.  Training
and submission inference import this exact module.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-zа-я0-9]+", re.I)
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_MODEL = re.compile(r"(?=[a-zа-я0-9._/-]*[a-zа-я])(?=[a-zа-я0-9._/-]*\d)[a-zа-я0-9][a-zа-я0-9._/-]*", re.I)
_BRAND_KEYS = ("brand", "бренд", "марка", "manufacturer", "производитель")
_MODEL_KEYS = ("model", "модель", "sku", "артикул", "mpn", "part number", "part_number", "код модели")

FEATURE_NAMES = (
    "brand_equal", "brand_conflict", "model_exact", "model_overlap", "model_conflict",
    "numeric_overlap", "numeric_conflict", "title_jaccard", "title_containment",
    "title_length_ratio", "attr_key_jaccard", "attr_agree", "attr_conflict",
    "same_title", "both_attrs_ok", "model_count_gap", "numeric_count_gap",
)


def _norm(x: Any) -> str:
    if x is None: return ""
    s=unicodedata.normalize("NFKC",str(x)).lower().replace("ё","е").strip()
    return _WS.sub(" ",s)


def _flatten(obj: Any, prefix: str=""):
    if isinstance(obj,dict):
        for k in sorted(obj,key=lambda z:_norm(z)):
            nk=_norm(k); child=f"{prefix}.{nk}" if prefix else nk
            yield from _flatten(obj[k],child)
    elif isinstance(obj,list):
        vals=[_norm(v) for v in obj if not isinstance(v,(dict,list)) and _norm(v)]
        if vals and prefix: yield prefix," | ".join(vals)
    elif prefix:
        v=_norm(obj)
        if v: yield prefix,v


def _leaf(k: str) -> str:
    return k.rsplit(".",1)[-1]


def _jac(a:set[str],b:set[str])->float:
    u=a|b
    return float(len(a&b)/len(u)) if u else 0.0


@dataclass(frozen=True)
class FastItem:
    title: str
    category: str
    title_tokens: frozenset[str]
    brand: str
    models: frozenset[str]
    numbers: frozenset[str]
    attrs: tuple[tuple[str,str], ...]
    attrs_ok: bool


def normalize_item(name: Any, attributes: Any, category: Any) -> FastItem:
    title=_norm(name); cat=_norm(category); attrs=(); ok=False
    raw="" if attributes is None else str(attributes)
    if raw.strip():
        try:
            obj=json.loads(raw)
            if isinstance(obj,dict): attrs=tuple(sorted(_flatten(obj))); ok=True
        except (ValueError,TypeError,json.JSONDecodeError): pass
    amap=dict(attrs); brand=""
    models=set(_MODEL.findall(title.replace(" ","")))
    for k,v in attrs:
        leaf=_leaf(k)
        if not brand and any(x==leaf or x in leaf for x in _BRAND_KEYS): brand=v
        if any(x==leaf or x in leaf for x in _MODEL_KEYS):
            compact=re.sub(r"\s+","",v)
            if compact and any(c.isdigit() for c in compact) and any(c.isalpha() for c in compact): models.add(compact)
            models.update(_MODEL.findall(compact))
    alltext=title+" "+" ".join(f"{k} {v}" for k,v in attrs)
    nums={m.replace(",",".") for m in _NUM.findall(alltext)}
    return FastItem(title,cat,frozenset(_TOKEN.findall(title)),brand,frozenset(models),frozenset(nums),attrs,ok)


def build_item_cache(items: pd.DataFrame) -> dict[object,FastItem]:
    req={"id","name","attributes","category"}
    if not req.issubset(items.columns): raise ValueError(f"items missing {sorted(req-set(items.columns))}")
    out={}
    for item_id,name,attributes,category in items[["id","name","attributes","category"]].itertuples(index=False,name=None):
        if item_id in out: raise ValueError(f"duplicate item id {item_id!r}")
        out[item_id]=normalize_item(name,attributes,category)
    return out


def pair_vector(a: FastItem,b: FastItem) -> np.ndarray:
    ma,mb=set(a.models),set(b.models); na,nb=set(a.numbers),set(b.numbers)
    aa,ab=dict(a.attrs),dict(b.attrs); ka,kb=set(aa),set(ab); common=ka&kb
    bt=float(bool(a.brand and b.brand and a.brand==b.brand)); bc=float(bool(a.brand and b.brand and a.brand!=b.brand))
    me=float(bool(ma and mb and ma==mb)); mo=_jac(ma,mb); mc=float(bool(ma and mb and not ma&mb))
    no=float(len(na&nb)); nc=float(len(na^nb)); tj=_jac(set(a.title_tokens),set(b.title_tokens))
    den=max(1,min(len(a.title_tokens),len(b.title_tokens))); tc=float(len(set(a.title_tokens)&set(b.title_tokens))/den)
    tl=float(min(len(a.title),len(b.title))/max(len(a.title),len(b.title))) if max(len(a.title),len(b.title)) else 1.0
    ak=_jac(ka,kb); agree=float(sum(aa[k]==ab[k] for k in common)); conflict=float(sum(aa[k]!=ab[k] for k in common))
    return np.asarray([bt,bc,me,mo,mc,no,nc,tj,tc,tl,ak,agree,conflict,float(bool(a.title and a.title==b.title)),float(a.attrs_ok and b.attrs_ok),float(abs(len(ma)-len(mb))),float(abs(len(na)-len(nb)))],dtype=np.float32)


def build_pair_matrix(items: pd.DataFrame,pairs: pd.DataFrame,*,cache:dict|None=None) -> tuple[np.ndarray,np.ndarray,list[str]]:
    c=cache if cache is not None else build_item_cache(items)
    x=np.empty((len(pairs),len(FEATURE_NAMES)),dtype=np.float32); cats=[]
    for i,(id1,id2) in enumerate(pairs[["id1","id2"]].itertuples(index=False,name=None)):
        if id1 not in c or id2 not in c: raise KeyError(f"pair references missing item {id1!r}/{id2!r}")
        a,b=c[id1],c[id2]
        if a.category!=b.category: raise ValueError(f"cross-category pair {id1!r}/{id2!r}")
        x[i]=pair_vector(a,b); cats.append(a.category)
    return x,np.asarray(cats,dtype=object),list(FEATURE_NAMES)
