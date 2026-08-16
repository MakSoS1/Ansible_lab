"""Fast deterministic typed pair features for the v15 residual ranker.

The implementation is deliberately independent of the legacy feature stack: one
JSON parse per item, then O(1) set/dict operations per candidate pair. Training
and submission inference import this exact module.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable

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
    return _WS.sub(" ",unicodedata.normalize("NFKC",str(x)).lower().replace("ё","е").strip())

def _flatten(obj: Any,prefix: str=""):
    if isinstance(obj,dict):
        for k in sorted(obj,key=lambda z:_norm(z)):
            nk=_norm(k); yield from _flatten(obj[k],f"{prefix}.{nk}" if prefix else nk)
    elif isinstance(obj,list):
        vals=[_norm(v) for v in obj if not isinstance(v,(dict,list)) and _norm(v)]
        if vals and prefix: yield prefix," | ".join(vals)
    elif prefix:
        v=_norm(obj)
        if v: yield prefix,v

def _leaf(k:str)->str: return k.rsplit(".",1)[-1]
def _jac(a:set[str],b:set[str])->float:
    u=a|b; return float(len(a&b)/len(u)) if u else 0.0

@dataclass(frozen=True)
class FastItem:
    title:str; category:str; title_tokens:frozenset[str]; brand:str; models:frozenset[str]; numbers:frozenset[str]; attrs:tuple[tuple[str,str],...]; attrs_ok:bool

def normalize_item(name:Any,attributes:Any,category:Any)->FastItem:
    title=_norm(name); cat=_norm(category); attrs=(); ok=False; raw="" if attributes is None else str(attributes)
    if raw.strip():
        try:
            obj=json.loads(raw)
            if isinstance(obj,dict): attrs=tuple(sorted(_flatten(obj))); ok=True
        except (ValueError,TypeError,json.JSONDecodeError): pass
    brand=""; models=set(_MODEL.findall(title.replace(" ","")))
    for k,v in attrs:
        leaf=_leaf(k)
        if not brand and any(x==leaf or x in leaf for x in _BRAND_KEYS): brand=v
        if any(x==leaf or x in leaf for x in _MODEL_KEYS):
            compact=re.sub(r"\s+","",v)
            if compact and any(c.isdigit() for c in compact) and any(c.isalpha() for c in compact): models.add(compact)
            models.update(_MODEL.findall(compact))
    nums={m.replace(",",".") for m in _NUM.findall(title+" "+" ".join(f"{k} {v}" for k,v in attrs))}
    return FastItem(title,cat,frozenset(_TOKEN.findall(title)),brand,frozenset(models),frozenset(nums),attrs,ok)

def build_item_cache(items:pd.DataFrame)->dict[object,FastItem]:
    req={"id","name","attributes","category"}
    if not req.issubset(items.columns): raise ValueError(f"items missing {sorted(req-set(items.columns))}")
    out={}
    for item_id,name,attributes,category in items[["id","name","attributes","category"]].itertuples(index=False,name=None):
        if item_id in out: raise ValueError(f"duplicate item id {item_id!r}")
        out[item_id]=normalize_item(name,attributes,category)
    return out

def build_item_cache_from_parquet(path:Path,item_ids:Iterable[object],*,batch_size:int=131072)->dict[object,FastItem]:
    import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
    wanted=set(item_ids)
    if not wanted:return {}
    pf=pq.ParquetFile(str(path)); cols=['id','name','attributes','category']; missing=set(cols)-set(pf.schema_arrow.names)
    if missing: raise ValueError(f"items parquet missing {sorted(missing)}")
    values=pa.array(list(wanted),type=pf.schema_arrow.field('id').type); out={}
    for batch in pf.iter_batches(batch_size=batch_size,columns=cols):
        ids=batch.column(batch.schema.get_field_index('id')); selected=batch.filter(pc.is_in(ids,value_set=values))
        if selected.num_rows:
            d=selected.to_pydict()
            for item_id,name,attributes,category in zip(d['id'],d['name'],d['attributes'],d['category']):
                if item_id not in out: out[item_id]=normalize_item(name,attributes,category)
        if len(out)==len(wanted):break
    missing_ids=wanted-set(out)
    if missing_ids: raise KeyError(f"items parquet missing {len(missing_ids)} requested ids")
    return out

def pair_vector(a:FastItem,b:FastItem)->np.ndarray:
    ma,mb=set(a.models),set(b.models); na,nb=set(a.numbers),set(b.numbers); aa,ab=dict(a.attrs),dict(b.attrs); ka,kb=set(aa),set(ab); common=ka&kb
    den=max(1,min(len(a.title_tokens),len(b.title_tokens)))
    return np.asarray([float(bool(a.brand and b.brand and a.brand==b.brand)),float(bool(a.brand and b.brand and a.brand!=b.brand)),float(bool(ma and mb and ma==mb)),_jac(ma,mb),float(bool(ma and mb and not ma&mb)),float(len(na&nb)),float(len(na^nb)),_jac(set(a.title_tokens),set(b.title_tokens)),float(len(set(a.title_tokens)&set(b.title_tokens))/den),float(min(len(a.title),len(b.title))/max(len(a.title),len(b.title))) if max(len(a.title),len(b.title)) else 1.,_jac(ka,kb),float(sum(aa[k]==ab[k] for k in common)),float(sum(aa[k]!=ab[k] for k in common)),float(bool(a.title and a.title==b.title)),float(a.attrs_ok and b.attrs_ok),float(abs(len(ma)-len(mb))),float(abs(len(na)-len(nb)))],dtype=np.float32)

def build_pair_matrix(items:pd.DataFrame|None,pairs:pd.DataFrame,*,cache:dict|None=None)->tuple[np.ndarray,np.ndarray,list[str]]:
    c=cache if cache is not None else build_item_cache(items)
    x=np.empty((len(pairs),len(FEATURE_NAMES)),dtype=np.float32); cats=[]
    for i,(id1,id2) in enumerate(pairs[["id1","id2"]].itertuples(index=False,name=None)):
        if id1 not in c or id2 not in c: raise KeyError(f"pair references missing item {id1!r}/{id2!r}")
        a,b=c[id1],c[id2]
        if a.category!=b.category: raise ValueError(f"cross-category pair {id1!r}/{id2!r}")
        x[i]=pair_vector(a,b); cats.append(a.category)
    return x,np.asarray(cats,dtype=object),list(FEATURE_NAMES)
