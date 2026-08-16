from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _logit(p: np.ndarray) -> np.ndarray:
    p=np.clip(np.asarray(p,dtype=np.float64),1e-6,1-1e-6)
    return (np.log(p)-np.log1p(-p)).astype(np.float32)


def predict_to_csv_v15_residual(*,items_path:Path,matches_path:Path,teacher_dir:Path,residual_path:Path,output_path:Path,max_length:int=256,max_chars:int=900,batch_size:int=64)->dict:
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    from ecup_matching.ml.v7_runtime import build_v7_text_cache_from_parquet,predict_pairs
    from ecup_matching.v15_fast_features import FEATURE_NAMES,build_item_cache_from_parquet,build_pair_matrix
    from ecup_matching.v15_residual import CategoryResidualHead
    if not torch.cuda.is_available(): raise RuntimeError('v15 residual submission requires CUDA')
    matches=pd.read_parquet(matches_path,columns=['id1','id2']).reset_index(drop=True)
    ids=set(matches.id1)|set(matches.id2)
    texts,categories_by_id=build_v7_text_cache_from_parquet(items_path,ids,max_chars=max_chars)
    pair_categories=[]
    for a,b in matches[['id1','id2']].itertuples(index=False,name=None):
        ca=str(categories_by_id[a]); cb=str(categories_by_id[b])
        if ca!=cb: raise RuntimeError(f'cross-category candidate pair {a!r}/{b!r}')
        pair_categories.append(ca)
    tokenizer=AutoTokenizer.from_pretrained(teacher_dir,local_files_only=True)
    teacher=AutoModelForSequenceClassification.from_pretrained(teacher_dir,local_files_only=True).cuda().eval()
    teacher_prob,teacher_inf=predict_pairs(model=teacher,tokenizer=tokenizer,frame=matches,texts=texts,device='cuda',max_length=max_length,batch_size=batch_size)
    teacher_logit=_logit(teacher_prob); del teacher,tokenizer,texts; torch.cuda.empty_cache()
    try: ck=torch.load(residual_path,map_location='cpu',weights_only=False)
    except TypeError: ck=torch.load(residual_path,map_location='cpu')
    if ck.get('version')!='v15-r1-fast-category-residual-v1': raise RuntimeError('wrong v15 residual checkpoint version')
    if ck.get('gold_metric_opened') is not False: raise RuntimeError('invalid sealed-gold provenance')
    names=list(ck['feature_names']); expected=list(FEATURE_NAMES)+['teacher_tanh','teacher_abs_tanh']
    if names!=expected: raise RuntimeError('fast feature contract mismatch')
    fast_cache=build_item_cache_from_parquet(items_path,ids)
    x,cats,_=build_pair_matrix(None,matches,cache=fast_cache)
    if [str(x) for x in cats.tolist()]!=[str(x).lower().replace('ё','е') for x in pair_categories]:
        # v7 categories preserve source casing, fast features normalize; compare normalized below.
        import unicodedata,re
        norm=lambda s:re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(s)).lower().replace('ё','е').strip())
        if [norm(x) for x in cats.tolist()]!=[norm(x) for x in pair_categories]: raise RuntimeError('category normalization mismatch')
    x=np.concatenate([x,np.tanh(teacher_logit[:,None]/4),np.abs(np.tanh(teacher_logit[:,None]/4))],1).astype(np.float32)
    mean=np.asarray(ck['mean'],np.float32); std=np.asarray(ck['std'],np.float32)
    if x.shape[1]!=len(mean) or len(mean)!=len(std): raise RuntimeError('residual normalization shape mismatch')
    x=np.nan_to_num((x-mean)/std).astype(np.float32)
    categories=list(map(str,ck['categories'])); cidx={c:i for i,c in enumerate(categories)}
    cat_ids=np.asarray([cidx[str(c)] if str(c) in cidx else cidx.get(str(c).lower(),-1) for c in cats],np.int64)
    if (cat_ids<0).any(): raise RuntimeError('unknown category for residual')
    head=CategoryResidualHead(feature_dim=x.shape[1],num_categories=len(categories),hidden_dim=64,dropout=.05,residual_limit=float(ck.get('residual_limit',3.0))).cuda().eval(); head.load_state_dict(ck['residual_state'],strict=True)
    out=[]
    with torch.inference_mode():
        for s in range(0,len(matches),32768):
            t=torch.from_numpy(teacher_logit[s:s+32768]).cuda(); xx=torch.from_numpy(x[s:s+32768]).cuda(); cc=torch.from_numpy(cat_ids[s:s+32768]).cuda(); out.append(torch.sigmoid(head(t,xx,cc)).float().cpu().numpy())
    pred=np.concatenate(out).astype(np.float64) if out else np.empty(0,np.float64)
    if len(pred)!=len(matches) or not np.isfinite(pred).all(): raise RuntimeError('invalid v15 predictions')
    result=matches.copy(); result['predict']=pred; output_path.parent.mkdir(parents=True,exist_ok=True); result.to_csv(output_path,index=False)
    return {'rows':len(result),'teacher_inference':teacher_inf,'predict_min':float(pred.min()) if len(pred) else None,'predict_max':float(pred.max()) if len(pred) else None}
