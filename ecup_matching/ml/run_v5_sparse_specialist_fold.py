from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .features_v2 import build_features_v2_chunked
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_category_specialists import fit_predict_category_specialists
from .v5_evaluation import macro_ap_report
from .v5_sparse_crossfit import fit_transform_sparse_outer_fold
from .v5_validation import manifest_sha256


def run_sparse_specialist_fold(*,items_path:Path,matches_path:Path,manifest_path:Path,output_dir:Path,expected_split_sha:str,held_fold:int,max_iter:int=300)->dict:
    started=time.perf_counter();output_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest_sha256(manifest)!=expected_split_sha: raise ValueError('sealed split SHA mismatch')
    matches=pd.read_parquet(matches_path,columns=['id1','id2','target'])
    dev_rows,fold_ids=development_rows_and_folds(manifest,total_rows=len(matches))
    if held_fold not in set(fold_ids.tolist()): raise ValueError('held fold absent')
    dev=matches.iloc[dev_rows].reset_index(drop=True);train_mask=fold_ids!=held_fold;valid_mask=fold_ids==held_fold
    train=dev.loc[train_mask].reset_index(drop=True);valid=dev.loc[valid_mask].reset_index(drop=True);held_rows=dev_rows[valid_mask]
    ids=pd.unique(pd.concat([dev['id1'],dev['id2']],ignore_index=True));items=select_items_by_ids(items_path,ids,include_attributes=True)
    cats=items.set_index('id')['category'].astype(str);train['category']=train['id1'].map(cats);valid['category']=valid['id1'].map(cats)
    if train['category'].isna().any() or valid['category'].isna().any(): raise RuntimeError('category attach failed')

    feature_started=time.perf_counter()
    base_train=build_features_v2_chunked(items,train,attribute_importance=None,chunk_size=25000)
    base_valid=build_features_v2_chunked(items,valid,attribute_importance=None,chunk_size=25000)
    sparse=fit_transform_sparse_outer_fold(items,train,valid,max_char_features=120000,max_word_features=60000)
    x_train=pd.concat([base_train.reset_index(drop=True),sparse['train_features'].reset_index(drop=True)],axis=1)
    x_valid=pd.concat([base_valid.reset_index(drop=True),sparse['valid_features'].reset_index(drop=True)],axis=1)
    feature_seconds=time.perf_counter()-feature_started
    score=fit_predict_category_specialists(x_train,train['target'].to_numpy(),x_valid,seed=2026+held_fold,max_iter=max_iter,min_samples_leaf=15,l2_regularization=2.0)
    report=macro_ap_report(valid,score)
    pd.DataFrame({'row_index':held_rows,'fold':np.full(len(held_rows),held_fold,dtype=np.int8),'score':score}).sort_values('row_index').to_parquet(output_dir/f'v5-sparse-fold-{held_fold}-oof.parquet',index=False)
    payload={'version':'v5-sparse-category-specialist','held_fold':int(held_fold),'split_sha256':expected_split_sha,'gold_metric_opened':False,'gold_rows_used':0,'train_rows':int(len(train)),'valid_rows':int(len(valid)),'train_sparse_items':int(sparse['train_item_count']),'valid_sparse_items':int(sparse['valid_item_count']),'name_word_vocabulary_size':int(len(sparse['name_word_vocabulary'])),'feature_seconds':float(feature_seconds),'macro_average_precision':float(report['macro_average_precision']),'per_category_ap':report['per_category_ap'],'elapsed_seconds':float(time.perf_counter()-started)}
    (output_dir/f'v5-sparse-fold-{held_fold}-metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True),encoding='utf-8');return payload


def main()->int:
    p=argparse.ArgumentParser();p.add_argument('--items',type=Path,required=True);p.add_argument('--matches',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--expected-split-sha',required=True);p.add_argument('--held-fold',type=int,required=True);p.add_argument('--max-iter',type=int,default=300);a=p.parse_args();r=run_sparse_specialist_fold(items_path=a.items,matches_path=a.matches,manifest_path=a.manifest,output_dir=a.output_dir,expected_split_sha=a.expected_split_sha,held_fold=a.held_fold,max_iter=a.max_iter);print(json.dumps(r,ensure_ascii=False,sort_keys=True));return 0
if __name__=='__main__': raise SystemExit(main())
