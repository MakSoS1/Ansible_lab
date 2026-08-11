from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from .data_subset import select_items_by_ids
from .features import normalize_items
from .features_v2 import build_pair_features_v2
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_evaluation import macro_ap_report
from .v5_explicit_attributes import build_explicit_attribute_features,learn_explicit_attribute_keys
from .v5_validation import manifest_sha256

def run(*,items_path:Path,matches_path:Path,manifest_path:Path,output_dir:Path,expected_split_sha:str,held_fold:int,max_keys:int=40,min_support:int=30)->dict:
 started=time.perf_counter();output_dir.mkdir(parents=True,exist_ok=True);manifest=json.loads(manifest_path.read_text())
 if manifest_sha256(manifest)!=expected_split_sha: raise ValueError('split SHA mismatch')
 matches=pd.read_parquet(matches_path,columns=['id1','id2','target']);rows,folds=development_rows_and_folds(manifest,total_rows=len(matches));dev=matches.iloc[rows].reset_index(drop=True);tm=folds!=held_fold;vm=folds==held_fold
 ids=pd.unique(pd.concat([dev['id1'],dev['id2']],ignore_index=True));items=select_items_by_ids(items_path,ids,include_attributes=True);cache=normalize_items(items);cats=items.set_index('id')['category'].astype(str);dev['category']=dev['id1'].map(cats)
 train=dev.loc[tm].reset_index(drop=True);valid=dev.loc[vm].reset_index(drop=True);held_rows=rows[vm]
 spec=learn_explicit_attribute_keys(items,train,max_keys_per_category=max_keys,min_support=min_support,item_cache=cache)
 scores=np.full(len(valid),np.nan,dtype=float);category_reports={}
 for cat in sorted(valid['category'].astype(str).unique()):
  tr=train['category'].astype(str)==cat;va=valid['category'].astype(str)==cat
  if not tr.any(): raise ValueError(f'no train rows for {cat}')
  trp=train.loc[tr].reset_index(drop=True);vap=valid.loc[va].reset_index(drop=True)
  xbtr=build_pair_features_v2(items,trp,item_cache=cache);xbva=build_pair_features_v2(items,vap,item_cache=cache)
  xetr=build_explicit_attribute_features(items,trp,spec,item_cache=cache,category=cat);xeva=build_explicit_attribute_features(items,vap,spec,item_cache=cache,category=cat)
  xtr=pd.concat([xbtr.drop(columns=['category']).reset_index(drop=True),xetr.reset_index(drop=True)],axis=1).to_numpy(np.float32);xva=pd.concat([xbva.drop(columns=['category']).reset_index(drop=True),xeva.reset_index(drop=True)],axis=1).to_numpy(np.float32)
  model=HistGradientBoostingClassifier(loss='log_loss',learning_rate=.06,max_iter=350,max_leaf_nodes=31,min_samples_leaf=15,l2_regularization=3.0,early_stopping=False,random_state=2026+held_fold)
  model.fit(xtr,trp['target'].to_numpy(np.int8));pred=model.predict_proba(xva)[:,1];scores[np.flatnonzero(va.to_numpy())]=pred;category_reports[cat]={'keys':len(spec.get(cat,[])),'rows':int(va.sum())}
 if not np.isfinite(scores).all(): raise RuntimeError('missing scores')
 report=macro_ap_report(valid,scores);pd.DataFrame({'row_index':held_rows,'fold':np.full(len(held_rows),held_fold,dtype=np.int8),'score':scores}).sort_values('row_index').to_parquet(output_dir/f'v5-explicit-fold-{held_fold}-oof.parquet',index=False)
 payload={'version':'v5-explicit-attribute-specialists','held_fold':held_fold,'split_sha256':expected_split_sha,'gold_metric_opened':False,'gold_rows_used':0,'macro_average_precision':report['macro_average_precision'],'per_category_ap':report['per_category_ap'],'selected_keys':int(sum(len(v) for v in spec.values())),'category_reports':category_reports,'elapsed_seconds':time.perf_counter()-started};(output_dir/f'v5-explicit-fold-{held_fold}-metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True));return payload

def main():
 p=argparse.ArgumentParser();p.add_argument('--items',type=Path,required=True);p.add_argument('--matches',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--expected-split-sha',required=True);p.add_argument('--held-fold',type=int,required=True);a=p.parse_args();print(json.dumps(run(items_path=a.items,matches_path=a.matches,manifest_path=a.manifest,output_dir=a.output_dir,expected_split_sha=a.expected_split_sha,held_fold=a.held_fold),ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
