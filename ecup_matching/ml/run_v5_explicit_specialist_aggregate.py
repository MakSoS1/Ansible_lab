from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from .data_subset import select_items_by_ids
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_oof_aggregate import aggregate_oof_scores
from .v5_validation import manifest_sha256

def run(*,fold_files:list[Path],items_path:Path,matches_path:Path,manifest_path:Path,category_oof_path:Path,output_dir:Path,expected_split_sha:str)->dict:
 output_dir.mkdir(parents=True,exist_ok=True);m=json.loads(manifest_path.read_text());
 if manifest_sha256(m)!=expected_split_sha:raise ValueError('split SHA mismatch')
 matches=pd.read_parquet(matches_path,columns=['id1','id2','target']);rows,folds=development_rows_and_folds(m,total_rows=len(matches));cand=pd.concat([pd.read_parquet(p) for p in fold_files],ignore_index=True).sort_values('row_index').reset_index(drop=True)
 if cand['row_index'].duplicated().any() or cand['row_index'].astype(np.int64).tolist()!=rows.tolist():raise ValueError('OOF coverage mismatch')
 dev=matches.iloc[rows].reset_index(drop=True);ids=pd.unique(pd.concat([dev.id1,dev.id2],ignore_index=True));items=select_items_by_ids(items_path,ids,include_attributes=False);dev['category']=dev.id1.map(items.set_index('id').category.astype(str));base=pd.read_parquet(category_oof_path,columns=['row_index','score']).sort_values('row_index')
 r=aggregate_oof_scores(dev,base.score.to_numpy(float),cand.score.to_numpy(float),folds);payload={'version':'v5-explicit-attribute-specialists','split_sha256':expected_split_sha,'gold_metric_opened':False,'gold_rows_scored':0,'category_base_oof_macro_ap':r['base_macro_average_precision'],'explicit_specialist_oof_macro_ap':r['macro_average_precision'],'delta_vs_category_base':r['delta_vs_base'],'fold_reports':r['fold_reports'],'per_category_ap':r['per_category_ap']};(output_dir/'v5-explicit-specialist-metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True));cand.to_parquet(output_dir/'v5-explicit-specialist-oof.parquet',index=False);return payload

def main():
 p=argparse.ArgumentParser();p.add_argument('--fold-dir',type=Path,required=True);p.add_argument('--items',type=Path,required=True);p.add_argument('--matches',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--category-oof',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--expected-split-sha',required=True);a=p.parse_args();print(json.dumps(run(fold_files=sorted(a.fold_dir.rglob('v5-explicit-fold-*-oof.parquet')),items_path=a.items,matches_path=a.matches,manifest_path=a.manifest,category_oof_path=a.category_oof,output_dir=a.output_dir,expected_split_sha=a.expected_split_sha),ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
