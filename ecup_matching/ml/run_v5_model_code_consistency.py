from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .features import normalize_items
from .run_v5_fixed_blend import align_oof_frame
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_evaluation import macro_ap_report
from .v5_fixed_blend import percentile_rank
from .v5_model_code_consistency import model_code_consistency
from .v5_validation import manifest_sha256

ANCHOR_COLUMN = "candidate_current4_plus_teacher"
CURRENT5_COLUMNS = (
    "source_weak",
    "source_sparse",
    "source_explicit",
    "source_contrastive_cosine",
    "source_teacher2_raw",
)


def run_model_code_consistency(*,items_path:Path,matches_path:Path,manifest_path:Path,anchor_oof_path:Path,output_dir:Path,expected_split_sha:str)->dict:
    output_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest_sha256(manifest)!=expected_split_sha: raise ValueError('sealed split SHA mismatch')
    matches=pd.read_parquet(matches_path,columns=['id1','id2','target'])
    dev_rows,folds=development_rows_and_folds(manifest,total_rows=len(matches));folds=np.asarray(folds,dtype=np.int16)
    anchor=align_oof_frame([anchor_oof_path],expected_rows=dev_rows,expected_folds=folds,required_columns=(ANCHOR_COLUMN,*CURRENT5_COLUMNS),source_name='current5_anchor')
    dev=matches.iloc[dev_rows].reset_index(drop=True)
    wanted=pd.unique(pd.concat([dev['id1'],dev['id2']],ignore_index=True));items=select_items_by_ids(items_path,wanted,include_attributes=True);cache=normalize_items(items);cats=items.set_index('id')['category'].astype(str);dev['category']=dev['id1'].map(cats)
    if dev['category'].isna().any(): raise RuntimeError('failed to attach official categories')
    signal=np.fromiter((model_code_consistency(cache[a],cache[b]) for a,b in dev[['id1','id2']].itertuples(index=False,name=None)),dtype=np.float64,count=len(dev))
    anchor_scores=anchor[ANCHOR_COLUMN].to_numpy(np.float64);anchor_ap=float(macro_ap_report(dev,anchor_scores,strict_official=True)['macro_average_precision']);expected_anchor=0.5952697490140912
    if abs(anchor_ap-expected_anchor)>1e-12: raise RuntimeError(f'anchor mismatch {anchor_ap}')
    candidate=np.mean(np.vstack([percentile_rank(anchor[c].to_numpy(np.float64)) for c in CURRENT5_COLUMNS]+[percentile_rank(signal)]),axis=0)
    direct=macro_ap_report(dev,signal,strict_official=True);report=macro_ap_report(dev,candidate,strict_official=True);candidate_ap=float(report['macro_average_precision'])
    fold_reports=[]
    for fold in sorted(np.unique(folds).tolist()):
        mask=folds==fold;ff=dev.loc[mask].reset_index(drop=True);fa=float(macro_ap_report(ff,anchor_scores[mask])['macro_average_precision']);fc=float(macro_ap_report(ff,candidate[mask])['macro_average_precision']);fold_reports.append({'fold':int(fold),'rows':int(mask.sum()),'anchor_macro_average_precision':fa,'macro_average_precision':fc,'delta_vs_anchor':float(fc-fa)})
    min_delta=min(x['delta_vs_anchor'] for x in fold_reports);keep=bool(candidate_ap>anchor_ap and min_delta>=-0.001)
    payload={'version':'v5-target-free-model-code-consistency','split_sha256':expected_split_sha,'development_rows':int(len(dev)),'gold_metric_opened':False,'gold_rows_scored':0,'target_fitted_blender':False,'predeclared_candidate_count':1,'anchor_macro_average_precision':anchor_ap,'model_code_consistency_macro_average_precision':float(direct['macro_average_precision']),'model_code_nonzero_rows':int((signal!=0).sum()),'model_code_nonzero_rate':float((signal!=0).mean()),'candidate_name':'current5_plus_model_code_consistency','candidate_macro_average_precision':candidate_ap,'delta_vs_anchor':float(candidate_ap-anchor_ap),'min_fold_delta_vs_anchor':float(min_delta),'fold_reports':fold_reports,'per_category_ap':report['per_category_ap'],'keep_eligible':keep,'target_0_60_reached':bool(candidate_ap>=0.60)}
    (output_dir/'v5-model-code-consistency-metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True),encoding='utf-8');pd.DataFrame({'row_index':dev_rows,'fold':folds,'model_code_consistency_score':signal,'candidate_score':candidate}).to_parquet(output_dir/'v5-model-code-consistency-oof.parquet',index=False);return payload

def main()->int:
    p=argparse.ArgumentParser();p.add_argument('--items',type=Path,required=True);p.add_argument('--matches',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--anchor-oof',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--expected-split-sha',required=True);a=p.parse_args();print(json.dumps(run_model_code_consistency(items_path=a.items,matches_path=a.matches,manifest_path=a.manifest,anchor_oof_path=a.anchor_oof,output_dir=a.output_dir,expected_split_sha=a.expected_split_sha),ensure_ascii=False,sort_keys=True));return 0
if __name__=='__main__': raise SystemExit(main())
