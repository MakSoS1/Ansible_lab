from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_subset import select_items_by_ids
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .v5_semantic_stack import crossfit_semantic_stack
from .v5_validation import manifest_sha256


def run_combo(
    *,
    items_path: Path,
    matches_path: Path,
    manifest_path: Path,
    category_oof_path: Path,
    weighted_oof_path: Path,
    pretrained_oof_path: Path,
    output_dir: Path,
    expected_split_sha: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest_sha256(manifest) != expected_split_sha:
        raise ValueError('sealed split SHA mismatch')
    matches=pd.read_parquet(matches_path,columns=['id1','id2','target'])
    dev_rows, folds=development_rows_and_folds(manifest,total_rows=len(matches))
    dev=matches.iloc[dev_rows].reset_index(drop=True)
    ids=pd.unique(pd.concat([dev['id1'],dev['id2']],ignore_index=True))
    items=select_items_by_ids(items_path,ids,include_attributes=False)
    categories=items.set_index('id')['category'].astype(str)
    dev['category']=dev['id1'].map(categories)
    if dev['category'].isna().any(): raise RuntimeError('missing category')

    def aligned(path: Path, cols: list[str]) -> pd.DataFrame:
        f=pd.read_parquet(path,columns=['row_index',*cols]).sort_values('row_index').reset_index(drop=True)
        if f['row_index'].astype(np.int64).tolist()!=dev_rows.tolist():
            raise ValueError(f'OOF rows misaligned: {path}')
        return f

    category=aligned(category_oof_path,['score'])
    weighted=aligned(weighted_oof_path,['score'])
    pretrained=aligned(pretrained_oof_path,['score','embedding_cosine','embedding_mean_abs_diff','embedding_l2','embedding_max_abs_diff','embedding_mean_product','embedding_min_product','embedding_max_product'])
    features=pd.DataFrame({
        'weighted_specialist_score': weighted['score'].to_numpy(dtype=np.float32),
        'pretrained_stack_score': pretrained['score'].to_numpy(dtype=np.float32),
        'pretrained_cosine': pretrained['embedding_cosine'].to_numpy(dtype=np.float32),
        'pretrained_l1': pretrained['embedding_mean_abs_diff'].to_numpy(dtype=np.float32),
        'pretrained_l2': pretrained['embedding_l2'].to_numpy(dtype=np.float32),
        'pretrained_product': pretrained['embedding_mean_product'].to_numpy(dtype=np.float32),
    })
    result=crossfit_semantic_stack(dev,category['score'].to_numpy(dtype=np.float64),features,folds,seed=2026,max_iter=220)
    payload={
        'version':'v5-combo-category-weighted-pretrained',
        'split_sha256':expected_split_sha,
        'gold_metric_opened':False,
        'gold_rows_scored':0,
        'base_oof_macro_ap':float(result['base_macro_average_precision']),
        'combo_oof_macro_ap':float(result['macro_average_precision']),
        'delta_vs_base':float(result['delta_vs_base']),
        'fold_reports':result['fold_reports'],
        'per_category_ap':result['per_category_ap'],
    }
    (output_dir/'v5-combo-metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True),encoding='utf-8')
    pd.DataFrame({'row_index':dev_rows,'fold':folds,'score':result['scores']}).to_parquet(output_dir/'v5-combo-oof.parquet',index=False)
    return payload


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument('--items',type=Path,required=True);p.add_argument('--matches',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--category-oof',type=Path,required=True)
    p.add_argument('--weighted-oof',type=Path,required=True);p.add_argument('--pretrained-oof',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--expected-split-sha',required=True)
    a=p.parse_args(); result=run_combo(items_path=a.items,matches_path=a.matches,manifest_path=a.manifest,category_oof_path=a.category_oof,weighted_oof_path=a.weighted_oof,pretrained_oof_path=a.pretrained_oof,output_dir=a.output_dir,expected_split_sha=a.expected_split_sha)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True));return 0

if __name__=='__main__': raise SystemExit(main())
