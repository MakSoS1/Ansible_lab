from __future__ import annotations
import argparse,json
from pathlib import Path
from ecup_matching.submission.predict_v15_residual import predict_to_csv_v15_residual

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--output_path',type=Path,required=True); p.add_argument('--items_path',type=Path,required=True); p.add_argument('--matches_path',type=Path,required=True); a=p.parse_args(); root=Path(__file__).resolve().parent if Path(__file__).name=='run.py' else Path(__file__).resolve().parents[2]
    meta=json.loads((root/'model_v15_metadata.json').read_text()); assert meta['gold_metric_opened'] is False and meta['gold_rows_scored']==0
    r=predict_to_csv_v15_residual(items_path=a.items_path,matches_path=a.matches_path,teacher_dir=root/'model_v7_teacher',residual_path=root/'v15_r1_residual.pt',output_path=a.output_path,max_length=int(meta['max_length']),max_chars=int(meta['max_chars']),batch_size=int(meta['inference_batch_size']))
    print('V15_RUNTIME='+json.dumps(r,sort_keys=True),flush=True); return 0
if __name__=='__main__': raise SystemExit(main())
