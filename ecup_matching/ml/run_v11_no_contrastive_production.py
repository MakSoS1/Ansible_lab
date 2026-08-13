from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecup_matching.ml.run_v6_fast_production import fit_v6_fast_production


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--items',type=Path,required=True); p.add_argument('--matches',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--anchor-oof',type=Path,required=True)
    p.add_argument('--typed-fusion-oof',type=Path,required=True); p.add_argument('--category-output',type=Path,required=True)
    p.add_argument('--hgb-output',type=Path,required=True); p.add_argument('--metadata-output',type=Path,required=True)
    p.add_argument('--expected-split-sha',required=True); a=p.parse_args()
    fit_v6_fast_production(candidate='no_contrastive',selected_oof_macro_ap=0.60,items_path=a.items,matches_path=a.matches,manifest_path=a.manifest,anchor_oof_path=a.anchor_oof,typed_fusion_oof_path=a.typed_fusion_oof,category_output_path=a.category_output,hgb_output_path=a.hgb_output,metadata_output_path=a.metadata_output,expected_split_sha=a.expected_split_sha)
    m=json.loads(a.metadata_output.read_text(encoding='utf-8'))
    m.update(version='v11-no-contrastive-production',strict_selected_oof_macro_ap=0.5928725263319,fold_local_graph_oof_macro_ap=0.5978943607354008,target_stress_mean=0.45353679907723865,quality_gate_macro_ap=None,leaderboard_score=None,contrastive_runtime=False,teacher_runtime=True)
    a.metadata_output.write_text(json.dumps(m,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return 0

if __name__=='__main__': raise SystemExit(main())
