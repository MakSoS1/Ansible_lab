from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest


def _source_bundle(path: Path) -> None:
    old_meta={'coverage':0.95,'strict_selected_oof_macro_ap':0.6006003614522999,'selection_gold_metric_opened':False,'selection_gold_rows_scored':0}
    with zipfile.ZipFile(path,'w') as zf:
        zf.writestr('run.py','STALE=True\n')
        zf.writestr('ecup_matching/submission/predict_v5.py','STALE=True\n')
        zf.writestr('ecup_matching/submission/predict_v6.py','STALE=True\n')
        zf.writestr('model_v5_structured.joblib',b'structured')
        zf.writestr('model_v5_contrastive/config.json','{}')
        zf.writestr('model_v5_teacher/config.json','{}')
        zf.writestr('model_v6_category_shrunk.json','{"old":true}')
        zf.writestr('model_v6_hgb_meta.joblib',b'old-hgb')
        zf.writestr('model_v6_gate_metadata.json',json.dumps(old_meta))


def _gate85_files(tmp_path: Path):
    cat=tmp_path/'category.json'; cat.write_text('{"coverage":0.85}',encoding='utf-8')
    hgb=tmp_path/'hgb.joblib'; hgb.write_bytes(b'gate85-hgb')
    meta=tmp_path/'meta.json'; meta.write_text(json.dumps({
        'version':'v8-gate-production-refit','coverage':0.85,
        'base_strict_oof_macro_ap':0.5999300791828578,
        'fold_local_graph_strict_oof_macro_ap':0.6021573018691804,
        'strict_selected_oof_macro_ap':0.6021573018691804,
        'split_sha256':'aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b',
        'graph_config':{'rb':0.0,'rt':0.0,'ep':0.02,'ap':0.01},
        'selection_gold_metric_opened':False,'selection_gold_rows_scored':0,
    }),encoding='utf-8')
    return cat,hgb,meta


def test_gate85_builder_uses_exact_refit_and_current_runtime(tmp_path,monkeypatch):
    import ecup_matching.submission.build_submission_v8_gate85 as module
    source=tmp_path/'source.zip'; _source_bundle(source)
    cat,hgb,meta=_gate85_files(tmp_path)
    graph=tmp_path/'v8_graph.py'; graph.write_text('GRAPH=True\n')
    post=tmp_path/'v8_submission_graph.py'; post.write_text('POST=True\n')
    def fake_copy(root):
        p=Path(root)/'ecup_matching/submission'; p.mkdir(parents=True,exist_ok=True)
        (p/'predict_v5.py').write_text('def _legacy_text_cache(*a,**k): pass\n')
        (p/'predict_v6.py').write_text('# _structured_scores_streaming run_structured_chunks torch_autocast build_dual_text_cache\ndef predict_to_csv_v6(**k): pass\n')
        (p/'v6_text_cache.py').write_text('def build_dual_text_cache(*a,**k): return {},{}\n')
        return ['ecup_matching/submission/predict_v5.py','ecup_matching/submission/predict_v6.py','ecup_matching/submission/v6_text_cache.py']
    monkeypatch.setattr(module,'copy_runtime_closure',fake_copy)
    monkeypatch.setattr(module,'runtime_import_closure',lambda:())
    out=tmp_path/'final.zip'
    result=module.build_v8_gate85(source_v6_zip=source,gate85_category_path=cat,gate85_hgb_path=hgb,gate85_metadata_path=meta,output_zip=out,v8_graph_source=graph,v8_submission_graph_source=post,source_commit='b'*40)
    assert result['coverage']==pytest.approx(.85)
    assert result['fold_local_graph_strict_oof_macro_ap']==pytest.approx(0.6021573018691804)
    with zipfile.ZipFile(out) as zf:
        saved=json.loads(zf.read('model_v6_gate_metadata.json'))
        v8=json.loads(zf.read('v8_metadata.json'))
        assert saved['coverage']==pytest.approx(.85)
        assert saved['fold_local_graph_strict_oof_macro_ap']==pytest.approx(0.6021573018691804)
        assert zf.read('model_v6_hgb_meta.joblib')==b'gate85-hgb'
        assert v8['version']=='v8-gate85-fp16-dualcache-graph'
        assert v8['base']['fold_local_graph_strict_oof_macro_ap'] > 0.6018115534135564
        assert 'build_dual_text_cache' in zf.read('ecup_matching/submission/predict_v6.py').decode()
        run=zf.read('run.py').decode(); assert 'EXPECTED_COVERAGE = 0.85' in run and 'apply_graph_to_prediction' in run
        assert v8['sealed_gold_opened'] is False
