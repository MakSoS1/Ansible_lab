from pathlib import Path
import json
import zipfile

import pytest

from ecup_matching.submission.build_submission_v8_graph import (
    GRAPH_CONFIG,
    build_v8_from_v5_zip,
    safe_extract_zip,
)


def _make_v5_zip(path: Path) -> None:
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr('run.py', 'OLD_V5_RUN = True\n')
        zf.writestr('ecup_matching/__init__.py', '')
        zf.writestr('ecup_matching/ml/__init__.py', '')
        zf.writestr('ecup_matching/submission/__init__.py', '')
        zf.writestr('ecup_matching/submission/predict_v5.py', 'def predict_to_csv_v5(**kwargs): return None\n')
        zf.writestr('model_v5_structured.joblib', b'model')


def _make_v6_zip(path: Path) -> None:
    metadata = {
        'coverage': 0.95,
        'strict_selected_oof_macro_ap': 0.6006003614522999,
        'selection_gold_metric_opened': False,
        'selection_gold_rows_scored': 0,
    }
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr('run.py', 'OLD_V6_RUN = True\n')
        zf.writestr('ecup_matching/__init__.py', '')
        zf.writestr('ecup_matching/ml/__init__.py', '')
        zf.writestr('ecup_matching/submission/__init__.py', '')
        # Deliberately stale runtime: models/metadata are valid but streaming code is absent.
        zf.writestr('ecup_matching/submission/predict_v6.py', 'def predict_to_csv_v6(**kwargs): return None\n')
        zf.writestr('model_v5_structured.joblib', b'model')
        zf.writestr('model_v5_contrastive/config.json', '{}')
        zf.writestr('model_v5_teacher/config.json', '{}')
        zf.writestr('model_v6_category_shrunk.json', '{}')
        zf.writestr('model_v6_hgb_meta.joblib', b'model')
        zf.writestr('model_v6_gate_metadata.json', json.dumps(metadata))


def _runtime_sources(tmp_path: Path) -> dict[str, Path]:
    predict = tmp_path/'predict_v6.py'
    parallel = tmp_path/'v6_parallel.py'
    fast = tmp_path/'v6_fast.py'
    gate = tmp_path/'v6_teacher_gate.py'
    predict.write_text(
        'def _structured_scores_streaming(): pass\n'
        'def predict_to_csv_v6(**kwargs): return None\n'
        'from .v6_parallel import run_structured_chunks\n',
        encoding='utf-8',
    )
    parallel.write_text('def run_structured_chunks(**kwargs): return {}\n', encoding='utf-8')
    fast.write_text('def collect_chunked_scores(**kwargs): return {}\n', encoding='utf-8')
    gate.write_text('def disagreement_gate_mask(*args, **kwargs): return None\n', encoding='utf-8')
    return {
        'predict_v6_source': predict,
        'v6_parallel_source': parallel,
        'v6_fast_source': fast,
        'v6_teacher_gate_source': gate,
    }


def test_safe_extract_rejects_traversal_and_symlink(tmp_path):
    bad = tmp_path/'bad.zip'
    with zipfile.ZipFile(bad, 'w') as zf:
        zf.writestr('run.py', 'x')
        zf.writestr('../escape', 'boom')
    with pytest.raises(ValueError, match='unsafe'):
        safe_extract_zip(bad, tmp_path/'bad-out')

    link = tmp_path/'link.zip'
    info = zipfile.ZipInfo('link')
    info.create_system = 3
    info.external_attr = (0o120777 << 16)
    with zipfile.ZipFile(link, 'w') as zf:
        zf.writestr('run.py', 'x')
        zf.writestr(info, 'target')
    with pytest.raises(ValueError, match='symlink'):
        safe_extract_zip(link, tmp_path/'link-out')


def test_builder_reuses_v5_payload_and_adds_only_graph_runtime(tmp_path):
    source = tmp_path/'v5.zip'; _make_v5_zip(source)
    graph = tmp_path/'v8_graph.py'; graph.write_text('GRAPH = True\n', encoding='utf-8')
    post = tmp_path/'v8_submission_graph.py'; post.write_text('POST = True\n', encoding='utf-8')
    out = tmp_path/'v8.zip'
    result = build_v8_from_v5_zip(
        source,
        out,
        v8_graph_source=graph,
        v8_submission_graph_source=post,
        source_v5_metric=0.6018115534135564,
        graph_oof_delta=0.00046740999622918444,
        source_commit='a'*40,
    )
    assert result['archive_bytes'] == out.stat().st_size
    with zipfile.ZipFile(out) as zf:
        names=set(zf.namelist())
        assert 'model_v5_structured.joblib' in names
        assert 'ecup_matching/ml/v8_graph.py' in names
        assert 'ecup_matching/ml/v8_submission_graph.py' in names
        assert 'ecup_matching/ml/__init__.py' in names
        run=zf.read('run.py').decode()
        assert 'predict_to_csv_v5' in run
        assert 'apply_graph_to_prediction' in run
        assert '--output_path' in run and '--items_path' in run and '--matches_path' in run
        assert "'rb': 0.0" in run and "'ep': 0.02" in run and "'ap': 0.01" in run
        meta=json.loads(zf.read('v8_metadata.json'))
        assert meta['version']=='v8-v5-best-plus-graph'
        assert meta['base']['strict_oof_macro_ap']==pytest.approx(0.6018115534135564)
        assert meta['graph']['config']==GRAPH_CONFIG
        assert meta['graph']['strict_oof_full_delta']==pytest.approx(0.00046740999622918444)
        assert meta['sealed_gold_opened'] is False
        assert meta['true_test_prevalence_claimed'] is False


def test_builder_refuses_source_without_v5_runtime_contract(tmp_path):
    source=tmp_path/'bad-v5.zip'
    with zipfile.ZipFile(source,'w') as zf:
        zf.writestr('run.py','x')
    graph=tmp_path/'g.py'; graph.write_text('x')
    post=tmp_path/'p.py'; post.write_text('x')
    with pytest.raises(ValueError, match='predict_v5'):
        build_v8_from_v5_zip(source,tmp_path/'out.zip',v8_graph_source=graph,v8_submission_graph_source=post,
                             source_v5_metric=.6,graph_oof_delta=.001,source_commit='a'*40)


def test_v8_fast_builder_uses_v6_runtime_not_v5_runtime(tmp_path):
    from ecup_matching.submission.build_submission_v8_v6graph import build_v8_from_v6_zip

    source = tmp_path/'v6.zip'; _make_v6_zip(source)
    graph = tmp_path/'v8_graph.py'; graph.write_text('GRAPH = True\n', encoding='utf-8')
    post = tmp_path/'v8_submission_graph.py'; post.write_text('POST = True\n', encoding='utf-8')
    out = tmp_path/'v8-fast.zip'
    result = build_v8_from_v6_zip(
        source,
        out,
        v8_graph_source=graph,
        v8_submission_graph_source=post,
        source_v6_metric=0.6006003614522999,
        graph_oof_delta=0.000442083907,
        source_commit='b'*40,
        **_runtime_sources(tmp_path),
    )
    assert result['archive_bytes'] == out.stat().st_size
    with zipfile.ZipFile(out) as zf:
        run=zf.read('run.py').decode()
        assert 'predict_to_csv_v6' in run
        assert 'predict_to_csv_v5(' not in run
        assert 'apply_graph_to_prediction' in run
        assert 'model_v6_gate_metadata.json' in zf.namelist()
        meta=json.loads(zf.read('v8_metadata.json'))
        assert meta['version']=='v8-v6-fast-gate95-plus-graph'
        assert meta['base']['strict_oof_macro_ap']==pytest.approx(0.6006003614522999)
        assert meta['base']['teacher_coverage']==pytest.approx(0.95)
        assert meta['sealed_gold_opened'] is False


def test_v8_fast_builder_overlays_streaming_runtime_over_stale_v6_zip(tmp_path):
    from ecup_matching.submission.build_submission_v8_v6graph import build_v8_from_v6_zip

    source = tmp_path/'v6-stale.zip'; _make_v6_zip(source)
    graph = tmp_path/'v8_graph.py'; graph.write_text('GRAPH = True\n', encoding='utf-8')
    post = tmp_path/'v8_submission_graph.py'; post.write_text('POST = True\n', encoding='utf-8')
    out = tmp_path/'v8-fast.zip'
    build_v8_from_v6_zip(
        source,
        out,
        v8_graph_source=graph,
        v8_submission_graph_source=post,
        source_v6_metric=0.6006003614522999,
        graph_oof_delta=0.000442083907,
        source_commit='c'*40,
        **_runtime_sources(tmp_path),
    )
    with zipfile.ZipFile(out) as zf:
        predictor = zf.read('ecup_matching/submission/predict_v6.py').decode('utf-8')
        parallel = zf.read('ecup_matching/submission/v6_parallel.py').decode('utf-8')
        assert '_structured_scores_streaming' in predictor
        assert 'run_structured_chunks' in predictor
        assert 'run_structured_chunks' in parallel
        assert zf.read('ecup_matching/submission/v6_fast.py')
        assert zf.read('ecup_matching/ml/v6_teacher_gate.py')
