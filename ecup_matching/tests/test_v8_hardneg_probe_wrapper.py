from ecup_matching.ml import run_v7_fold0_probe as base
from ecup_matching.ml import run_v8_hardneg_probe_frozen as wrapper
from ecup_matching.ml.run_v7_outer_oof_frozen import _load_immutable_manifest
from ecup_matching.ml.v8_hardneg import train_pair_phase_v8_hardneg


def test_probe_wrapper_patches_training_and_frozen_split_then_restores(monkeypatch):
    original_train = base.train_pair_phase
    original_split = base._build_immutable_manifest
    observed = {}

    def fake_main():
        observed['train_pair_phase'] = base.train_pair_phase
        observed['split_builder'] = base._build_immutable_manifest
        return 17

    monkeypatch.setattr(base, 'main', fake_main)
    assert wrapper.main() == 17
    assert observed['train_pair_phase'] is train_pair_phase_v8_hardneg
    assert observed['split_builder'] is _load_immutable_manifest
    assert base.train_pair_phase is original_train
    assert base._build_immutable_manifest is original_split
