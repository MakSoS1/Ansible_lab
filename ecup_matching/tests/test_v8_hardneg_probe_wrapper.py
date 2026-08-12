from ecup_matching.ml import run_v7_fold0_probe as base
from ecup_matching.ml import run_v8_hardneg_probe_frozen as wrapper
from ecup_matching.ml.v8_hardneg import train_pair_phase_v8_hardneg


def test_probe_wrapper_patches_only_training_function_and_restores(monkeypatch):
    original = base.train_pair_phase
    observed = {}

    def fake_main():
        observed['train_pair_phase'] = base.train_pair_phase
        return 17

    monkeypatch.setattr(base, 'main', fake_main)
    assert wrapper.main() == 17
    assert observed['train_pair_phase'] is train_pair_phase_v8_hardneg
    assert base.train_pair_phase is original
