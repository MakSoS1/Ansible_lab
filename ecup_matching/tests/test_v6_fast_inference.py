import pytest

from ecup_matching.submission.v6_fast import batch_index_ranges, select_runtime_config


GIB = 1024**3


def test_select_runtime_config_cpu_is_conservative_and_deterministic():
    config = select_runtime_config(total_memory_bytes=0, device_type="cpu")
    assert config.device == "cpu"
    assert config.contrastive_batch == 32
    assert config.teacher_batch == 16
    assert config.autocast_dtype is None
    assert config.non_blocking is False


def test_select_runtime_config_8gib_cuda_targets_rtx2060():
    config = select_runtime_config(total_memory_bytes=8 * GIB, device_type="cuda")
    assert config.device == "cuda"
    assert config.contrastive_batch == 256
    assert config.teacher_batch == 96
    assert config.autocast_dtype == "float16"
    assert config.non_blocking is True


def test_select_runtime_config_24gib_cuda_scales_batches():
    config = select_runtime_config(total_memory_bytes=24 * GIB, device_type="cuda")
    assert config.contrastive_batch == 512
    assert config.teacher_batch == 192
    assert config.autocast_dtype == "float16"


def test_select_runtime_config_60gib_cuda_uses_large_batches():
    config = select_runtime_config(total_memory_bytes=60 * GIB, device_type="cuda")
    assert config.contrastive_batch == 1024
    assert config.teacher_batch == 384
    assert config.autocast_dtype == "float16"


def test_batch_index_ranges_covers_every_row_once():
    assert list(batch_index_ranges(10, 4)) == [(0, 4), (4, 8), (8, 10)]
    assert list(batch_index_ranges(0, 4)) == []


def test_batch_index_ranges_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="row_count"):
        list(batch_index_ranges(-1, 4))
    with pytest.raises(ValueError, match="batch_size"):
        list(batch_index_ranges(10, 0))
