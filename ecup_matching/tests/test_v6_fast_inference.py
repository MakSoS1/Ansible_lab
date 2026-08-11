import contextlib

import pytest

from ecup_matching.submission.v6_fast import (
    aligned_pair_texts,
    batch_index_ranges,
    move_token_batch,
    select_runtime_config,
    torch_autocast,
)


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


class _FakeTensor:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def to(self, device, *, non_blocking=False):
        self.calls.append((device, non_blocking))
        return self


class _FakeTorch:
    float16 = object()

    def __init__(self):
        self.calls = []

    def autocast(self, *, device_type, dtype):
        self.calls.append((device_type, dtype))
        return contextlib.nullcontext("cuda-autocast")


def test_move_token_batch_uses_non_blocking_cuda_transfer():
    config = select_runtime_config(total_memory_bytes=8 * GIB, device_type="cuda")
    first = _FakeTensor(1)
    second = _FakeTensor(2)
    moved = move_token_batch({"input_ids": first, "attention_mask": second}, config)
    assert moved == {"input_ids": first, "attention_mask": second}
    assert first.calls == [("cuda", True)]
    assert second.calls == [("cuda", True)]


def test_torch_autocast_uses_float16_on_cuda_and_noop_on_cpu():
    fake_torch = _FakeTorch()
    cuda_config = select_runtime_config(total_memory_bytes=8 * GIB, device_type="cuda")
    with torch_autocast(fake_torch, cuda_config) as marker:
        assert marker == "cuda-autocast"
    assert fake_torch.calls == [("cuda", fake_torch.float16)]

    cpu_config = select_runtime_config(total_memory_bytes=0, device_type="cpu")
    with torch_autocast(fake_torch, cpu_config) as marker:
        assert marker is None
    assert fake_torch.calls == [("cuda", fake_torch.float16)]


def test_aligned_pair_texts_preserves_pair_order_without_dataframe_slicing():
    texts = {10: "ten", 20: "twenty", 30: "thirty"}
    left, right = aligned_pair_texts([30, 10, 20], [10, 30, 10], texts)
    assert left == ["thirty", "ten", "twenty"]
    assert right == ["ten", "thirty", "ten"]


def test_aligned_pair_texts_rejects_misaligned_or_missing_ids():
    with pytest.raises(ValueError, match="equal length"):
        aligned_pair_texts([1], [1, 2], {1: "one", 2: "two"})
    with pytest.raises(KeyError):
        aligned_pair_texts([1], [2], {1: "one"})
