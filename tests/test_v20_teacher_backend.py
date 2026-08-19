from __future__ import annotations

import pytest

from ecup_matching.ml.v20_teacher_backend import (
    TeacherBackendSpec,
    build_openai_chat_request,
    extract_openai_text,
    validate_backend_spec,
)


def test_http_backend_requires_loopback_endpoint():
    spec = TeacherBackendSpec(
        name="qwen35-q4",
        family="qwen",
        model_id="Qwen/Qwen3.5-4B",
        revision="a" * 40,
        backend="openai-http",
        quantization="Q4_K_M",
        endpoint="https://example.com/v1/chat/completions",
        model_file="Qwen3.5-4B-Q4_K_M.gguf",
        model_file_sha256="1" * 64,
    )
    with pytest.raises(ValueError, match="loopback"):
        validate_backend_spec(spec)


def test_http_backend_accepts_local_pinned_quantized_model():
    spec = TeacherBackendSpec(
        name="gemma4-q4",
        family="gemma4",
        model_id="google/gemma-4-E2B-it",
        revision="b" * 40,
        backend="openai-http",
        quantization="Q4_K_M",
        endpoint="http://127.0.0.1:18081/v1/chat/completions",
        model_file="google_gemma-4-E2B-it-Q4_K_M.gguf",
        model_file_sha256="2" * 64,
    )
    assert validate_backend_spec(spec) == spec


def test_transformers_backend_rejects_quantized_file_fields():
    spec = TeacherBackendSpec(
        name="euro",
        family="eurollm",
        model_id="utter-project/EuroLLM-1.7B-Instruct",
        revision="c" * 40,
        backend="transformers-causal",
        quantization="none",
        model_file="unexpected.gguf",
        model_file_sha256="3" * 64,
    )
    with pytest.raises(ValueError, match="model_file"):
        validate_backend_spec(spec)


def test_openai_request_is_deterministic_json_constrained_and_nonthinking():
    spec = TeacherBackendSpec(
        name="qwen35-q4",
        family="qwen",
        model_id="Qwen/Qwen3.5-4B",
        revision="d" * 40,
        backend="openai-http",
        quantization="Q4_K_M",
        endpoint="http://localhost:18080/v1/chat/completions",
        model_file="Qwen3.5-4B-Q4_K_M.gguf",
        model_file_sha256="4" * 64,
    )
    url, payload = build_openai_chat_request(
        spec,
        system="SYSTEM",
        user="USER",
        max_new_tokens=128,
        seed=2026,
    )
    assert url == spec.endpoint
    assert payload["temperature"] == 0
    assert payload["seed"] == 2026
    assert payload["max_tokens"] == 128
    assert payload["response_format"]["type"] == "json_object"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["reasoning_effort"] == "none"
    assert payload["messages"] == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
    ]


def test_extract_openai_text_reads_chat_completion_shape():
    payload = {"choices": [{"message": {"content": '{"verdict":"MATCH"}'}}]}
    assert extract_openai_text(payload) == '{"verdict":"MATCH"}'


def test_revision_and_file_hash_must_be_exact():
    spec = TeacherBackendSpec(
        name="qwen",
        family="qwen",
        model_id="Qwen/Qwen3.5-4B",
        revision="main",
        backend="openai-http",
        quantization="Q4_K_M",
        endpoint="http://127.0.0.1:18080/v1/chat/completions",
        model_file="x.gguf",
        model_file_sha256="short",
    )
    with pytest.raises(ValueError, match="revision"):
        validate_backend_spec(spec)
