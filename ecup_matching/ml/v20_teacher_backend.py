from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from typing import Any
from urllib.parse import urlparse


_BACKENDS = {"openai-http", "transformers-causal", "transformers-seq2seq"}


@dataclass(frozen=True)
class TeacherBackendSpec:
    name: str
    family: str
    model_id: str
    revision: str
    backend: str
    quantization: str = "none"
    endpoint: str | None = None
    model_file: str | None = None
    model_file_sha256: str | None = None


def _is_exact_hex(value: str, length: int) -> bool:
    if len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_loopback_http(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_backend_spec(spec: TeacherBackendSpec) -> TeacherBackendSpec:
    if spec.backend not in _BACKENDS:
        raise ValueError(f"unsupported teacher backend: {spec.backend}")
    if not spec.name or not spec.family or not spec.model_id:
        raise ValueError("teacher name/family/model_id must be non-empty")
    if not _is_exact_hex(str(spec.revision), 40):
        raise ValueError("revision must be an exact 40-hex commit SHA")

    if spec.backend == "openai-http":
        if not spec.endpoint or not _is_loopback_http(spec.endpoint):
            raise ValueError("openai-http teacher endpoint must be loopback http")
        if not spec.model_file:
            raise ValueError("openai-http teacher requires model_file")
        if not spec.model_file_sha256 or not _is_exact_hex(spec.model_file_sha256, 64):
            raise ValueError("openai-http teacher requires exact model_file_sha256")
        if str(spec.quantization).lower() in {"", "none", "fp16", "bf16"}:
            raise ValueError("openai-http teacher requires explicit quantization")
    else:
        if spec.endpoint is not None:
            raise ValueError("transformers teacher must not define endpoint")
        if spec.model_file is not None or spec.model_file_sha256 is not None:
            raise ValueError("transformers teacher must not define model_file/model_file_sha256")
    return spec


def build_openai_chat_request(
    spec: TeacherBackendSpec,
    *,
    system: str,
    user: str,
    max_new_tokens: int,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    validate_backend_spec(spec)
    if spec.backend != "openai-http":
        raise ValueError("build_openai_chat_request requires openai-http backend")
    if int(max_new_tokens) <= 0:
        raise ValueError("max_new_tokens must be positive")
    payload: dict[str, Any] = {
        "model": spec.name,
        "messages": [
            {"role": "system", "content": str(system)},
            {"role": "user", "content": str(user)},
        ],
        "temperature": 0,
        "seed": int(seed),
        "max_tokens": int(max_new_tokens),
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    return str(spec.endpoint), payload


def extract_openai_text(payload: dict[str, Any]) -> str:
    try:
        choices = payload["choices"]
        content = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("invalid OpenAI-compatible chat completion payload") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenAI-compatible response content is empty")
    return content


__all__ = [
    "TeacherBackendSpec",
    "validate_backend_spec",
    "build_openai_chat_request",
    "extract_openai_text",
]
