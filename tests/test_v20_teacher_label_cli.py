from __future__ import annotations

from ecup_matching.ml.run_v20_teacher_label import SYSTEM_RU, build_arg_parser


def test_teacher_prompt_is_russian_but_keeps_machine_schema_keys():
    lower = SYSTEM_RU.lower()
    assert "товар" in lower
    assert "одним и тем же" in lower
    assert "verdict" in SYSTEM_RU
    assert "reason_code" in SYSTEM_RU
    assert "UNCERTAIN" in SYSTEM_RU


def test_cli_accepts_http_quantized_teacher_provenance():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--pairs", "pairs.parquet",
            "--item-db", "items.sqlite",
            "--output", "labels.jsonl",
            "--model-id", "Qwen/Qwen3.5-4B",
            "--resolved-revision", "a" * 40,
            "--family", "qwen35",
            "--backend", "openai-http",
            "--quantization", "Q4_K_M",
            "--endpoint", "http://127.0.0.1:18080/v1/chat/completions",
            "--model-file", "Qwen3.5-4B-Q4_K_M.gguf",
            "--model-file-sha256", "b" * 64,
            "--prompt-sha256", "c" * 64,
        ]
    )
    assert args.backend == "openai-http"
    assert args.family == "qwen35"
    assert args.quantization == "Q4_K_M"
    assert args.resolved_revision == "a" * 40


def test_cli_accepts_seq2seq_russian_control():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--pairs", "pairs.parquet",
            "--item-db", "items.sqlite",
            "--output", "labels.jsonl",
            "--model-id", "ai-forever/FRED-T5-1.7B",
            "--resolved-revision", "d" * 40,
            "--family", "fred-t5",
            "--backend", "transformers-seq2seq",
            "--prompt-sha256", "e" * 64,
        ]
    )
    assert args.backend == "transformers-seq2seq"
    assert args.quantization == "none"
