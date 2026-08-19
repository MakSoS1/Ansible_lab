import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.memora_safe_server import build_safe_env
from scripts.memory_common import scan_text_for_secrets
from scripts.memory_policy import validate_repository


def test_secret_scanner_catches_high_risk_values():
    text = "token=ghp_" + "A" * 36 + " password=supersecret"
    findings = scan_text_for_secrets(text)
    assert findings
    assert any("GitHub" in finding or "Password" in finding for finding in findings)


def test_secret_scanner_ignores_documented_names_and_placeholders():
    safe_doc = (
        "GitHub secret: `HF_TOKEN`. "
        "password=<your-password>; secret=${SECRET_NAME}; secret=[REDACTED]."
    )
    assert scan_text_for_secrets(safe_doc) == []


def test_safe_env_strips_cloud_and_llm_credentials(tmp_path: Path):
    inherited = {
        "PATH": os.environ.get("PATH", ""),
        "OPENAI_API_KEY": "secret",
        "OPENROUTER_API_KEY": "secret",
        "CLOUDFLARE_API_TOKEN": "secret",
        "AWS_ACCESS_KEY_ID": "secret",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "MEMORA_STORAGE_URI": "s3://unsafe/db",
        "MEMORA_EMBEDDING_MODEL": "openai",
        "MEMORA_LLM_ENABLED": "true",
        "MEMORA_AUTO_CAPTURE": "true",
        "MEMORA_ALLOW_ANY_TAG": "0",
    }
    safe = build_safe_env(tmp_path, inherited)
    assert safe["MEMORA_EMBEDDING_MODEL"] == "tfidf"
    assert safe["MEMORA_LLM_ENABLED"] == "false"
    assert safe["MEMORA_AUTO_CAPTURE"] == "false"
    assert safe["MEMORA_ALLOW_ANY_TAG"] == "1"
    assert safe["MEMORA_DB_PATH"] == str(tmp_path / ".agent-memory" / "memories.db")
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "CLOUDFLARE_API_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "MEMORA_STORAGE_URI"):
        assert key not in safe


def _minimal_repo(root: Path, *, version: str = "v2", status: str = "completed") -> None:
    (root / "ecup_matching" / "experiments" / version).mkdir(parents=True)
    (root / "docs" / "agent-memory").mkdir(parents=True)
    (root / "AGENTS.md").write_text(f"Current {version}\n", encoding="utf-8")
    (root / "docs" / "agent-memory" / "PROJECT_STATE.md").write_text(f"Current experiment {version}\n", encoding="utf-8")
    (root / "docs" / "agent-memory" / "EXPERIMENT_INDEX.md").write_text(f"| {version} | {status} |\n", encoding="utf-8")
    (root / "docs" / "agent-memory" / "DECISIONS.md").write_text("decisions\n", encoding="utf-8")
    (root / "docs" / "agent-memory" / "SECURITY.md").write_text("security\n", encoding="utf-8")
    (root / "docs" / "agent-memory" / "ITERATION_PROTOCOL.md").write_text("protocol\n", encoding="utf-8")
    current = {"version": version, "status": status, "plan": f"ecup_matching/experiments/{version}/PLAN.md", "results": f"ecup_matching/experiments/{version}/RESULTS.md", "next_version": "v3"}
    (root / "ecup_matching" / "experiments" / "CURRENT.json").write_text(json.dumps(current), encoding="utf-8")


def test_completed_v2_requires_plan_and_results(tmp_path: Path):
    _minimal_repo(tmp_path)
    errors = validate_repository(tmp_path)
    assert any("PLAN.md" in error for error in errors)
    assert any("RESULTS.md" in error for error in errors)
    exp = tmp_path / "ecup_matching" / "experiments" / "v2"
    (exp / "PLAN.md").write_text("# v2 plan\n", encoding="utf-8")
    (exp / "RESULTS.md").write_text("# v2 results\n", encoding="utf-8")
    assert validate_repository(tmp_path) == []


def test_memory_cli_scripts_work_when_invoked_by_documented_file_path():
    root = Path(__file__).resolve().parents[2]
    scripts = ["tools/memora_hardened/install.py", "tools/memora_hardened/verify_install.py", "scripts/memory_bootstrap.py", "scripts/memory_ingest.py", "scripts/memory_checkpoint.py", "scripts/memory_policy.py"]
    for relative in scripts:
        result = subprocess.run([sys.executable, relative, "--help"], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        assert result.returncode == 0, f"{relative} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


def test_memory_workflow_checkpoints_current_iteration_dynamically():
    root = Path(__file__).resolve().parents[2]
    text = (root / ".github/workflows/ecup-memora-memory.yml").read_text(encoding="utf-8")
    assert "--iteration v1" not in text
    assert "CURRENT.json" in text
    assert 'current["version"]' in text or "current['version']" in text


def test_real_repository_current_state_is_policy_valid():
    root = Path(__file__).resolve().parents[2]
    assert validate_repository(root) == []
