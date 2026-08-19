from pathlib import Path

from scripts.memory_ingest import canonical_sources


def _touch(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n" if path.suffix == ".json" else "# test\n", encoding="utf-8")


def test_canonical_sources_include_machine_readable_current_and_safe_metrics(tmp_path: Path):
    for relative in (
        "AGENTS.md",
        "docs/agent-memory/PROJECT_STATE.md",
        "ecup_matching/experiments/CURRENT.json",
        "ecup_matching/experiments/v5/PLAN.md",
        "ecup_matching/experiments/v5/RESULTS.md",
        "ecup_matching/experiments/v5/SAFE_METRICS.json",
    ):
        _touch(tmp_path, relative)

    relative = {
        path.relative_to(tmp_path).as_posix()
        for path in canonical_sources(tmp_path)
    }

    assert "ecup_matching/experiments/CURRENT.json" in relative
    assert "ecup_matching/experiments/v5/SAFE_METRICS.json" in relative
    assert "ecup_matching/experiments/v5/PLAN.md" in relative
    assert "ecup_matching/experiments/v5/RESULTS.md" in relative
