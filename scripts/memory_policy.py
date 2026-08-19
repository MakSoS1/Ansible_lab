from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_ALLOWED_STATUS = {"planned", "in_progress", "completed", "rejected", "blocked"}
_REQUIRED_DOCS = [
    "AGENTS.md",
    "docs/agent-memory/PROJECT_STATE.md",
    "docs/agent-memory/EXPERIMENT_INDEX.md",
    "docs/agent-memory/DECISIONS.md",
    "docs/agent-memory/SECURITY.md",
    "docs/agent-memory/ITERATION_PROTOCOL.md",
]


def _safe_relative(root: Path, raw: str, label: str, errors: list[str]) -> Path | None:
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        errors.append(f"{label} must be a safe repository-relative path: {raw}")
        return None
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        errors.append(f"{label} escapes repository root: {raw}")
        return None
    return resolved


def validate_repository(repo_root: Path) -> list[str]:
    root = Path(repo_root).resolve()
    errors: list[str] = []

    for rel in _REQUIRED_DOCS:
        if not (root / rel).is_file():
            errors.append(f"missing required memory document: {rel}")

    current_path = root / "ecup_matching" / "experiments" / "CURRENT.json"
    if not current_path.is_file():
        errors.append("missing ecup_matching/experiments/CURRENT.json")
        return sorted(errors)

    try:
        current = json.loads(current_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"invalid CURRENT.json: {exc}")
        return sorted(errors)

    version = current.get("version")
    status = current.get("status")
    if not isinstance(version, str) or not re.fullmatch(r"v\d+", version):
        errors.append(f"CURRENT.version must match vN, got {version!r}")
        return sorted(errors)
    if status not in _ALLOWED_STATUS:
        errors.append(f"CURRENT.status must be one of {sorted(_ALLOWED_STATUS)}, got {status!r}")

    plan_raw = current.get("plan")
    results_raw = current.get("results")
    plan_path = _safe_relative(root, plan_raw, "CURRENT.plan", errors) if isinstance(plan_raw, str) else None
    results_path = _safe_relative(root, results_raw, "CURRENT.results", errors) if isinstance(results_raw, str) else None
    if not isinstance(plan_raw, str):
        errors.append("CURRENT.plan must be a repository-relative string")
    if not isinstance(results_raw, str):
        errors.append("CURRENT.results must be a repository-relative string")

    number = int(version[1:])
    if number >= 2 and status in {"in_progress", "completed", "rejected", "blocked"}:
        expected_plan = root / "ecup_matching" / "experiments" / version / "PLAN.md"
        if not expected_plan.is_file():
            errors.append(f"{version} requires ecup_matching/experiments/{version}/PLAN.md")
        elif plan_path and plan_path != expected_plan.resolve():
            errors.append(f"CURRENT.plan for {version} must point to ecup_matching/experiments/{version}/PLAN.md")
    elif plan_path and not plan_path.is_file():
        errors.append(f"CURRENT.plan does not exist: {plan_raw}")

    if status in {"completed", "rejected"}:
        expected_results = root / "ecup_matching" / "experiments" / version / "RESULTS.md"
        if not expected_results.is_file():
            errors.append(f"{version} requires ecup_matching/experiments/{version}/RESULTS.md")
        if results_path and not results_path.is_file():
            errors.append(f"CURRENT.results does not exist: {results_raw}")
        if number >= 2 and results_path and results_path != expected_results.resolve():
            errors.append(f"CURRENT.results for {version} must point to ecup_matching/experiments/{version}/RESULTS.md")

    index_path = root / "docs" / "agent-memory" / "EXPERIMENT_INDEX.md"
    if index_path.is_file():
        index = index_path.read_text(encoding="utf-8")
        if f"| {version} |" not in index:
            errors.append(f"EXPERIMENT_INDEX.md has no row for {version}")

    project_state = root / "docs" / "agent-memory" / "PROJECT_STATE.md"
    if project_state.is_file() and version not in project_state.read_text(encoding="utf-8"):
        errors.append(f"PROJECT_STATE.md does not mention current iteration {version}")

    agents = root / "AGENTS.md"
    if agents.is_file() and version not in agents.read_text(encoding="utf-8"):
        errors.append(f"AGENTS.md does not mention current iteration {version}")

    return sorted(set(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate E-CUP experiment documentation/memory policy")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    errors = validate_repository(args.repo_root)
    if errors:
        for error in errors:
            print(f"POLICY ERROR: {error}")
        return 1
    print("E-CUP memory/documentation policy: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
