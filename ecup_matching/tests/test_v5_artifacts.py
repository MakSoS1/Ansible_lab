from pathlib import Path

import pytest

from ecup_matching.ci.v5_artifacts import resolve_structured_artifact


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_resolves_macos_common_parent_layout(tmp_path: Path) -> None:
    _write(tmp_path / "out" / "model_v5_structured.joblib")
    _write(tmp_path / "legacy" / "legacy_ecup" / "__init__.py", "")
    _write(tmp_path / "legacy" / "legacy_ecup" / "ml" / "textnorm.py")

    paths = resolve_structured_artifact(tmp_path)

    assert paths.model == tmp_path / "out" / "model_v5_structured.joblib"
    assert paths.legacy_runtime == tmp_path / "legacy" / "legacy_ecup"


def test_resolves_flat_layout(tmp_path: Path) -> None:
    _write(tmp_path / "model_v5_structured.joblib")
    _write(tmp_path / "legacy_ecup" / "__init__.py", "")
    _write(tmp_path / "legacy_ecup" / "ml" / "textnorm.py")

    paths = resolve_structured_artifact(tmp_path)

    assert paths.model == tmp_path / "model_v5_structured.joblib"
    assert paths.legacy_runtime == tmp_path / "legacy_ecup"


def test_missing_layout_reports_inventory(tmp_path: Path) -> None:
    _write(tmp_path / "unexpected" / "file.txt")

    with pytest.raises(FileNotFoundError) as exc:
        resolve_structured_artifact(tmp_path)

    message = str(exc.value)
    assert "model_v5_structured.joblib" in message
    assert "unexpected/file.txt" in message
