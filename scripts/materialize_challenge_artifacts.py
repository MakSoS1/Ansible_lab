from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_KINDS = {"scenario", "finalist", "robustness"}


def _target_name(artifact_name: str, kind: str) -> str:
    if kind == "scenario":
        prefix = "challenge-scenario-"
        if not artifact_name.startswith(prefix):
            raise ValueError(f"unexpected scenario artifact: {artifact_name}")
        value = artifact_name[len(prefix) :]
        if not value.isdigit():
            raise ValueError(f"scenario artifact must end in numeric id: {artifact_name}")
        return str(int(value))
    if kind == "finalist":
        prefix = "challenge-finalist-"
        if not artifact_name.startswith(prefix):
            raise ValueError(f"unexpected finalist artifact: {artifact_name}")
        value = artifact_name[len(prefix) :]
        if not value:
            raise ValueError(f"empty finalist name: {artifact_name}")
        return value
    if kind == "robustness":
        prefix = "challenge-robustness-"
        if not artifact_name.startswith(prefix):
            raise ValueError(f"unexpected robustness artifact: {artifact_name}")
        value = artifact_name[len(prefix) :]
        name, separator, seed = value.rpartition("-")
        if not separator or not name or not seed.isdigit():
            raise ValueError(f"robustness artifact must end in -<seed>: {artifact_name}")
        return f"{name}__perturb_{int(seed)}"
    raise ValueError(f"unsupported artifact kind: {kind}")


def materialize_artifacts(source: Path, destination: Path, kind: str) -> list[Path]:
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {sorted(_KINDS)}")
    source = Path(source)
    destination = Path(destination)
    if not source.is_dir():
        raise FileNotFoundError(source)
    artifact_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    if not artifact_dirs:
        raise ValueError(f"no artifact directories found in {source}")
    destination.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for artifact_dir in artifact_dirs:
        target = destination / _target_name(artifact_dir.name, kind)
        if target.exists():
            raise FileExistsError(target)
        shutil.copytree(artifact_dir, target)
        created.append(target)
    return created


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--kind", choices=sorted(_KINDS), required=True)
    args = parser.parse_args()
    created = materialize_artifacts(args.source, args.destination, args.kind)
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
