from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.memora_hardened.harden import PINNED_COMMIT, PROFILE, harden_tree

UPSTREAM = "https://github.com/agentic-box/memora.git"


def _run(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, cwd=cwd, env=env, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def install(prefix: Path, *, install_dev: bool = False) -> dict[str, object]:
    prefix = Path(prefix).resolve()
    prefix.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(prefix, 0o700)

    source = prefix / "src"
    venv = prefix / "venv"
    dist = prefix / "dist"
    for path in (source, venv, dist):
        if path.exists():
            if path.is_symlink():
                raise RuntimeError(f"refusing to replace symlink: {path}")
            shutil.rmtree(path)

    _run(["git", "clone", "--filter=blob:none", "--no-checkout", UPSTREAM, str(source)])
    _run(["git", "checkout", "--detach", PINNED_COMMIT], cwd=source)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != PINNED_COMMIT:
        raise RuntimeError(f"Memora pin mismatch: expected {PINNED_COMMIT}, got {actual}")

    hardening_report = harden_tree(source)

    _run([sys.executable, "-m", "venv", str(venv)])
    python = _venv_python(venv)
    constraints = Path(__file__).resolve().with_name("constraints.txt")
    _run([str(python), "-m", "pip", "install", "--upgrade", "pip", "build"])
    _run([str(python), "-m", "pip", "install", "-c", str(constraints), str(source)])
    if install_dev:
        _run([str(python), "-m", "pip", "install", "-c", str(constraints), f"{source}[dev]"])

    dist.mkdir(parents=True, exist_ok=True)
    _run([str(python), "-m", "build", "--wheel", "--outdir", str(dist), str(source)])
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one hardened wheel, found {len(wheels)}")
    wheel = wheels[0]

    mcp_version = subprocess.check_output(
        [str(python), "-c", "import importlib.metadata as m; print(m.version('mcp'))"],
        text=True,
    ).strip()
    major = int(mcp_version.split(".", 1)[0])
    if major >= 2:
        raise RuntimeError(f"unsafe mcp major resolved: {mcp_version}")

    manifest = {
        "profile": PROFILE,
        "upstream_repository": UPSTREAM,
        "upstream_commit": PINNED_COMMIT,
        "mcp_version": mcp_version,
        "wheel": wheel.name,
        "wheel_sha256": _sha256(wheel),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardening": hardening_report,
    }
    manifest_path = prefix / "runtime-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Install pinned hardened Memora into an isolated runtime prefix")
    parser.add_argument("--prefix", type=Path, default=Path(".agent-memory/runtime"))
    parser.add_argument("--install-dev", action="store_true", help="install upstream dev dependencies for its test suite")
    args = parser.parse_args()
    manifest = install(args.prefix, install_dev=args.install_dev)
    safe = {key: value for key, value in manifest.items() if key not in {"environment"}}
    print(json.dumps(safe, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
