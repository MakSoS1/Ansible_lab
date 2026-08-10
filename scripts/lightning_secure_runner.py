from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


BASE_URL = "https://storage.yandexcloud.net/ozon-ecup-2026/Matching"
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def studio_training_commands(
    *,
    repo_url: str,
    branch: str,
    sha: str,
    workdir: str = "ecup-v2-work",
) -> list[str]:
    """Build secret-free commands executed inside a Lightning Studio."""
    q_repo = shlex.quote(repo_url)
    q_branch = shlex.quote(branch)
    q_sha = shlex.quote(sha)
    q_workdir = shlex.quote(workdir)
    return [
        f"rm -rf {q_workdir} && git clone --depth 1 --branch {q_branch} {q_repo} {q_workdir}",
        f"cd {q_workdir} && git fetch --depth 1 origin {q_sha} && git checkout --detach {q_sha}",
        (
            f"cd {q_workdir} && python -m pip install -U "
            "'pandas>=2.2,<3' 'pyarrow>=17,<24' 'scikit-learn>=1.5,<2' "
            "'transformers>=4.46,<5' 'safetensors>=0.4' 'huggingface-hub>=0.26'"
        ),
        (
            f"cd {q_workdir} && mkdir -p data output && "
            f"for file in matches.parquet items_human.parquet matches_llm.parquet items.parquet; do "
            f"curl --fail --location --retry 5 --retry-all-errors -o data/$file {shlex.quote(BASE_URL)}/$file || exit 1; "
            "test -s data/$file || exit 1; done"
        ),
        (
            f"cd {q_workdir} && python -m ecup_matching.ml.train_reranker_v2 "
            "--human-items data/items_human.parquet "
            "--human-matches data/matches.parquet "
            "--llm-matches data/matches_llm.parquet "
            "--full-items data/items.parquet "
            "--output-dir output "
            "--base-model cointegrated/rubert-tiny2 "
            "--weak-presample-rows 500000 "
            "--weak-final-rows 300000 "
            "--transitive-cap 1000 "
            "--max-attrs 12 "
            "--max-chars 900 "
            "--max-length 256 "
            "--train-batch-size 96 "
            "--eval-batch-size 256 "
            "--gradient-accumulation 1 "
            "--epochs 1.0 "
            "--hard-epochs 0.30 "
            "--hard-negative-count 50000 "
            "--learning-rate 3e-5 "
            "--hard-learning-rate 1e-5"
        ),
        (
            f"cd {q_workdir} && "
            "rm -f output/train_examples.parquet output/validation_examples.parquet && "
            "rm -rf output/stage1 data && "
            "test -f output/metrics.json && test -f output/model/config.json && "
            "test -f output/validation_predictions.parquet"
        ),
    ]


def _decode_ciphertext_b64(data: bytes) -> bytes:
    stripped = data.strip(b" \t\r\n")
    if not stripped:
        raise ValueError("encrypted credential response is empty")
    try:
        return base64.b64decode(stripped, validate=True)
    except binascii.Error as exc:
        raise ValueError("encrypted credential response is not strict base64") from exc


def decrypt_credentials(private_key_path: Path, ciphertext_path: Path) -> dict[str, str]:
    """Decrypt an RSA-OAEP payload. Plaintext never leaves this process."""
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError as exc:  # pragma: no cover - integration runtime
        raise RuntimeError("cryptography is required for the Lightning bridge") from exc

    private_key = serialization.load_pem_private_key(private_key_path.read_bytes(), password=None)
    ciphertext = _decode_ciphertext_b64(ciphertext_path.read_bytes())
    plaintext = private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    try:
        payload = json.loads(plaintext.decode("utf-8"))
    finally:
        del plaintext
    expected = {"LIGHTNING_USER_ID", "LIGHTNING_API_KEY"}
    if set(payload) != expected:
        raise ValueError("credential payload has unexpected fields")
    for key in expected:
        value = payload[key]
        if not isinstance(value, str) or not value or len(value) > 512:
            raise ValueError(f"invalid credential field: {key}")
    return {key: payload[key] for key in expected}


def _extract_lightning_username(identity: object) -> str:
    candidates: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key in ("username", "user_name", "name"):
                candidate = value.get(key)
                if isinstance(candidate, str):
                    candidates.append(candidate.strip())
            for key in ("user", "identity", "principal", "subject"):
                if key in value:
                    visit(value[key])

    visit(identity)
    for candidate in candidates:
        if _USERNAME_RE.fullmatch(candidate):
            return candidate
    raise ValueError("Lightning authenticated identity did not contain a safe username")


def _authenticated_username() -> str:
    try:
        output = subprocess.check_output(
            ["lightning", "auth", "whoami", "--json"],
            text=True,
            stderr=subprocess.DEVNULL,
            env=dict(os.environ),
            timeout=30,
        )
        identity = json.loads(output)
    except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeError("Lightning authenticated identity lookup failed") from exc
    return _extract_lightning_username(identity)


def _teamspace_sort_key(teamspace: object, username: str) -> tuple[int, str]:
    owner = getattr(teamspace, "owner", None)
    owner_name = str(getattr(owner, "name", ""))
    name = str(getattr(teamspace, "name", ""))
    personal = owner_name == username or name == username or username in name
    return (0 if personal else 1, name)


def _resolve_studio(user, Studio, username: str, studio_name: str):
    """Prefer an existing accessible Studio; creation is a fallback only."""
    teamspaces = sorted(list(user.teamspaces), key=lambda item: _teamspace_sort_key(item, username))
    if not teamspaces:
        raise RuntimeError("authenticated Lightning user has no accessible Teamspace")

    for teamspace in teamspaces:
        try:
            studios = sorted(
                list(teamspace.studios),
                key=lambda item: str(getattr(item, "name", "")),
            )
        except Exception:
            studios = []
        if studios:
            print("Using an existing accessible Lightning Studio", flush=True)
            return studios[0], False

    last_error: Exception | None = None
    for teamspace in teamspaces:
        try:
            studio = Studio(
                name=studio_name,
                teamspace=teamspace,
                user=user,
                create_ok=True,
            )
            print("Created a dedicated Lightning Studio", flush=True)
            return studio, True
        except Exception as exc:  # pragma: no cover - cloud permissions
            last_error = exc
    raise RuntimeError(
        "no existing Lightning Studio is accessible and Studio creation is not permitted"
    ) from last_error


def _select_machine(studio, Machine) -> str:
    last_error: Exception | None = None
    for machine_name in ("L4", "A100", "T4"):
        machine = getattr(Machine, machine_name, None)
        if machine is None:
            continue
        try:
            studio.start(machine)
            return machine_name
        except Exception as start_exc:
            last_error = start_exc
            try:
                studio.switch_machine(machine)
                return machine_name
            except Exception as switch_exc:  # pragma: no cover - availability/quota
                last_error = switch_exc
                print(f"machine {machine_name} unavailable; trying fallback", flush=True)
    raise RuntimeError("no requested Lightning GPU machine could be started") from last_error


def run_lightning(
    *,
    credentials: dict[str, str],
    repo_url: str,
    branch: str,
    sha: str,
    output_dir: Path,
    studio_name: str,
) -> dict[str, Any]:
    """Run v2 GPU training remotely and download only final safe artifacts."""
    os.environ["LIGHTNING_USER_ID"] = credentials["LIGHTNING_USER_ID"]
    os.environ["LIGHTNING_API_KEY"] = credentials["LIGHTNING_API_KEY"]
    studio = None
    created_studio = False
    machine_name = None
    try:
        from lightning_sdk import Machine, Studio, User

        username = _authenticated_username()
        user = User(name=username)
        studio, created_studio = _resolve_studio(user, Studio, username, studio_name)
        machine_name = _select_machine(studio, Machine)
        print(f"Lightning Studio ready on {machine_name}", flush=True)

        commands = studio_training_commands(repo_url=repo_url, branch=branch, sha=sha)
        for number, command in enumerate(commands, start=1):
            print(f"studio command {number}/{len(commands)}", flush=True)
            output = studio.run(command)
            if output:
                print(output[-20_000:], flush=True)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        studio.download_folder("ecup-v2-work/output", target_path=str(output_dir))

        metrics_path = output_dir / "metrics.json"
        model_config = output_dir / "model" / "config.json"
        predictions = output_dir / "validation_predictions.parquet"
        for required in (metrics_path, model_config, predictions):
            if not required.is_file():
                raise RuntimeError(f"Lightning download missing artifact: {required.name}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {
            "machine": machine_name,
            "studio_name": str(getattr(studio, "name", studio_name)),
            "reused_existing_studio": not created_studio,
            "target_sha": sha,
            "selected_stage": metrics.get("selected_stage"),
            "selected_macro_average_precision": metrics.get("selected_macro_average_precision"),
        }
    finally:
        os.environ.pop("LIGHTNING_API_KEY", None)
        os.environ.pop("LIGHTNING_USER_ID", None)
        credentials.clear()
        if studio is not None:
            try:
                studio.stop()
            except Exception:
                pass
            if created_studio:
                try:
                    studio.delete()
                except Exception:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--ciphertext", type=Path, required=True)
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--studio-name", required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()

    credentials = decrypt_credentials(args.private_key, args.ciphertext)
    result = run_lightning(
        credentials=credentials,
        repo_url=args.repo_url,
        branch=args.branch,
        sha=args.sha,
        output_dir=args.output_dir,
        studio_name=args.studio_name,
    )
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
