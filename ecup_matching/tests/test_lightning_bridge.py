# Lightning bridge regressions are intentionally dependency-free unit tests.
from scripts.lightning_secure_runner import (
    _decode_ciphertext_b64,
    _extract_lightning_username,
    _resolve_studio,
    studio_training_commands,
)


def test_studio_training_commands_are_pinned_and_secret_free():
    commands = studio_training_commands(
        repo_url="https://github.com/MakSoS1/Ansible_lab.git",
        branch="ecup-matching-2026",
        sha="abc123def456",
        workdir="ecup-v2-work",
    )
    joined = "\n".join(commands)
    assert "abc123def456" in joined
    assert "ecup-matching-2026" in joined
    assert "cointegrated/rubert-tiny2" in joined
    assert "LIGHTNING_API_KEY" not in joined
    assert "LIGHTNING_USER_ID" not in joined
    assert "HF_TOKEN" not in joined
    assert "--weak-final-rows 300000" in joined
    assert "--max-length 256" in joined


def test_studio_training_commands_shell_quote_untrusted_strings():
    commands = studio_training_commands(
        repo_url="https://github.com/example/repo.git",
        branch="feature safe",
        sha="abc; touch /tmp/pwned",
        workdir="safe workdir",
    )
    joined = "\n".join(commands)
    assert "'feature safe'" in joined
    assert "'abc; touch /tmp/pwned'" in joined
    assert "'safe workdir'" in joined


def test_ciphertext_decode_accepts_only_outer_ascii_whitespace():
    raw = b"encrypted-bytes"
    encoded = b"ZW5jcnlwdGVkLWJ5dGVz"
    assert _decode_ciphertext_b64(encoded + b"\n") == raw
    assert _decode_ciphertext_b64(b"  " + encoded + b"\r\n") == raw
    try:
        _decode_ciphertext_b64(b"ZW5j\ncnlwdGVkLWJ5dGVz")
        assert False, "embedded whitespace must still fail strict validation"
    except ValueError:
        pass


def test_extract_lightning_username_accepts_current_and_nested_identity_shapes():
    assert _extract_lightning_username({"username": "maksim"}) == "maksim"
    assert _extract_lightning_username({"name": "maksim"}) == "maksim"
    assert _extract_lightning_username({"user": {"name": "maksim"}}) == "maksim"
    assert _extract_lightning_username({"identity": {"username": "maksim"}}) == "maksim"


def test_extract_lightning_username_rejects_missing_or_suspicious_values():
    for payload in ({}, {"role": "member"}, {"username": ""}, {"username": "a/b"}):
        try:
            _extract_lightning_username(payload)
            assert False, f"expected identity rejection for {payload!r}"
        except ValueError:
            pass


def test_resolve_studio_prefers_existing_and_marks_it_non_destructive():
    class ExistingStudio:
        name = "already-there"
    existing = ExistingStudio()
    class Teamspace:
        name = "default"
        studios = [existing]
    class User:
        teamspaces = [Teamspace()]
    class StudioFactory:
        def __init__(self, *args, **kwargs):
            raise AssertionError("existing studio must be preferred over creation")
    studio, created = _resolve_studio(User(), StudioFactory, "maksim", "new-name")
    assert studio is existing
    assert created is False


def test_resolve_studio_falls_back_across_teamspaces_when_creation_is_forbidden():
    class Teamspace:
        def __init__(self, name):
            self.name = name
            self.studios = []
    class User:
        teamspaces = [Teamspace("denied"), Teamspace("allowed")]
    class StudioFactory:
        def __init__(self, *, name, teamspace, user, create_ok):
            if teamspace.name == "denied":
                raise RuntimeError("403 forbidden")
            self.name = name
            self.teamspace = teamspace
    studio, created = _resolve_studio(User(), StudioFactory, "maksim", "gpu-run")
    assert studio.name == "gpu-run"
    assert studio.teamspace.name == "allowed"
    assert created is True
