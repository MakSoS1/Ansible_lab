from scripts.lightning_secure_runner import studio_training_commands


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
