from scripts.lightning_job_runner import (
    _docker_job_kwargs,
    _resolve_teamspace,
    probe_command,
)


def test_docker_job_kwargs_use_image_without_studio_and_are_secret_free():
    kwargs = _docker_job_kwargs(
        name="ecup-v3-probe",
        image="python:3.11-slim",
        machine="CPU",
        command="python -c 'print(123)'",
        teamspace="personal-space",
        username="maksim",
        env={"SAFE_FLAG": "1"},
        interruptible=True,
    )

    assert kwargs["image"] == "python:3.11-slim"
    assert kwargs["teamspace"] == "personal-space"
    assert kwargs["user"] == "maksim"
    assert kwargs["interruptible"] is True
    assert "studio" not in kwargs
    joined = repr(kwargs)
    assert "LIGHTNING_API_KEY" not in joined
    assert "LIGHTNING_USER_ID" not in joined


def test_docker_job_kwargs_reject_lightning_credentials_in_remote_env():
    for key in ("LIGHTNING_API_KEY", "LIGHTNING_USER_ID"):
        try:
            _docker_job_kwargs(
                name="ecup-v3-probe",
                image="python:3.11-slim",
                machine="CPU",
                command="echo ok",
                teamspace="personal-space",
                username="maksim",
                env={key: "must-not-leave-submitter"},
                interruptible=True,
            )
        except ValueError as exc:
            assert "credential" in str(exc).lower()
        else:
            raise AssertionError("Lightning submitter credentials must never enter remote job env")


def test_resolve_teamspace_prefers_personal_teamspace():
    class Owner:
        def __init__(self, name):
            self.name = name

    class Teamspace:
        def __init__(self, name, owner):
            self.name = name
            self.owner = Owner(owner)

    class User:
        teamspaces = [
            Teamspace("shared", "someone-else"),
            Teamspace("maksim", "maksim"),
        ]

    selected = _resolve_teamspace(User(), "maksim")
    assert selected.name == "maksim"


def test_probe_command_is_small_deterministic_and_has_no_secret_names():
    command = probe_command()
    assert "python" in command
    assert "LIGHTNING_API_KEY" not in command
    assert "LIGHTNING_USER_ID" not in command
    assert "HF_TOKEN" not in command
    assert len(command) < 500
