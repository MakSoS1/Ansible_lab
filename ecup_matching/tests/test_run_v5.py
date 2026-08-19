from pathlib import Path

from ecup_matching.submission.run_v5 import submission_root


def test_submission_root_is_directory_containing_packaged_run_py() -> None:
    assert submission_root(Path("/submission/run.py")) == Path("/submission")
