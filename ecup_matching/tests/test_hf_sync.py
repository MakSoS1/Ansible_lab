from pathlib import Path

import pytest

from ecup_matching.hf_sync import SourceFile, download_file, mirror_files


REPO_ID = "Maksim123321/e-cup-2026-matching-private"


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size: int):
        assert chunk_size > 0
        if self.payload:
            yield self.payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeSession:
    def __init__(self, responses: dict[str, bytes]):
        self.responses = responses
        self.calls = []

    def get(self, url: str, *, stream: bool, timeout):
        self.calls.append({"url": url, "stream": stream, "timeout": timeout})
        return FakeResponse(self.responses[url])


class FakeApi:
    def __init__(self):
        self.create_repo_calls = []
        self.upload_calls = []
        self.files = []

    def create_repo(self, **kwargs):
        self.create_repo_calls.append(kwargs)
        return {"url": "https://huggingface.co/datasets/test/private"}

    def upload_file(self, **kwargs):
        path = Path(kwargs["path_or_fileobj"])
        assert path.exists()
        self.upload_calls.append(kwargs)
        self.files.append(kwargs["path_in_repo"])
        return {"path": kwargs["path_in_repo"]}

    def list_repo_files(self, **kwargs):
        return list(self.files)


def test_mirror_requires_token(tmp_path):
    with pytest.raises(ValueError, match="HF_TOKEN"):
        mirror_files(REPO_ID, "", tmp_path, sources=())


def test_download_file_streams_payload_and_returns_size(tmp_path):
    source = SourceFile("a.parquet", "https://example.test/a")
    session = FakeSession({source.url: b"abcdef"})
    destination = tmp_path / source.name

    size = download_file(source, destination, session)

    assert size == 6
    assert destination.read_bytes() == b"abcdef"
    assert session.calls == [
        {"url": source.url, "stream": True, "timeout": (15, 180)}
    ]


def test_zero_byte_download_is_rejected(tmp_path):
    source = SourceFile("a.parquet", "https://example.test/a")
    session = FakeSession({source.url: b""})

    with pytest.raises(RuntimeError, match="zero bytes"):
        download_file(source, tmp_path / source.name, session)


def test_mirror_creates_private_dataset_verifies_upload_and_cleans_local_file(tmp_path):
    source = SourceFile("a.parquet", "https://example.test/a")
    api = FakeApi()
    session = FakeSession({source.url: b"abc"})

    files = mirror_files(
        REPO_ID,
        "secret",
        tmp_path,
        sources=(source,),
        api=api,
        session=session,
    )

    assert files == ["a.parquet"]
    assert api.create_repo_calls == [
        {
            "repo_id": REPO_ID,
            "repo_type": "dataset",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert api.upload_calls[0]["repo_id"] == REPO_ID
    assert api.upload_calls[0]["repo_type"] == "dataset"
    assert api.upload_calls[0]["path_in_repo"] == "a.parquet"
    assert not (tmp_path / "a.parquet").exists()


def test_mirror_rejects_unverified_upload_and_still_cleans_local_file(tmp_path):
    class MissingFileApi(FakeApi):
        def list_repo_files(self, **kwargs):
            return []

    source = SourceFile("a.parquet", "https://example.test/a")
    api = MissingFileApi()
    session = FakeSession({source.url: b"abc"})

    with pytest.raises(RuntimeError, match="not visible"):
        mirror_files(
            REPO_ID,
            "secret",
            tmp_path,
            sources=(source,),
            api=api,
            session=session,
        )

    assert not (tmp_path / "a.parquet").exists()
