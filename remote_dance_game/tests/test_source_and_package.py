from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.source_resolver import resolve_source


def test_file_source_resolves_existing_video(tmp_path: Path) -> None:
    source = tmp_path / "dance clip.mp4"
    source.write_bytes(b"not-a-real-video-but-resolution-does-not-decode")
    resolved, title = resolve_source("file", str(source), None, str(tmp_path / "source"))
    assert Path(resolved).resolve() == source.resolve()
    assert title == "dance clip"


def test_file_source_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_source("file", str(tmp_path / "missing.mp4"), None, str(tmp_path / "source"))


def test_portable_preview_paths_and_safe_import(tmp_path: Path, monkeypatch) -> None:
    import services.packager as packager

    dances = tmp_path / "dances"
    dances.mkdir()
    monkeypatch.setattr(packager, "DANCES_DIR", str(dances))

    did = "dance_portable"
    dance_dir = dances / did
    (dance_dir / "assets").mkdir(parents=True)
    (dance_dir / "audio").mkdir(parents=True)
    video = dance_dir / "assets" / "preview.mp4"
    audio = dance_dir / "audio" / "track.mp3"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")

    packager.save_dance_package(did, {
        "title": "Portable",
        "duration_ms": 1000,
        "preview": {"video_path": str(video), "audio_path": str(audio)},
    })
    raw_preview = json.loads((dance_dir / "preview.json").read_text())
    assert raw_preview["video_path"] == "assets/preview.mp4"
    assert raw_preview["audio_path"] == "audio/track.mp3"

    loaded = packager.load_dance_package(did, load_pose=False)
    assert loaded is not None
    assert Path(loaded["preview"]["video_path"]) == video.resolve()

    exported = tmp_path / "routine.dancepack"
    packager.export_dance_pack(did, str(exported))
    # Simulate importing on another machine/path.
    import_root = tmp_path / "imported_dances"
    import_root.mkdir()
    monkeypatch.setattr(packager, "DANCES_DIR", str(import_root))
    imported_id = packager.import_dance_pack(str(exported))
    loaded_import = packager.load_dance_package(imported_id, load_pose=False)
    assert loaded_import is not None
    assert Path(loaded_import["preview"]["video_path"]).exists()


def test_dancepack_rejects_zip_slip(tmp_path: Path, monkeypatch) -> None:
    import zipfile
    import services.packager as packager

    dances = tmp_path / "dances"
    dances.mkdir()
    monkeypatch.setattr(packager, "DANCES_DIR", str(dances))
    malicious = tmp_path / "malicious.dancepack"
    with zipfile.ZipFile(malicious, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"dance_id": "safe_id"}))
        zf.writestr("../escape.txt", "nope")
    with pytest.raises(ValueError, match="Unsafe path"):
        packager.import_dance_pack(str(malicious))
