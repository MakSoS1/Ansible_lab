from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MODEL = "mlx-community/whisper-large-v3-turbo"
COMMUNITY = "-84793390"
SOURCES = {
    "O8hy7q6JGqI": {
        "original_url": "https://youtube.com/live/O8hy7q6JGqI?feature=share",
        "title": "AI Agents Security Week 2026 | Лекция 1. AI security and vulnerability landscape",
        "parts": [(456240095, "AI Agents Security Week 2026 | Лекция 1. AI security and vulnerability landscape")],
    },
    "axJLEpDbAu4": {
        "original_url": "https://youtube.com/live/axJLEpDbAu4?feature=share",
        "title": "AI Agents Security Week 2026 | Лекция 2. AI agent security: Assessment and defense",
        "parts": [(456240096, "AI Agents Security Week 2026 | Лекция 2. AI agent security: Assessment and defense")],
    },
    "xT1OxGl4lFI": {
        "original_url": "https://youtube.com/live/xT1OxGl4lFI?feature=share",
        "title": "AI Agents Security Week 2026 | Лекция 3. AI Agent defense architecture + DevSecOps for AI",
        "parts": [
            (456240097, "AI Agents Security Week 2026 | Лекция 3.1. AI Agent defense architecture"),
            (456240098, "AI Agents Security Week 2026 | Лекция 3.2. DevSecOps for AI"),
        ],
    },
    "h_XfNaUKPQ8": {
        "original_url": "https://youtube.com/live/h_XfNaUKPQ8?feature=share",
        "title": "AI Agents Security Week 2026 | Лекция 4. Enterprise AI agent security",
        "parts": [(456240099, "AI Agents Security Week 2026 | Лекция 4. Enterprise AI agent security")],
    },
    "eq5DQ_7hc3s": {
        "original_url": "https://youtube.com/live/eq5DQ_7hc3s?feature=share",
        "title": "AI Agents Security Week 2026 | Лекция 5. Breaking the guardrails",
        "parts": [(456240100, "AI Agents Security Week 2026 | Лекция 5. Breaking the guardrails")],
    },
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_id(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()[:24]


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-6000:]}")
    return p


@dataclass(frozen=True)
class Speech:
    id: str
    video_id: str
    source_kind: str
    start_seconds: float
    end_seconds: float
    text: str
    text_sha256: str
    backend: str
    model: str
    language: str
    ordinal: int


@dataclass(frozen=True)
class Slide:
    id: str
    video_id: str
    timestamp_seconds: float
    image_path: str
    image_sha256: str
    ocr_text: str
    ocr_text_sha256: str
    ocr_engine: str
    ordinal: int


def write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")


def dhash(image: Image.Image) -> int:
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def mad(a: Image.Image, b: Image.Image) -> float:
    aa = np.asarray(a.convert("L").resize((256, 144), Image.Resampling.BILINEAR), dtype=np.float32)
    bb = np.asarray(b.convert("L").resize((256, 144), Image.Resampling.BILINEAR), dtype=np.float32)
    return float(np.abs(aa - bb).mean())


def acquire(url: str, work: Path) -> tuple[dict, Path, Path, list[Path]]:
    work.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(run(["yt-dlp", "--dump-single-json", "--no-playlist", url]).stdout)
    (work / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    captions = work / "captions"
    captions.mkdir(exist_ok=True)
    subprocess.run([
        "yt-dlp", "--write-subs", "--write-auto-subs", "--sub-langs", "ru,en,ru.*,en.*",
        "--skip-download", "--no-playlist", "-o", str(captions / "%(id)s.%(language)s.%(ext)s"), url,
    ], capture_output=True, text=True, check=False)

    audio = work / "audio"
    video = work / "video"
    audio.mkdir(exist_ok=True)
    video.mkdir(exist_ok=True)
    run(["yt-dlp", "-f", "ba/b", "-x", "--audio-format", "wav", "--audio-quality", "0", "--no-playlist", "-o", str(audio / "%(id)s.%(ext)s"), url])
    run(["yt-dlp", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b", "--merge-output-format", "mp4", "--no-playlist", "-o", str(video / "%(id)s.%(ext)s"), url])
    wavs = list(audio.glob("*.wav"))
    vids = [p for p in video.iterdir() if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
    if not wavs or not vids:
        raise RuntimeError("media acquisition produced no audio/video")
    return metadata, wavs[0], vids[0], sorted(p for p in captions.iterdir() if p.is_file())


def transcribe(video_id: str, audio: Path, offset: float, part_index: int, ordinal_offset: int) -> list[Speech]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(audio), path_or_hf_repo=MODEL, language="ru", word_timestamps=True, verbose=False
    )
    language = str(result.get("language") or "unknown")
    rows: list[Speech] = []
    for i, seg in enumerate(result.get("segments") or []):
        text = str(seg.get("text", ""))
        start = float(seg.get("start", 0.0)) + offset
        end = float(seg.get("end", seg.get("start", 0.0))) + offset
        ordinal = ordinal_offset + i
        kind = f"asr:part-{part_index:02d}"
        rows.append(Speech(
            stable_id(video_id, kind, ordinal, start, end), video_id, kind, start, end,
            text, sha256_text(text), "mlx-whisper", MODEL, language, ordinal,
        ))
    if not rows:
        raise RuntimeError("ASR returned zero segments")
    return rows


def ocr_slides(video_id: str, video: Path, out: Path, offset: float, part_index: int, ordinal_offset: int) -> list[Slide]:
    sampled = out / "sampled"
    sampled.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vf", "fps=2", "-q:v", "2", str(sampled / "frame_%08d.jpg")])
    frames = sorted(sampled.glob("frame_*.jpg"))
    selected: list[tuple[Path, float]] = []
    prev_img: Image.Image | None = None
    prev_hash: int | None = None
    for i, frame in enumerate(frames):
        ts = i * 0.5
        with Image.open(frame) as opened:
            cur = opened.convert("RGB")
            cur_hash = dhash(cur)
            changed = prev_img is None
            if prev_img is not None and prev_hash is not None:
                changed = hamming(prev_hash, cur_hash) >= 8 or mad(prev_img, cur) >= 12.0
            if changed:
                selected.append((frame, ts))
            prev_img = cur.copy()
            prev_hash = cur_hash

    images = out / "images"
    texts = out / "ocr"
    images.mkdir(parents=True, exist_ok=True)
    texts.mkdir(parents=True, exist_ok=True)
    rows: list[Slide] = []
    for local_ord, (frame, ts) in enumerate(selected):
        name = f"part-{part_index:02d}-{local_ord:06d}.jpg"
        target = images / name
        shutil.copy2(frame, target)
        p = subprocess.run(["tesseract", str(target), "stdout", "-l", "rus+eng", "--psm", "6"], capture_output=True, check=False)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.decode("utf-8", errors="replace")[-4000:])
        text = p.stdout.decode("utf-8", errors="replace")
        (texts / f"part-{part_index:02d}-{local_ord:06d}.txt").write_text(text, encoding="utf-8", newline="\n")
        ordinal = ordinal_offset + local_ord
        abs_ts = offset + ts
        rel = f"slides/images/{name}"
        rows.append(Slide(
            stable_id(video_id, "slide", part_index, ordinal, abs_ts), video_id, abs_ts, rel,
            sha256_file(target), text, sha256_text(text), "tesseract-rus+eng", ordinal,
        ))
    shutil.rmtree(sampled, ignore_errors=True)
    if not rows:
        raise RuntimeError("slide extraction returned zero records")
    return rows


def process(video_id: str, output_root: Path) -> Path:
    if video_id not in SOURCES:
        raise KeyError(video_id)
    spec = SOURCES[video_id]
    root = output_root / video_id
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    work = root / "_work"
    all_speech: list[Speech] = []
    all_slides: list[Slide] = []
    source_rows = []
    offset = 0.0

    try:
        for part_index, (vk_id, expected_title) in enumerate(spec["parts"], 1):
            url = f"https://vkvideo.ru/video{COMMUNITY}_{vk_id}"
            part_work = work / f"part-{part_index:02d}"
            metadata, audio, video, captions = acquire(url, part_work)
            actual_title = str(metadata.get("title") or "")
            if actual_title.strip() != expected_title.strip():
                raise RuntimeError(f"source title mismatch: expected={expected_title!r} actual={actual_title!r}")
            duration = float(metadata.get("duration") or 0.0)
            if duration <= 0:
                raise RuntimeError("missing source duration")

            audit = root / "sources" / f"part-{part_index:02d}"
            audit.mkdir(parents=True, exist_ok=True)
            shutil.copy2(part_work / "metadata.json", audit / "metadata.json")
            cap_target = root / "speech" / "captions" / f"part-{part_index:02d}"
            cap_target.mkdir(parents=True, exist_ok=True)
            for c in captions:
                shutil.copy2(c, cap_target / c.name)

            part_speech = transcribe(video_id, audio, offset, part_index, len(all_speech))
            all_speech.extend(part_speech)
            part_slides_dir = work / f"slides-{part_index:02d}"
            part_slides = ocr_slides(video_id, video, part_slides_dir, offset, part_index, len(all_slides))
            for row in part_slides:
                src_img = part_slides_dir / "images" / Path(row.image_path).name
                dst_img = root / row.image_path
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_img, dst_img)
                src_txt = part_slides_dir / "ocr" / (Path(row.image_path).stem + ".txt")
                dst_txt = root / "slides" / "ocr" / src_txt.name
                dst_txt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_txt, dst_txt)
            all_slides.extend(part_slides)

            source_rows.append({
                "part": part_index, "kind": "official_vk_mirror", "source_url": url,
                "vk_video_id": vk_id, "title": actual_title, "duration_seconds": duration,
                "offset_seconds": offset, "captions_files": len(captions),
                "asr_segments": len(part_speech), "slides": len(part_slides),
            })
            offset += duration

        write_jsonl(root / "speech" / "asr.jsonl", all_speech)
        (root / "speech" / "asr.txt").write_text("\n".join(r.text for r in all_speech), encoding="utf-8", newline="\n")
        (root / "speech" / "asr_meta.json").write_text(json.dumps({"backend": "mlx-whisper", "model": MODEL, "segment_count": len(all_speech)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_jsonl(root / "slides" / "slides.jsonl", all_slides)
        (root / "metadata.json").write_text(json.dumps({
            "id": video_id, "webpage_url": spec["original_url"], "title": spec["title"],
            "duration": offset, "source_policy": "official_vk_mirror_of_user_supplied_youtube_recording",
            "sources": source_rows,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (root / "source_manifest.json").write_text(json.dumps({
            "video_id": video_id, "url": spec["original_url"], "source_kind": "official_vk_mirror",
            "parts": source_rows,
            "canonical_text_policy": "verbatim; no correction, censorship, normalization, or rewriting",
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "video_id": video_id,
            "asr": {"ok": True, "segments": len(all_speech)},
            "slides": {"ok": True, "slides": len(all_slides), "sample_seconds": 0.5},
            "captions": {"files": sum(x["captions_files"] for x in source_rows)},
            "sources": source_rows, "duration_seconds": offset,
        }
        (root / "processing_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return root


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video-id", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    process(args.video_id, args.output_root)


if __name__ == "__main__":
    main()
