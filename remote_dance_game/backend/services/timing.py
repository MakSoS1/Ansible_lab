import librosa
import numpy as np
import json
from typing import Dict, Any, Optional


def extract_timing(audio_path: str) -> Dict[str, Any]:
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    else:
        tempo = float(tempo)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_ms = [int(round(t * 1000)) for t in beat_times]

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_ms = [int(round(t * 1000)) for t in onset_times]

    strong_beats = []
    if len(beat_ms) > 0 and len(onset_ms) > 0:
        onset_set = set(onset_ms)
        for bm in beat_ms:
            nearest = min(onset_set, key=lambda x: abs(x - bm), default=None)
            if nearest is not None and abs(nearest - bm) < 80:
                strong_beats.append(bm)

    duration_ms = int(len(y) / sr * 1000)

    return {
        "tempo": round(tempo, 1),
        "duration_ms": duration_ms,
        "beat_ms": beat_ms,
        "onset_ms": onset_ms,
        "strong_beat_ms": strong_beats,
    }
