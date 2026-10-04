# DanceFlow Unity Client

Unity is now the primary TV/game presentation layer. Python/FastAPI remains responsible for import, pose extraction, choreography, phone tracking and scoring.

## Current production mode: enhanced source video

The game is deliberately **video-first** for now. 3D coach support is kept in the code for later, but it is disabled by default.

Import flow:

1. uploaded/source video is validated;
2. a 720p proxy is made only for pose extraction;
3. choreography and timing are extracted;
4. a separate high-quality gameplay master is encoded from the ORIGINAL source;
5. Unity preloads that master before the 3–2–1 countdown;
6. the video timeline is the authoritative clock for audio and realtime scoring.

For a 720p input, the default game master is 2560×1440 using Lanczos scaling plus a deliberately mild sharpening pass and CRF 17 H.264. This does not invent detail; it avoids throwing away source detail and gives the 4K TV/Unity scaler a much cleaner input than the original low-resolution stream.

Override output size with:

`DANCE_GAME_TARGET_HEIGHT=2160`

Optional external AI super-resolution can be connected later with:

`DANCE_AI_UPSCALE_CMD='your-tool --input {input} --output {output}'`

If that tool is missing or fails, import automatically falls back to the normal high-quality master.

## Unity baseline

Use Unity 6.3 LTS. The project baseline is `6000.3.0f1`.

The UI reference resolution is 3840×2160 and the app targets 60 FPS. Keyboard and gamepad input use Unity Input System.

## Run

1. Start the backend from `remote_dance_game`.
2. Open `remote_dance_game/unity` in Unity Hub.
3. The editor bootstrap creates `Assets/Scenes/DanceFlowMain.unity`.
4. Press Play.

The backend URL defaults to `http://127.0.0.1:8000`. Override with `DANCE_API_BASE`.

## 3D is opt-in, not current production

Future 3D mode is enabled only when `DANCE_ENABLE_3D=1` (or PlayerPrefs `dance_enable_3d=1`) **and** a Humanoid prefab exists.

Until then, Unity always displays the enhanced game master. This prevents unfinished 3D assets from changing production behaviour.

## Stability choices

- video calls `Prepare()` before countdown;
- `skipOnDrop = false` so choreography frames are not silently discarded;
- media clock sent to scoring at ~20 Hz from Unity VideoPlayer time;
- video is already H.264/yuv420p with faststart for broad desktop hardware support;
- no frame interpolation or heavy denoise in preprocessing;
- UI remains edge-bound around the dancer.
