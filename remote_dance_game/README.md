# Dance Coach — realtime dance game

A local-first dance game inspired by the interaction model of rhythm/dance games: a PC plays a dance routine, a phone tracks the player's body with its camera, and the PC awards `Perfect / Super / Good / OK / X` in realtime.

This project does **not** require streaming camera video from the phone to the PC. MediaPipe Pose runs on the phone; only 33 pose landmarks plus timestamps are sent over WebSocket. That keeps bandwidth tiny and makes LAN play responsive.

## What is implemented

- Import a local video with a real file picker.
- Import a public video URL supported by `yt-dlp` (including YouTube/TikTok where the source is accessible and you have the right to use it).
- Offline dance preparation: normalized preview video, pose extraction, beat/onset analysis, dynamic body-part weights and hold events.
- Phone pose tracking at interactive frame rates with MediaPipe in the browser or the included Android WebView wrapper.
- Desktop media clock is the single source of truth for gameplay synchronization; there is no separate drifting audio clock.
- Server derives joint angles, velocities and motion energy from the phone's landmark sequence.
- Realtime scoring with temporal tolerance, amplitude/direction comparison, body-part accuracy and stale-frame rejection.
- `Perfect / Super / Good / OK / X`, combo, hold/"YEAH", per-limb meters and final statistics.
- Beat-reactive background flashes, animated score/grade feedback, neon coach skeleton overlay and a high-contrast dancer video treatment.
- Portable `.dancepack` export/import support for generated routine metadata.
- GitHub Actions tests on Linux + Windows, media pipeline smoke test, frontend build and Android debug build.

## Architecture

```text
                         PREPARATION (once per routine)
local file / URL
      |
      v
  FastAPI backend
      |-- ffmpeg/ffprobe ---------> normalized 720p preview
      |-- MediaPipe Pose ---------> reference 33-point pose timeline
      |-- librosa ----------------> beats / onsets / tempo
      `-- choreography -----------> weights + hold events + dance package

                           GAMEPLAY (realtime)
Phone camera
   |
   | MediaPipe Pose in phone (no camera video upload)
   v
33 landmarks + capture timestamp --WebSocket--> FastAPI scoring engine
                                                ^
                                                |
PC video.currentTime --media_clock WebSocket----+
                                                |
                                                v
                               grade + score + combo + coach pose
                                                |
                                                v
                                      React/Electron HUD
```

The gameplay path is intentionally light: video decoding/audio playback stay on the PC, inference stays on the phone, and the network transports only pose coordinates. The backend rejects a pose frame when it has been stale for more than ~260 ms, so a frozen phone feed cannot keep earning points.

## Scoring

Each reference and player frame is normalized around the hip center and torso size. The engine computes:

- limb-vector direction and relative length;
- joint-angle similarity;
- motion direction and magnitude from consecutive frames;
- normalized joint-position accuracy for arms, legs and torso;
- temporal offset inside the matching window.

The combined similarity is mapped to the default thresholds:

| Grade | Similarity | Base points |
|---|---:|---:|
| Perfect | `>= 0.92` | 300 |
| Super | `>= 0.82` | 250 |
| Good | `>= 0.68` | 200 |
| OK | `>= 0.50` | 150 |
| X | `< 0.50` | 0 |

A bounded combo bonus is added on successful judgments. Scores and thresholds are configurable through the backend settings API.

## Requirements

### PC

- Python 3.11+
- Node.js 20+
- FFmpeg / ffprobe in `PATH`
- Windows, macOS or Linux for browser mode; Electron is the intended desktop shell

A discrete GPU is useful while building a routine, but gameplay itself is designed not to depend on PC-side pose inference. Your RTX 5060 Ti is more than enough for the desktop side.

### Phone

- Modern Android/iOS browser with camera support, **or** the included Android wrapper.
- Phone and PC on the same LAN for the lowest latency.
- Browser camera APIs normally require HTTPS except for localhost. If using a normal phone browser, expose the backend through an HTTPS origin and set `PUBLIC_BASE_URL`. The Android wrapper can connect directly to the LAN HTTP backend and grants camera permission inside its secure appassets page.

## Install

```bash
python -m pip install -r requirements.txt
cd desktop
npm ci
```

Make sure FFmpeg is available:

```bash
ffmpeg -version
ffprobe -version
```

## Run

Backend, from the project root:

```bash
cd backend
python main.py
```

Desktop in another terminal:

```bash
cd desktop
npm run electron:dev
```

For browser-only desktop mode:

```bash
cd desktop
npm run build
cd ../backend
python main.py
```

Then open `http://127.0.0.1:8000/app/`.

## Create a routine

1. Open **Create Dance**.
2. Choose **File** and select a video, or choose **Link** and paste a supported public URL.
3. Optionally set clip start/end, difficulty and mirroring.
4. Start the build and wait for pose/beat preprocessing to finish.
5. The routine appears in the library.

For reliable automatic conversion, use footage where one main dancer's full body is visible most of the time. The builder refuses clips with very poor pose coverage instead of silently producing unusable choreography. Hard cuts, multiple equally prominent dancers and long occlusions remain intrinsically ambiguous in arbitrary source videos and should be trimmed before import for best results.

## Play

1. Choose a routine and create a game session.
2. Scan/open the connection URL on the phone.
3. Allow camera access and place the phone so the full body is visible.
4. Complete calibration.
5. Press **Start Dance** on the PC.

The PC video clock drives scoring. The phone only streams pose landmarks; if tracking is lost or a frame becomes stale, the HUD shows `STEP BACK INTO FRAME` and awards no points until fresh tracking resumes.

## Android wrapper

The Android module is in `mobile-android/`. It embeds the same tracker UI and preserves the original `http://` or `https://` backend origin from the scanned connection URL.

Build locally with Android SDK + JDK 17:

```bash
cd mobile-android
gradle :app:assembleDebug
```

The CI workflow builds the same debug APK.

## Tests

Fast local checks:

```bash
python -m compileall -q backend tests
python -m pytest -q tests
```

Frontend type-check/build:

```bash
cd desktop
npx tsc --noEmit
npm run build
```

FFmpeg pipeline smoke test:

```bash
python tests/media_smoke.py
```

The scoring tests explicitly verify that:

- exact phone-format landmarks can reach `Perfect`;
- wrong movement amplitude scores below an exact movement;
- a temporally mismatched movement is penalized;
- a stale pose frame earns zero points and breaks combo.

## Main endpoints

- `POST /api/uploads/video` — upload a source video.
- `POST /api/dances` — start routine preprocessing.
- `GET /api/jobs/{job_id}` — build progress.
- `GET /api/dances/{dance_id}/playback` — beats/events/playback metadata.
- `POST /api/session/create` — create a game session and phone QR URL.
- `GET /api/session/{session_id}/status` — phone/calibration state.
- `WS /ws/phone/{session_id}` — phone pose stream.
- `WS /ws/game/{session_id}` — desktop media clock, controls and scores.
- `GET /api/results/{session_id}` — final stats.
- `GET /api/health` — healthcheck.

## Notes on source videos

The importer is intended for videos you own, licensed clips, or sources whose terms allow downloading/processing. Provider access can change independently of the game, so URL imports should fail cleanly while local-file import remains deterministic.
