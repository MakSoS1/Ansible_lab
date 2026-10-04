# DanceFlow Unity Client

Unity is now the presentation/game layer. The Python/FastAPI backend remains responsible for video import, pose extraction, choreography, phone tracking and scoring. Unity owns the 4K TV menu, controller input, video/3D coach presentation, HUD and future effects.

## Baseline
Use Unity 6.3 LTS. The project baseline is 6000.3.0f1; newer 6000.3 LTS patches can upgrade it.

## Run
1. Start the existing backend from remote_dance_game.
2. Open remote_dance_game/unity in Unity Hub.
3. The editor bootstrap creates Assets/Scenes/DanceFlowMain.unity automatically.
4. Press Play.

Backend URL defaults to http://127.0.0.1:8000. Override with DANCE_API_BASE.

## 3D coach migration
Import a dancer, set its Rig to Humanoid and save a prefab at:
Assets/Resources/Coaches/DefaultCoach.prefab

When that prefab exists, Unity automatically switches from source-video visuals to 3D coach mode. The source video stays hidden as the authoritative audio/timing clock. Unity loads the dance pose timeline from the backend and drives the Humanoid arms and legs. Without a prefab, video mode remains the safe fallback.

The 3D camera auto-frames the model so the full-body coach occupies a dance-stage scale rather than a portrait close-up.

## 4K
The UI reference resolution is 3840x2160. On 4K displays the video and 3D coach render textures are also 3840x2160. Song backgrounds can later be placed in Assets/Resources/Backgrounds or served by the backend without changing the gameplay/scoring architecture.

## Next retargeting pass
The current driver is a functional first Humanoid retargeter. Next improvements are torso/hips orientation, root motion, foot locking, fingers, AnimationClip baking and per-song coach selection.
