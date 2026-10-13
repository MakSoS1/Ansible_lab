#!/usr/bin/env bash
set -euo pipefail
adb install _donor/app/build/outputs/apk/debug/app-debug.apk
adb logcat -c
adb shell am start -n com.tsto.offline.sidecarprobe/com.tsto.offline.ProbeActivity
adb forward tcp:19000 tcp:9000
health=0
for n in $(seq 1 50); do
    if curl -sSf --max-time 2 http://127.0.0.1:19000/ >/dev/null; then
        echo "SUCCESS: Full TSTO Flask+protobuf backend responded from Android process"
        health=1
        break
    fi
    sleep 2
done
if [ "$health" -eq 0 ]; then
    echo "FAILURE: Android-integrated Python backend did not respond"
    adb logcat -d | grep -E -A 14 'TSTO_OFFLINE_SERVER|AndroidRuntime|FATAL EXCEPTION|chaquo|python' | tail -180 || true
    exit 1
fi
adb shell dumpsys activity activities | grep -E 'topResumedActivity|ResumedActivity' || true
echo "Emulator smoke test passed. This donor contains no EA game or DLC."
