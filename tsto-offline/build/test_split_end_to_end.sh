#!/usr/bin/env bash
# Diagnostic end-to-end run: game ARM64 via NDK translation, *separate*
# x86_64 Android server APK. This does NOT validate a single APK.
set -euo pipefail
mkdir -p reports
GAME="_output/tsto-experimental.apk"
DONOR="_donor/app/build/outputs/apk/debug/app-debug.apk"
adb shell getprop ro.build.version.release > reports/os-version.txt
adb shell getprop ro.product.cpu.abilist > reports/abis.txt
adb shell getprop ro.dalvik.vm.native.bridge > reports/native-bridge.txt
adb shell getconf PAGE_SIZE > reports/memory-page-size.txt
adb install "$DONOR" >reports/server-install.txt 2>&1
adb install "$GAME" >reports/game-install.txt 2>&1
adb logcat -c

# Startup from the actual x86-64 server app (same device network namespace).
adb shell am start -n com.tsto.offline.sidecarprobe/com.tsto.offline.ProbeActivity | tee reports/server-launch.txt
adb forward tcp:19000 tcp:9000 >/dev/null
ready=0
for i in $(seq 1 40); do
  if curl --fail --silent --max-time 3 http://127.0.0.1:19000/ >reports/endpoint-root.txt; then ready=1; break; fi
  sleep 2
done
if [ "$ready" -ne 1 ]; then
  adb logcat -d -v time > reports/logcat.txt
  echo "FAIL: independent server app did not become reachable" > reports/result.txt
  exit 1
fi
echo "PASS: Flask game server reachable on the emulator" | tee reports/result.txt

# This file comes from the archived original CDN; no DLC or APK is published.
if [ -s "_dlc_mini/DLCIndex.zip" ]; then
  adb exec-in run-as com.tsto.offline.sidecarprobe sh -c \
    'mkdir -p files/offline_tsto/dlc/dlc && cat > files/offline_tsto/dlc/dlc/DLCIndex.zip' \
    < _dlc_mini/DLCIndex.zip
  adb shell run-as com.tsto.offline.sidecarprobe \
    ls -lh files/offline_tsto/dlc/dlc/DLCIndex.zip >reports/staged-master-index.txt
fi

# Launch the actual EA game, with its known redirect and patched native engine.
adb shell monkey -p com.ea.game.simpsons4_row -c android.intent.category.LAUNCHER 1 >reports/game-launch.txt 2>&1
sleep 25
adb shell dumpsys activity activities >reports/activities.txt || true
adb shell ps -A | grep -E 'simpsons4_row|sidecarprobe' > reports/app-processes.txt || true
adb logcat -d -v time >reports/logcat.txt
grep -Ei 'tsto_offline_server|/gameassets/|/netstorage/|DLCIndex|404|simpsons4_row|fatal exception|libscorpio|python' \
   reports/logcat.txt | tail -300 >reports/relevant-logcat.txt || true
adb exec-out screencap -p >reports/game-screenshot.png || true
curl -sS -m 5 -o reports/second-response.txt -w '%{http_code}' \
   http://127.0.0.1:19000/ >reports/server-after-client-status.txt || true
if ! grep -q 'com.ea.game.simpsons4_row' reports/app-processes.txt; then
   echo "FAIL: game process absent after 25 seconds" | tee -a reports/result.txt
   exit 1
fi
if ! grep -q '^200$' reports/server-after-client-status.txt; then
   echo "FAIL: server stopped during EA client startup" | tee -a reports/result.txt
   exit 1
fi
echo "PASS: game process and independent local server coexisted" | tee -a reports/result.txt
echo "INCOMPLETE: tutorial/saves/airplane mode unavailable until required DLC packs are staged" | tee -a reports/result.txt
