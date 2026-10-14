#!/usr/bin/env bash
# Test the *real* Scorpio client with all DLC on the host. The emulator reaches
# the host server through ADB reverse localhost port 9000. This avoids copying
# 31GB through emulated /data and isolates server network/resource bugs.
set -euo pipefail
mkdir -p reports
adb shell getprop ro.product.cpu.abilist >reports/device-abis.txt
adb shell getconf PAGE_SIZE >reports/device-pagesize.txt
adb install _output/tsto-experimental.apk >reports/install.txt 2>&1
adb reverse tcp:9000 tcp:9000
adb logcat -c
curl -fsS --max-time 10 http://127.0.0.1:9000/ >/dev/null
adb shell monkey -p com.ea.game.simpsons4_row -c android.intent.category.LAUNCHER 1 >reports/launch.txt 2>&1
sleep 25
adb shell input tap 1695 495 || true
sleep 35
adb exec-out screencap -p >reports/online-game.png || true
adb shell ps -A | grep 'simpsons4_row' > reports/online-process.txt || true
# The device cannot use Internet here; ADB reverse still provides a wired
# local link to runner-hosted DLC and is therefore not a complete offline test.
adb shell svc wifi disable || true
adb shell svc data disable || true
adb shell settings put global airplane_mode_on 1 || true
adb shell am force-stop com.ea.game.simpsons4_row
adb shell monkey -p com.ea.game.simpsons4_row -c android.intent.category.LAUNCHER 1 >reports/offline-launch.txt 2>&1
sleep 65
adb shell input tap 1695 495 || true
adb exec-out screencap -p >reports/airplane-game.png || true
adb shell ps -A | grep 'simpsons4_row' >reports/airplane-process.txt || true
adb logcat -d -v time >reports/whole-logcat.txt
grep -Ei 'scorpio|gameassets|fatal signal|sigsegv|DLCIndex|GET /mh/gameplayconfig|bad XML' \
  reports/whole-logcat.txt | tail -250 >reports/game-logs-compact.txt || true
if [ ! -s reports/airplane-process.txt ]; then
  echo "FAIL: game process not running after airplane restart" | tee reports/test-result.txt
  exit 1
fi
echo "PASS: real game process persists after airplane restart and host DLC server" | tee reports/test-result.txt
echo "CAVEAT: server still outside emulator, connected over ADB reverse; no standalone APK validation" | tee -a reports/test-result.txt
