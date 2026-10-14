#!/usr/bin/env bash
# Multi-hour end-to-end local preservation benchmark with complete staged DLC.
# The server APK is separate solely for emulator ABI isolation; this is NOT
# validation of the requested one-APK Pixel 10 build.
set -euo pipefail
mkdir -p reports
adb shell getprop ro.build.version.release >reports/android-release.txt
adb shell getprop ro.product.cpu.abilist >reports/abis.txt
adb shell getconf PAGE_SIZE >reports/page-size.txt
adb install _donor/app/build/outputs/apk/debug/app-debug.apk >reports/server-install.txt 2>&1
adb install _output/tsto-experimental.apk >reports/client-install.txt 2>&1
adb logcat -c
adb shell am start -n com.tsto.offline.sidecarprobe/com.tsto.offline.ProbeActivity | tee reports/server-launch.txt
adb forward tcp:19000 tcp:9000 >/dev/null
up=0
for i in $(seq 1 75); do
  if curl -fsS --max-time 2 http://127.0.0.1:19000/ >/dev/null; then up=1; break; fi
  sleep 2
done
if [ "$up" -eq 0 ]; then echo "FAIL: local game server never booted" >reports/result.txt;exit 1;fi
echo "PASS: full Flask/protobuf server ready" | tee reports/result.txt
adb shell run-as com.tsto.offline.sidecarprobe mkdir -p files/offline_tsto/dlc

# Stream the *extracted* DLC files directly to the application's private
# directory, eliminating a duplicate giant tar file inside the emulator.
du -sh _dlc_stage >reports/host-dlc-size.txt
adb shell df -h /data >reports/emulator-data-before.txt
echo "Streaming historical game assets directly into emulator private storage..."
tar -cf - -C _dlc_stage . | adb exec-in run-as com.tsto.offline.sidecarprobe \
  tar -xf - -C files/offline_tsto/dlc
adb shell run-as com.tsto.offline.sidecarprobe ls -lh files/offline_tsto/dlc/dlc/DLCIndex.zip >reports/staged-master-index.txt
adb shell df -h /data >reports/emulator-data-after.txt
curl -fsS --max-time 10 http://127.0.0.1:19000/gameassets/ |
  python -c 'import xml.etree.ElementTree as E,sys; d=sys.stdin.buffer.read();print(E.fromstring(d).tag,len(d))' \
  >reports/root-xml.txt

adb shell monkey -p com.ea.game.simpsons4_row -c android.intent.category.LAUNCHER 1 >reports/game-launch.txt 2>&1
sleep 30
adb shell input tap 1695 495 || true
adb exec-out screencap -p >reports/game-online-30s.png || true
sleep 80
adb exec-out screencap -p >reports/game-online-110s.png || true
adb shell run-as com.tsto.offline.sidecarprobe ls -lha files/offline_tsto/towns > reports/saves-online.txt || true

# Simulate airplane mode, then cold-launch again without remote connectivity.
adb shell svc wifi disable || true
adb shell svc data disable || true
adb shell settings put global airplane_mode_on 1 || true
adb shell am broadcast -a android.intent.action.AIRPLANE_MODE --ez state true >reports/airplane-broadcast.txt || true
adb shell dumpsys connectivity >reports/airplane-connectivity.txt || true
adb shell am force-stop com.ea.game.simpsons4_row
adb shell monkey -p com.ea.game.simpsons4_row -c android.intent.category.LAUNCHER 1 >reports/offline-restart.txt 2>&1 || true
sleep 65
adb shell input tap 1695 495 || true
adb exec-out screencap -p >reports/game-offline-65s.png || true
adb shell ps -A | grep -E 'simpsons4_row|sidecarprobe' >reports/processes.txt || true
adb shell run-as com.tsto.offline.sidecarprobe ls -lha files/offline_tsto/towns >reports/saves-offline.txt || true
adb logcat -d -v time >reports/total-logcat.txt
grep -Ei 'tsto_offline_server|gameassets|dlcindex|mh/gameplayconfig|getanonuid|protoWholeLand|404|bad Xml|cannot connect|fatal exception|sigsegv' \
  reports/total-logcat.txt | tail -900 >reports/protocol-summary.txt || true
if grep -q 'GET /mh/gameplayconfig.*200' reports/total-logcat.txt; then
  echo "PASS: protocol reached gameplay configuration" | tee -a reports/result.txt
else
  echo "FAIL: gameplay config absent" | tee -a reports/result.txt
  exit 1
fi
if ! grep -q com.ea.game.simpsons4_row reports/processes.txt; then
  echo "FAIL: no active EA client after airplane restart" | tee -a reports/result.txt
  exit 1
fi
echo "CHECK MANUALLY: screenshots do not automatically establish tutorial completion or real town-save progress." | tee -a reports/result.txt
echo "NOT A SINGLE-APK TEST: client and native-x86 backend have separate package IDs." | tee -a reports/result.txt
