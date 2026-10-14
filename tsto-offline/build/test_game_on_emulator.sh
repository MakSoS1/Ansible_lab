#!/usr/bin/env bash
set -euo pipefail
mkdir -p reports
echo "Supported ABIs:"
adb shell getprop ro.product.cpu.abilist | tee reports/emulator-abis.txt
adb shell getprop ro.dalvik.vm.native.bridge | tee reports/emulator-native-bridge.txt
adb shell getprop ro.build.version.release | tee reports/android-version.txt
adb shell getconf PAGE_SIZE | tee reports/page-size.txt
adb shell getprop ro.build.version.sdk | tee reports/api-level.txt
if ! adb install _output/tsto-experimental.apk >reports/install.txt 2>&1; then
    cat reports/install.txt
    exit 1
fi
adb logcat -c
adb shell monkey -p com.ea.game.simpsons4_row -c android.intent.category.LAUNCHER 1 >reports/launch.txt 2>&1 || true
adb forward tcp:19000 tcp:9000
ok=0
for attempt in $(seq 1 45); do
    if curl -fsS --max-time 2 http://127.0.0.1:19000/ >reports/backend-response.txt 2>/dev/null; then
        echo "Loopback backend responded inside merged TSTO game APK" | tee reports/result.txt
        ok=1
        break
    fi
    sleep 2
done
adb shell ps -A | grep -E "simpsons4_row|tsto_backend" > reports/app-processes.txt || true
adb logcat -d -v time > reports/emulator-logcat.txt
grep -E 'TSTO_OFFLINE_SERVER|FATAL EXCEPTION|AndroidRuntime|Python|python|libscorpio|UnsatisfiedLinkError|Zygote.*exited|Process.*tsto_backend' reports/emulator-logcat.txt | tail -140 >reports/relevant-logcat.txt || true
adb exec-out screencap -p >reports/screenshot.png || true
adb shell dumpsys activity activities >reports/activity.txt || true
if [ "$ok" -ne 1 ]; then
    echo "Backend failed to start or game process never reached Application.onCreate" | tee reports/result.txt
    cat reports/relevant-logcat.txt
    exit 1
fi
echo "NOT A PLAYABILITY TEST: DLC are missing; tutorial and save not tested." >>reports/result.txt
echo "Embedded server started in merged game APK; playable-offline verification still required."
