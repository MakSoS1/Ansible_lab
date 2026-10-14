#!/usr/bin/env bash
set -euo pipefail
mkdir -p reports
adb shell getprop ro.build.version.release >reports/android-version.txt
adb shell getprop ro.product.cpu.abilist >reports/abi-list.txt
adb shell getprop ro.dalvik.vm.native.bridge >reports/native-bridge.txt
adb install tsto-offline/gl-probe/app/build/outputs/apk/debug/app-debug.apk >reports/install.txt 2>&1
adb logcat -c
adb shell am start -n com.tsto.glprobe/.MainActivity | tee reports/launch.txt
# First launch under SWANGLE can be substantially slower than 12 seconds.
# Poll for the actual JNI stress result rather than relying on a fixed delay.
for i in $(seq 1 35); do
    adb logcat -d -v brief >reports/logcat-poll.txt
    if grep -q 'PASS native ARM64 shader stress=256' reports/logcat-poll.txt; then
        break
    fi
    if grep -Eq 'Fatal signal .*\(SIGSEGV\)|FAIL native ARM64 shader stress' reports/logcat-poll.txt; then
        break
    fi
    sleep 2
done
adb logcat -d -v time >reports/logcat.txt
grep -E 'TSTO_GLPROBE|Fatal signal|libOpenglCodecCommon|libGLESv2_enc|SIGSEGV|AndroidRuntime' reports/logcat.txt | tail -170 >reports/graphics-summary.txt || true
if ! grep -q 'PASS native ARM64 glCreateShader=' reports/logcat.txt;then
 echo 'FAIL: ARM64 JNI glCreateShader did not return' | tee reports/result.txt; cat reports/graphics-summary.txt; exit 1
fi
if ! grep -q 'PASS Java GLES20 glCreateShader=' reports/logcat.txt;then
 echo 'FAIL: Java GLES20 glCreateShader did not return' | tee reports/result.txt; exit 1
fi
if ! grep -q 'PASS native ARM64 shader stress=256' reports/logcat.txt;then
 echo 'FAIL: ARM64 256-shader GLSharedGroup stress did not complete' | tee reports/result.txt; cat reports/graphics-summary.txt; exit 1
fi
echo 'PASS: ARM64 JNI and Java shader creation completed under native translation' | tee reports/result.txt
