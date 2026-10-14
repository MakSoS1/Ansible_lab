#!/usr/bin/env bash
set -euo pipefail
mkdir -p reports
printf '%s\n' 'TEST_KIND=unmodified-original-client' >reports/metadata.txt
adb shell getprop ro.build.version.release >>reports/metadata.txt
adb shell getprop ro.product.cpu.abilist >>reports/metadata.txt
adb shell getprop ro.dalvik.vm.native.bridge >>reports/metadata.txt
adb shell getconf PAGE_SIZE >>reports/metadata.txt
APK="_game_source/Original Files/tsto_original.apk"
test -s "$APK"
adb install "$APK" 2>&1 | tee reports/install.txt
adb logcat -c
adb shell monkey -p com.ea.game.simpsons4_row -c android.intent.category.LAUNCHER 1 >reports/launch.txt 2>&1 || true
for n in 1 2 3 4 5; do
  sleep 9
  adb shell ps -A | grep "com.ea.game.simpsons4_row" >>reports/process-state.txt || true
done
adb logcat -d -v time >reports/logcat.txt
adb exec-out screencap -p >reports/screen.png || true
grep -Ei 'Cannot process signal 11|Fatal signal|SIGSEGV|SIGABRT|libscorpio|ScorpioOGLESRenderer|FATAL EXCEPTION|No implementation found|ndk_translation' reports/logcat.txt | tail -180 >reports/crash-summary.txt || true
grep -E 'signal [0-9]+ \(SIG' reports/logcat.txt >reports/fatal-signals.txt || true
if test -s reports/fatal-signals.txt; then
  echo "BASELINE_CRASH: unmodified EA client also crashed on this x86_64 Android emulator" | tee reports/result.txt
elif ! grep -q 'com.ea.game.simpsons4_row' reports/process-state.txt; then
  echo "BASELINE_MISSING: original EA client process not seen" | tee reports/result.txt
else
  echo "BASELINE_SURVIVED_45S: original EA client was observed without logged fatal signal" | tee reports/result.txt
fi
