#!/usr/bin/env bash
# REA 6.3.0 / JADX - caller supplies the original APK and engine JAR.
set -eo pipefail
if [ "$#" -ne 2 ]; then
  echo "Usage: bash analyze_with_rea.sh /absolute/path/to/original.apk /output" >&2
  exit 2
fi
APK="$(realpath "$1")"
OUT="$(realpath -m "$2")"
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
test -f "$APK" || { echo "Original APK not found" >&2; exit 3; }
test -f "$REA_JADX_MCP_JAR" || { echo "REA_JADX_MCP_JAR missing or does not exist" >&2; exit 3; }
command -v java >/dev/null
command -v node >/dev/null
command -v npx >/dev/null
mkdir -p "$OUT"
python3 "$HERE/quick_recon.py" "$APK" --output "$OUT/apk-recon.json" >/dev/null
npx --yes rea-agents@6.3.0 inspect-android-package "$APK" --json > "$OUT/rea-apk.json"
echo "Analysis saved to $OUT"
