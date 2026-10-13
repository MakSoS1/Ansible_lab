# TSTO REA native/Android research

**This is an APK static analyzer, not the game.** The real project is [morluto/rea](https://github.com/morluto/rea), not `morluto/rearea`.

## Use on an original game APK that you own or are permitted to analyze

```bash
python3 quick_recon.py /absolute/path/to/original.apk --output apk-recon.json
python3 -m unittest discover -s tests -v
```

Reports ARM64 ELF segment alignments, SHA-256, EA endpoint host references and whether the public native patcher recognizes this engine. This tool **never runs** the APK, modifies it or verifies its signature.

## REA JADX examination

Install a compatible Node.js 22.19+/24.11+/26+, a full Java 17+ JDK and provide your own `jadx-headless-mcp-0.7.1-all.jar` as documented in [REA Android analysis](https://github.com/morluto/rea/blob/main/docs/android-analysis.md). Then:

```bash
export REA_JADX_MCP_JAR="$HOME/tools/jadx-headless-mcp-0.7.1-all.jar"
bash analyze_with_rea.sh /absolute/path/to/original.apk ./analysis
```

The script uses REA version 6.3.0 and is not end-to-end validated here: the required original APK and JADX JAR were unavailable. Native `libscorpio.so` decompilation additionally requires REA Ghidra/Hopper/IDA support configured separately.

## Why this does not make an APK

* Original game engine and artwork are proprietary, not included in this repo.
* Old ARM64 Scorpio engine may fail on Pixel 10's 16 KiB page size; a manifest patch alone is insufficient.
* A local Node server plus sqlite3 requires an Android-compatible embedded runtime and device testing.
* **Do not publish the proprietary game APK or paid assets in a public GitHub repository.**
* A single offline Android installation still needs verified EN/RU DLC, launcher/server lifecycle, endpoint rewriting, original city saving and a real Pixel 10 compatibility run.

Related: https://github.com/sp00nznet/tstorecomp ; https://github.com/TappedOutReborn/GameServer-Reborn .
