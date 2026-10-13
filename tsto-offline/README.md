# TSTO Offline for Pixel 10 — technical preflight

**This is a working static analyzer and DLC selection planner, not a playable Android game.**

The original EA game APK, artwork, licensed content, and full DLC archives are not
included. A patched game client by itself will NOT work without a compatible
server and the required DLC. This branch does not falsely claim to build an
all-in-one APK.

## How to use

Using Python 3.12 and a locally obtained APK:

~~~bash
python tsto-offline/tsto_prepare.py --output apk-report.json audit-apk TappedOut.apk
python tsto-offline/tsto_prepare.py --output dlc-plan.json plan-dlc DLCIndex.zip --languages ru,en --tiers 100,retina,mp3,wav
cd tsto-offline && python -m unittest discover -s tests -v
~~~

The tool checks exact native-engine hash support for the existing public
patcher, ABI presence, ELF alignment, and makes a **non-destructive** list of
potentially required RU/EN packs. It does not delete, distribute or download
copyrighted DLC. Actual tier and dependency coverage still require a
real-client trace.

## Why a single game APK is not ready

The game engine is proprietary. GameServer-Reborn (GPLv3) is a Node.js process
using a native SQLite module. Embedding it in the original game APK requires an
Android-compatible Node runtime, a startup service tied to the game's lifecycle,
SQLite integration, a working localhost patch to the game's native engine, and
full game assets.

The alternative Python server tsto-springfield is not licensed for redistribution
so it cannot just be bundled into a published APK.

Modern Pixel 10 firmware can have 16-KiB native-page compatibility concerns.
The check in this repository only reads ELF headers; it does not constitute
on-device testing.

## Original sources

* https://github.com/TappedOutReborn/GameServer-Reborn
* https://github.com/TappedOutReborn/Patch-Apk
* https://github.com/TappedOutReborn/DLC-Downloader
* https://github.com/sp00nznet/tsto-springfield
* https://developer.android.com/guide/practices/page-sizes

## Definition of done for a future build

One installable APK; server boots within the same app; loopback-only
communications; the tutorial starts with radio disabled; EN/RU localization
works; save/load works over multiple restarts; no extra downloads; app shuts
down its server after exit; native libraries run on Pixel 10 with 16-KiB pages;
economy remains unmodified. None of these runtime criteria have been verified.
