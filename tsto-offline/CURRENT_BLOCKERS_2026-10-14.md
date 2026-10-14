# TSTO Pixel 10 offline recovery — status, 14 October 2026

This is a preservation and interoperability experiment, **not a playable or redistributable complete APK**.

## Latest full-content evidence

[Run 38580372087](https://github.com/MakSoS1/Ansible_lab/actions/runs/38580372087) downloaded and extracted 410 archived DLC sets (31,424,954,905 bytes, plus 328 archived supplemental indexes), started Android 16 (API 36) x86_64 emulator, and connected the original ARM64 4.69.5 client to a host-side Flask/protobuf local server using ADB reverse.

- Authentication, player identity, server time, gameplay config, master DLC index, event gamescripts: successful HTTP responses.
- Original client proceeded to native GL shader initialization, then crashed after 19s, in `GLThread`.
- Guest backtrace contains `glCreateShader`, then guest GLES-to-x86 translation. Host backtrace: `libOpenglCodecCommon.so!std::__tree_balance_after_insert` -> `GLSharedGroup::addShaderData` -> `libGLESv2_enc.so!GL2Encoder::s_glCreateShader`. SIGSEGV null pointer. **Do not present a lingering Android process as proof of gameplay.**
- Real screenshots show home screen, not a playable Springfield, so tutorial and save are not proven. The successful workflow status was a lax acceptance condition, subsequently tightened.
- Client also requested `DLCIndex-v4-69-5-412AA04C483EBCA5.zip` and got 404. This file is *not referenced* in the 151-index master, and the exact file's absence has been reported by TSTO private-server users since 2025; do not equate this with the shader crash without evidence.
- Master index's 151 linked subindices are all present in combined `tsto-dlc-cdn-oct2024` + `tsto-cdn-content` archives (run 38590301007). Several ZIPs have `__MACOSX/._*.xml` metadata; the DLC coverage parser was repaired.
- `run 38578190930`: unmodified client starts on API 30, 35, 36 but cannot exercise native graphics without DLC. API 33 emulator rejects installation due to no ARM translation support.

## Current focused experiments

- [38591127595](https://github.com/MakSoS1/Ansible_lab/actions/runs/38591127595): two complete 31GB DLC runs with alternate Android emulator GPU backends `software` / `swangle`; strict town load/save acceptance.
- [38592114744](https://github.com/MakSoS1/Ansible_lab/actions/runs/38592114744): tiny ARM64 JNI `glCreateShader` probe passed with `software`, `swangle`, `swiftshader` GPU modes. This tests one shader only, not the full renderer.
- [38592788610](https://github.com/MakSoS1/Ansible_lab/actions/runs/38592788610): stress 256 shader insertions/deletions to reproduce graphics encoder tree corruption.
- [38590623796](https://github.com/MakSoS1/Ansible_lab/actions/runs/38590623796): GitHub Ubuntu ARM64 runner can start but Google does not publish a Linux ARM64 Android Emulator package on that environment. Prior macOS ARM64 emulator runs did not boot.

## Unresolved blockers

1. Validate the original game's native Scorpio renderer in an actual ARM64 Android 16 runtime, ideally a Pixel 10. The repo includes `tsto-offline/build/pixel10_usb_probe.py` (ADB USB connected, server and DLC on PC, loopback over ADB reverse). It is a *diagnostic two-process architecture*, not an all-in-one APK and must not be presented as such.
2. Validate local Python backend boot in the **same APK** on a native ARM64 Android device. The x86 emulator chooses translation ABI and fails Python initialization, while the separate x86 server app works.
3. Prove tutorial can be completed, town GET/POST and save/reload succeed, no remote network required.
4. Define a subset of EN/RU required resources smaller than non-ZIP64 Android APK packaging limits. Full archived assets are over 31GB and do not fit a single conventional APK. Do not silently remove quest/event content and call it complete.
5. Avoid publishing EA copyrighted APKs, DLC, or unlicensed third-party server code in public repositories/artifacts.

Run green != playable. Final acceptance requires a screenshot of Springfield, actual town persistence across app restarts, and true offline cold start on ARM64 Android 16.


## Latest definitive validation — October 14 (evening)

- [GPU A/B full-DLC workflow 38591127595](https://github.com/MakSoS1/Ansible_lab/actions/runs/38591127595): **both** Android 16 full-DLC `software` and `swangle` jobs FAILED strict gameplay acceptance. Actual town GET 200 = 0, town POST 200 = 0, initial `mytown` placeholder remains 33 bytes. Screenshots show Android launcher, not Springfield.
- `swangle` tombstone: `SIGSEGV` in ART's `ModUnionTableReferenceCache::UpdateAndMarkReferences` on `HeapTaskDaemon`, after native ARM64 Scorpio loading, under x86_64 guest translation. It is a crash, despite HTTP 200 authentication and master DLC index response.
- `software`: game still terminates in graphics/native code; changing GPU mode did not fix it.
- [ARM64 JNI shader test 38593820427](https://github.com/MakSoS1/Ansible_lab/actions/runs/38593820427): all three emulator GPU modes passed 256 shader creations/deletions; passing this does not establish that the complete game renderer is sound.
- [Native ARM64 Android16 macOS emulator 38595139675](https://github.com/MakSoS1/Ansible_lab/actions/runs/38595139675): FAILED boot; no real ARM64 full-game execution available on public GitHub runners. `HV_UNSUPPORTED` occurs with hardware acceleration; emulation without it timed out.
- Actual restored DLC coverage report: 151/151 subindexes loaded, 19,207 unique package references, **15,239 present / 3,968 missing** across tiers and languages. For approximate RU/EN/tier candidates, **3,343 present / 790 missing** out of 4,133. This is an archive completeness problem independent of emulator rendering.
- Extracted primary archive is 31,424,954,905 bytes. One conventional APK cannot simply embed all assets; native game/Flask sidecar cannot yet support verified completely offline gameplay. There is no acceptable complete `.apk` to distribute.
- The missing dynamically requested `DLCIndex-v4-69-5-412AA04C483EBCA5.zip` has been independently reported as never existing, even on original official servers (community discussion, Feb 2025); don't misattribute ART heap corruption to that HTTP 404 without evidence.

**Current deliverable acceptance: FAIL. Do not label merged APK playable or distribute as a finished offline game.**
