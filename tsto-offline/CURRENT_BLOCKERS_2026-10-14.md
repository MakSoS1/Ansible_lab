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

## Native ARM64 cloud-only test progress (no Pixel/USB needed)

The user cannot connect a physical Pixel. A cloud-only alternative was implemented
in `.github/workflows/tsto-native-arm64-container-probe.yml`, with GitHub-hosted
native `ubuntu-26.04-arm` and separate Redroid Android 15/16 images. All
original game APK bytes are kept on ephemeral runners and deliberately omitted
from uploaded artifacts.

- Initial Redroid errors traced to missing ADB, host binderfs nodes passed via
  Docker `--device` from the wrong binderfs instance, and root-only access to
  binder device nodes. Linux binderfs instances are independent. Letting
  Redroid create its own private BinderFS devices **fixed** the system-service
  crashes (`servicemanager` now runs; no ARM instruction translation).
- **Android 15 native ARM64 boot verified** in run
  [38617037083](https://github.com/MakSoS1/Ansible_lab/actions/runs/38617037083).
  `sys.boot_completed=1`, `ro.product.cpu.abilist=arm64-v8a`,
  no native bridge, 4-KiB page size. This is native ARM64 cloud infrastructure
  and not a Pixel 10 or 16-KiB page test.
- Android 16 reaches working Binder, ADB and package infrastructure but
  `SurfaceFlinger` crashes while priming a SkiaRenderEngine shader cache,
  reporting `output buffer not gpu writeable` under software GPU, preventing
  completed boot. This matches the Redroid issue #934 reported when ashmem is
  unavailable on new Linux kernels. `androidboot.use_memfd=1` was added and
  Android 15/16 compared; full Android 16 boot is still unverified.
- Native ARM64 game APK **installation** succeeded in Redroid Android 15
  (run [38617135606](https://github.com/MakSoS1/Ansible_lab/actions/runs/38617135606),
  reports/original-install.txt). Initial launcher automation returned
  `monkey` exit 251 before capturing full logs; the test now tolerates
  monkey failure, also tries explicit `am start`, and captures screenshot/logcat.
  This is **NOT** proof of gameplay.
- The merged-APK x86/ARM-translated Android 15 test
  [38613958873](https://github.com/MakSoS1/Ansible_lab/actions/runs/38613958873)
  started `:tsto_backend` and loaded libpython3.13.so, but the process
  exited during `Python.start` without responding on localhost.
- `merge_game.py --external-server` previously **still merged** the Python
  donor DEX and runtime. Fixed: the external control now omits the donor
  entirely. Added `tests/test_merge_modes.py`; preflight passing.

**Do not label any CI APK, source artifact, or alive process a playable single
offline APK.** Real native ARM64 gameplay, tutorial, persistent town save and
asset-complete offline install remain unverified. The complete preserved asset
archive is approximately 31.4 GB and the RU/EN subset still has missing
referenced packages. See prior blockers above.

## One-APK native ARM64 smoke test — build succeeds, gameplay NOT validated

- [GitHub Actions run 38618502598](https://github.com/MakSoS1/Ansible_lab/actions/runs/38618502598) completed successfully on native Linux ARM64 + Redroid Android **15**, with single signed experimental APK (~88 MiB), native Scorpio loaded, embedded Python/Flask backend available over localhost, and EA authentication/director endpoints returning 200.
- This is an **integration smoke test**, not a fully working single-APK game: the APK contains no historical 31.4GB DLC. Its screenshot is the Android immersive-mode hint over a blank game area, **not** Springfield, and neither tutorial nor town GET/POST/save were observed in the short test.
- [Full-DLC native ARM64 acceptance run 38628132176](https://github.com/MakSoS1/Ansible_lab/actions/runs/38628132176) added to distinguish rendering/protocol issues from asset loss, with ~31GB DLC staged **on runner host**. It uses ADB reverse, so even a successful result is NOT evidence that all assets fit in one APK, or that Android 16/Pixel 10 is supported.
- Fixed the host game-server's `/gameassets/` root route to return uncompressed XML, and aligned first-run tutorial behavior for new towns.
- Missing archives: 790 of 4,133 candidate RU/EN package references missing (archived reconstruction), independently of app/server integration. Full historical game content must NOT be advertised as complete.
- No copyrighted original game binary or DLC is published in this public repository or GitHub Actions artifacts. Full signed APK is not distributed; the existing `88 MiB` file is a **non-playable diagnostic build**.

**Acceptance still FAIL:** standalone offline Pixel 10/Android 16 cold start, complete asset set, visible Springfield, tutorial completion and persistent saves have not all been demonstrated.


## Fully archived ARM64 test — completed failure diagnosis (Oct 14, latest)

The strict 31.4GB archive run [38628132176](https://github.com/MakSoS1/Ansible_lab/actions/runs/38628132176) **FAILED**.
It restored 410/410 7z historical archive parts and served `/gameassets/`
as XML (HTTP 200). ARM64 Android 15 booted, patched EA client 4.69.5
installed and started, loaded `libscorpio.so`, and local server returned
200 for director, device, anonymous login, token info and tracking.

Nevertheless, **zero `/mh/gameplayconfig` calls, zero town GET/POST**
were observed; saved `mytown` remained 33 bytes. At 45/110 seconds the
screenshot was obscured by Android's "Viewing full screen" modal;
a SETTINGS/RETRY network-style dialog was partially visible underneath.
After restart, the actual game process (PID 2219) crashed in
`RenderThread`: Scudo `corrupted chunk header` in
`/vendor/lib64/hw/vulkan.pastel.so`, tracing through LLVM AArch64
`TargetSubtargetInfo` destructor, SwiftShader JIT, and Vulkan draw.
That is a **guest software-renderer** failure, not proof of a Pixel GPU
crash. Google Play services and Facebook warnings are present, but no
direct causation has been shown. After restart, the captured screen was
the Android launcher, not Springfield.

Commit `7daf36c4` updated the full-archive workflow to suppress Android's
immersive-mode modal **before launch**, capture an 18s failure image and
UI dump, press the observed RETRY control, and capture post-retry state.
Fresh run [38636313925](https://github.com/MakSoS1/Ansible_lab/actions/runs/38636313925)
started; it must be inspected before claiming that gameplay works.

For reference, [official Redroid configuration](https://github.com/remote-android/redroid-doc)
identifies `androidboot.redroid_gpu_mode=guest` as software rendering,
while `host` requires compatible GPU access unavailable on these
GitHub ARM runners. **No complete playable offline Pixel 10 APK exists in
our artifacts.** Continue to require real town rendering, tutorial
completion, saved-city persistence and offline cold-start validation.


## October 15 — solved one concrete network gate (not yet gameplay)

- ARM64 Android 15 Redroid's network was Ethernet TYPE_ETHERNET=9 and
  `VALIDATED` with local HTTP 200. Original app nonetheless showed
  **"Cannot Connect to the Internet. This one's your fault, not ours.
  Check your settings and retry."** before `/mh/gameplayconfig`.
  The screenshot was captured without the system immersive-mode overlay
  in [fast A/B run 38637497043](https://github.com/MakSoS1/Ansible_lab/actions/runs/38637497043).
  Changing Android's captive portal mode and HWUI renderer settings did
  NOT make the error disappear.
- Decoded the **original** APK in a CI ephemeral runner, scanned 9,840
  Java smali classes, and found the old guard in
  `com/ea/simpsons/NetworkStatus.smali`,
  `com/bight/android/app/BGActivity.smali` and
  `com/ea/nimble/NetworkImpl.smali`. In particular,
  `BGActivity.isConnectivityOn()` checks **only** cellular and Wi-Fi,
  not the validated Ethernet transport. True localhost offline mode
  also needs these outdated network gates bypassed.
- Patched 4 zero-argument boolean network methods *only in experimental
  local-loopback APKs* via `merge_game.py`, preserving an opt-out
  `--retain-legacy-network-gates` for comparison. Dedicated unit tests
  added in `tests/test_merge_modes.py`; latest preflight run
  [38638748599](https://github.com/MakSoS1/Ansible_lab/actions/runs/38638748599)
  PASSED.
- [Patched early-protocol A/B run 38638648697](https://github.com/MakSoS1/Ansible_lab/actions/runs/38638648697)
  now reached **2x `GET /mh/gameplayconfig 200`, 2x
  `GET /gameassets/dlc/DLCIndex.zip 200`**; before the patch
  both were 0. The quick probe used deliberately **synthetic empty DLC**
  and showed the launcher after the client exited, so this is strictly
  **authentication/network progress and NOT a graphics/gameplay PASS**.
- Updated ARM64 integrated single-APK build
  [38638649782](https://github.com/MakSoS1/Ansible_lab/actions/runs/38638649782)
  SUCCESS (build + embedded backend smoke test, still no bundled real DLC).
- Strict real-archive reruns
  [38638648732](https://github.com/MakSoS1/Ansible_lab/actions/runs/38638648732)
  and [38636313925](https://github.com/MakSoS1/Ansible_lab/actions/runs/38636313925)
  were still active when status was recorded. Check their town acceptance
  and actual screenshots; older runner 38636313925 predates the Java patch.
- Real offline Android 16 / Pixel 10 gameplay, a complete archived asset set,
  tutorial completion and durable save/reload **remain unverified**. The
  790 known missing RU/EN DLC refs were not recovered by this network fix.
  There is still no verified complete distributable APK.
