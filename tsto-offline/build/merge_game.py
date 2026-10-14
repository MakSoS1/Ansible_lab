#!/usr/bin/env python3
"""Experimental private APK merger: original game + Python/Flask server donor.
Never publish copyright-protected APKs without permission.
"""
import argparse
import hashlib
import re
import struct
import shutil
import subprocess
import tempfile
from pathlib import Path

EXPECTED = {
    "8137faf7a4349043e6e0b0516249c673eb402cf96012c87d805c969cdccae3c2",
    "5b6bf2e4ee386a825e58c3e12aae923d66db8b6fbdd21df05b81b4e1edeb44ae",
}
HOST = "http://127.0.0.1:9000"
DLC = HOST + "/gameassets/"
CDN = b"http://oct2018-4-35-0-uam5h44a.tstodlc.eamobile.com/netstorage/gameasset/direct/simpsons/"
# Diagnostic experiment: retain original 89-byte CDN string length exactly.
# A host Flask route strips this path segment for the local server.
PADDED_PATH_PREFIX = "/" + "p" * (len(CDN) - len(DLC) - 1)
PADDED_DLC = HOST + PADDED_PATH_PREFIX + "/gameassets/"
assert len(PADDED_DLC.encode("ascii")) == len(CDN)
SIG_BYPASS_469 = {
    "lib/arm64-v8a/libscorpio-neon.so": (7519612,b"\x9f\x00\x00\x14"),
    "lib/arm64-v8a/libscorpio.so": (7519464,b"\x9f\x00\x00\x14"),
}

def run(*args):
    print("+", " ".join(map(str,args)),flush=True)
    subprocess.run(args,check=True)

def next_smali(root):
    numbers=[]
    for path in root.iterdir():
        if path.is_dir() and re.fullmatch(r"smali(?:_classes\d+)?",path.name):
            numbers.append(1 if path.name=="smali" else int(path.name[13:]))
    dest=root/("smali_classes"+str(max(numbers,default=1)+1))
    dest.mkdir()
    return dest

def merge_dex(game, donor):
    dest=next_smali(game)
    copied=0
    for d in donor.glob("smali*"):
        if not d.is_dir(): continue
        for f in d.rglob("*.smali"):
            relative=f.relative_to(d)
            target=dest/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists(): raise RuntimeError("Duplicate class "+str(relative))
            shutil.copyfile(f,target)
            copied+=1
    if copied<30: raise RuntimeError("Unexpectedly few Java class definitions copied")
    print("Embedded",copied,"server runtime Java classes")

def merge_payloads(game,donor):
    count=0
    for f in (donor/"assets").rglob("*"):
        if not f.is_file():continue
        t=game/"assets"/f.relative_to(donor/"assets")
        t.parent.mkdir(parents=True,exist_ok=True)
        if t.exists():raise RuntimeError("Asset conflict: "+str(t))
        shutil.copy2(f,t);count+=1
    for f in (donor/"lib"/"arm64-v8a").glob("*.so"):
        t=game/"lib"/"arm64-v8a"/f.name
        t.parent.mkdir(parents=True,exist_ok=True)
        if t.exists():
            if hashlib.sha256(t.read_bytes()).digest()!=hashlib.sha256(f.read_bytes()).digest():
                raise RuntimeError("Native library conflict: "+str(t))
            continue
        shutil.copy2(f,t);count+=1
    print("Embedded",count,"Python assets/native runtime components")

def patch_java(game):
    count=0
    substitutions = {
        "https://prod.simpsons-ea.com": HOST,
        "https://syn-dir.sn.eamobile.com": HOST,
        "https://ping1.tnt-ea.com": HOST,
        "https://www.google.com": HOST,
    }
    cdn_regex=re.compile(r"https?://[A-Za-z0-9.-]+\.tstodlc\.eamobile\.com/netstorage/gameasset/direct/simpsons/")
    extensions = {".smali", ".xml", ".txt", ".yml", ".json", ".properties"}
    for f in game.rglob("*"):
        if not f.is_file() or f.suffix not in extensions or f.stat().st_size > 20_000_000:
            continue
        original=f.read_text(encoding="utf-8", errors="ignore")
        patched=original
        for old,new in substitutions.items():
            patched=patched.replace(old,new)
        patched=cdn_regex.sub(DLC,patched)
        if patched != original:
            f.write_text(patched,encoding="utf-8")
            count+=1
            print("Patched URLs in", f.relative_to(game))
    print("Rewrote EA addresses in",count,"text/config files including XML")
    if not count:
        raise RuntimeError("No EA endpoints patched: unexpected client format")

def patch_android_loopback_connectivity(game):
    """Permit localhost gameplay when Android has no Wi-Fi/mobile transport.

    EA's original Java bridge checks *only* TYPE_WIFI / TYPE_MOBILE,
    so native ARM64 Redroid's validated TYPE_ETHERNET (9) is incorrectly
    reported as offline. More importantly, a truly offline phone also
    needs local loopback communication without an active data network.
    Leave real HTTP errors to the bundled/hosted server rather than
    pretending that absent mobile service means 127.0.0.1 is unreachable.
    """
    targets = (
        ("smali/com/bight/android/app/BGActivity.smali", "isConnectivityOn", True),
        ("smali/com/bight/android/app/BGActivity.smali", "isAirplaneModeOn", False),
        ("smali/com/ea/simpsons/NetworkStatus.smali", "HasInternetAccess", True),
        ("smali/com/ea/nimble/NetworkImpl.smali", "reachabilityCheck", True),
    )
    for name, method, result in targets:
        file = game / name
        if not file.is_file():
            raise RuntimeError(f"Expected original 4.69.5 connectivity guard missing: {name}")
        source = file.read_text(encoding="utf-8")
        pat = re.compile(r"(?ms)^\.method[^\n]*\b" + re.escape(method) +
                         r"\(\)Z[ \t]*\n.*?^\.end method")
        found = list(pat.finditer(source))
        if len(found) != 1:
            raise RuntimeError(f"Expected unique network guard {name}:{method}, got {len(found)}")
        original = found[0].group(0)
        signature = original.split("\n", 1)[0]
        literal = "0x1" if result else "0x0"
        replacement = (signature + "\n    .locals 1\n    const/4 v0, " +
                       literal + "\n    return v0\n.end method")
        file.write_text(source[:found[0].start()] + replacement +
                        source[found[0].end():], encoding="utf-8")
        print(f"Patched localhost connectivity {name}:{method} -> {result}")


# These length-immediate sites are documented by
# TappedOutReborn/Patch-Apk/patch_native.py for the exact original Scorpio SHA.
# A naive shorter URL + trailing NUL silently breaks string construction:
# the native game starts requesting /gameassets/ rather than the DLC filename.
A64_CDN_IMMEDIATES = {
    "lib/arm64-v8a/libscorpio.so": (
        0x00373230,
        [(0x012cd644,"mov_w0","alloc"),(0x012cd674,"add_x9_x0","alloc"),
         (0x012cd6b4,"add_x9_x0","string")],
    ),
    "lib/arm64-v8a/libscorpio-neon.so": (
        0x00373227,
        [(0x012cc6fc,"mov_w0","alloc"),(0x012cc72c,"add_x9_x0","alloc"),
         (0x012cc76c,"add_x0_x0","string")],
    ),
}
A64_TEMPLATES = {
    "mov_w0":(0x52800000,0xFF80001F,5,0xFFFF),
    "add_x9_x0":(0x91000009,0xFFC003FF,10,0xFFF),
    "add_x0_x0":(0x91000000,0xFFC003FF,10,0xFFF),
}

def patch_a64_url_lengths(code,name):
    expected_url_offset,instructions=A64_CDN_IMMEDIATES[name]
    position=code.find(CDN)
    if position!=expected_url_offset:
        raise RuntimeError(f"Unexpected CDN address in {name}: found {position:#x}, expected {expected_url_offset:#x}")
    for offset,kind,value_type in instructions:
        base,mask,shift,max_imm=A64_TEMPLATES[kind]
        if offset+4>len(code):raise RuntimeError("AArch64 instruction offset beyond file")
        before=struct.unpack_from("<I",code,offset)[0]
        allowed_original_bases={base}
        # The upstream patcher deliberately rewrites this neon instruction
        # from 'add x9,x0,imm' into 'add x0,x0,imm'.
        if name.endswith("libscorpio-neon.so") and kind=="add_x0_x0":
            allowed_original_bases.add(A64_TEMPLATES["add_x9_x0"][0])
        if (before & mask) not in allowed_original_bases:
            raise RuntimeError(f"Unexpected AArch64 instruction at {offset:#x}: {before:#x}; expected {kind}")
        value=len(DLC)+1 if value_type=="alloc" else len(DLC)
        if value>max_imm:raise RuntimeError("DLC URL exceeds immediate capacity")
        encoded=base | (value<<shift)
        struct.pack_into("<I",code,offset,encoded)
        print(f"Patched A64 {name}@{offset:#x}: {before:#010x} -> {encoded:#010x}; imm={value}")

def patch_native(game):
    patched=0
    for name,(offset,patch) in SIG_BYPASS_469.items():
        f=game/name
        if not f.is_file():raise RuntimeError("Native engine missing "+str(f))
        b=bytearray(f.read_bytes())
        sha=hashlib.sha256(b).hexdigest()
        if sha not in EXPECTED:raise RuntimeError("Unexpected engine sha: "+sha)
        pos=b.find(CDN)
        if pos<0:raise RuntimeError("Expected CDN URL not found in "+str(f))
        if len(DLC.encode("ascii"))>len(CDN):raise RuntimeError("DLC URL too long")
        patch_a64_url_lengths(b,name)
        b[pos:pos+len(CDN)] = DLC.encode("ascii") + b"\0"*(len(CDN)-len(DLC))
        if offset+4>len(b):raise RuntimeError("Signature offset outside engine binary")
        b[offset:offset+4]=patch
        f.write_bytes(b)
        patched+=1
    print("Updated",patched,"ARM64 engine binaries; 4 KiB ELF segments remain")

def hook_lifecycle(game):
    candidates=list(game.glob("smali*/com/ea/game/simpsons4_row/SimpsonsApplication.smali"))
    if len(candidates)!=1:raise RuntimeError("Expected one SimpsonsApplication class, got "+str(len(candidates)))
    f=candidates[0]
    text=f.read_text(encoding="utf-8")
    methods=list(re.finditer(r"(?ms)^\.method[^\n]* onCreate\(\)V\s*\n(.*?)^\.end method",text))
    if len(methods)!=1:raise RuntimeError("Could not locate unique onCreate method")
    fragment=methods[0].group(0)
    site=re.search(r"(?m)^\s*invoke-super \{p0\}, L[^;]+;->onCreate\(\)V\s*$",fragment)
    if not site:raise RuntimeError("Could not find standard Application.onCreate invoke-super")
    # Reserve one additional register without overwriting an existing register.
    # The secondary Android process needs Application context but not game SDK
    # initialization, which wastes memory and sometimes starts native Scorpio.
    local = re.search(r"(?m)^\s*\.locals\s+(\d+)\s*$", fragment)
    regs = re.search(r"(?m)^\s*\.registers\s+(\d+)\s*$", fragment)
    if local:
        old_count = int(local.group(1))
        scratch = "v" + str(old_count)
        fragment = fragment[:local.start(1)] + str(old_count + 1) + fragment[local.end(1):]
    elif regs:
        old_count = int(regs.group(1))
        scratch = "v" + str(old_count - 1)   # exactly one implicit p0 register
        fragment = fragment[:regs.start(1)] + str(old_count + 1) + fragment[regs.end(1):]
    else:
        raise RuntimeError("onCreate has no .locals/.registers declaration")
    site = re.search(r"(?m)^\s*invoke-super \{p0\}, L[^;]+;->onCreate\(\)V\s*$",fragment)
    hook = (
        "\n    invoke-static {}, Lcom/tsto/offline/ServerBootstrap;->isBackendProcess()Z"
        + "\n    move-result " + scratch
        + "\n    if-eqz " + scratch + ", :tsto_game_main_process"
        + "\n    return-void"
        + "\n    :tsto_game_main_process"
        + "\n    invoke-static {p0}, Lcom/tsto/offline/ServerBootstrap;->start(Landroid/content/Context;)V"
    )
    new_fragment=fragment[:site.end()]+hook+fragment[site.end():]
    text=text[:methods[0].start()]+new_fragment+text[methods[0].end():]
    f.write_text(text,encoding="utf-8")
    manifest=game/"AndroidManifest.xml"
    xml=manifest.read_text()
    if "android:usesCleartextTraffic=" not in xml:
        xml=xml.replace("<application ","<application android:usesCleartextTraffic=\"true\" ",1)
    if "android:extractNativeLibs=" not in xml:
        xml=xml.replace("<application ","<application android:extractNativeLibs=\"true\" ",1)
    # Android 16 auto-enables 16-KiB backcompat for legacy 4-KiB ELF pages.
    # We intentionally omit android:pageSizeCompat because the original
    # apktool 2.10 framework used for this 4.69.5 build cannot resolve that
    # Android 16-only resource attribute (AAPT2 manifest-link failure).
    # Run Python in another Linux process. Original Scorpio and Python native
    # runtimes crashed together on the translated Android 15 ARM64 emulator.
    service = '<service android:name="com.tsto.offline.ServerService" android:process=":tsto_backend" android:exported="false" android:stopWithTask="true" />'
    if 'com.tsto.offline.ServerService' not in xml:
        if "</application>" not in xml:
            raise RuntimeError("Manifest has no application closing tag")
        xml = xml.replace("</application>", "    " + service + "\n</application>", 1)
    manifest.write_text(xml)
    print("Hooked",f)

def allow_external_loopback(game):
    # Diagnostic only: same modified client but no Python service startup.
    # A separately installed x86 server serves local DLC on the emulator.
    p=game/"AndroidManifest.xml"
    xml=p.read_text()
    if 'android:usesCleartextTraffic=' not in xml:
        xml=xml.replace("<application ","<application android:usesCleartextTraffic=\"true\" ",1)
    if 'android:extractNativeLibs=' not in xml:
        xml=xml.replace("<application ","<application android:extractNativeLibs=\"true\" ",1)
    p.write_text(xml)
    print("External-server test variant: no in-game Python process is started")

def main():
    global DLC
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--donor",type=Path,help="Required only for the embedded-server build; not used in --external-server control")
    p.add_argument("--apktool",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--aapt2",type=Path,help="Native ARM64 aapt2 override for APKtool on ARM cloud runners")
    p.add_argument("--external-server",action="store_true",help="Diagnostic: skip starting embedded Python service; external localhost server required")
    p.add_argument("--retain-legacy-network-gates",action="store_true",help="A/B diagnostic: leave original Wi-Fi/mobile-only network guards intact")
    p.add_argument("--pad-native-dlc-url",action="store_true",help="A/B diagnostic: make replacement CDN string exactly as long as original")
    a=p.parse_args()
    DLC = PADDED_DLC if a.pad_native_dlc_url else HOST + "/gameassets/"
    if not a.external_server and not a.donor:
        p.error("--donor is required for the embedded-server build")
    with tempfile.TemporaryDirectory(prefix="tsto-merge-") as tmp:
        game=Path(tmp)/"game"
        run("java","-jar",str(a.apktool.resolve()),"d",str(a.original.resolve()),"-f","-o",str(game))
        if not a.external_server:
            # A/B isolation: the external-server control must contain *no*
            # Chaquopy classes, Python runtime or server native libraries.
            # Otherwise GL/ART crashes could be caused by the merge itself.
            donor=Path(tmp)/"donor"
            run("java","-jar",str(a.apktool.resolve()),"d",str(a.donor.resolve()),"-f","-o",str(donor))
            merge_dex(game,donor)
            merge_payloads(game,donor)
        patch_java(game)
        if not a.retain_legacy_network_gates:
            patch_android_loopback_connectivity(game)
        patch_native(game)
        if a.external_server:
            allow_external_loopback(game)
        else:
            hook_lifecycle(game)
        build_args=["java","-jar",str(a.apktool.resolve()),"b",str(game)]
        if a.aapt2:
            build_args.extend(["-a",str(a.aapt2.resolve())])
        build_args.extend(["-o",str(a.out.resolve())])
        run(*build_args)
        print("Wrote experimental UNSIGNED APK",a.out.resolve())
if __name__=="__main__":main()
