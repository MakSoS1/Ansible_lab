#!/usr/bin/env python3
"""Experimental private APK merger: original game + Python/Flask server donor.
Never publish copyright-protected APKs without permission.
"""
import argparse
import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

EXPECTED = {
    "8137faf7a4349043e6e0b0516249c673eb402cf96012c87d805c969cdccae3c2",
    "5b6bf2e4ee386a825e58c3e12aae923d66db8b6fbdd21df05b81b4e1edeb44ae",
}
HOST = "http://127.0.0.2:9000"
DLC = HOST + "/gameassets/"
CDN = b"http://oct2018-4-35-0-uam5h44a.tstodlc.eamobile.com/netstorage/gameasset/direct/simpsons/"
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
        "https://prod.simpsons-ea.com":HOST,
        "https://syn-dir.sn.eamobile.com":HOST,
        "https://ping1.tnt-ea.com":HOST,
    }
    cdn_regex=re.compile(r"https?://[A-Za-z0-9.-]+\.tstodlc\.eamobile\.com/netstorage/gameasset/direct/simpsons/")
    for f in game.glob("smali*/**/*.smali"):
        s=f.read_text(encoding="utf-8",errors="ignore")
        ns=s
        for old,new in substitutions.items():ns=ns.replace(old,new)
        ns=cdn_regex.sub(DLC,ns)
        if ns!=s:f.write_text(ns,encoding="utf-8");count+=1
    print("Rewrote embedded EA addresses in",count,"smali files")

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
        if len(DLC.encode("ascii"))>=len(CDN):raise RuntimeError("DLC URL too long")
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
    hook="\n    invoke-static {p0}, Lcom/tsto/offline/ServerBootstrap;->start(Landroid/content/Context;)V"
    new_fragment=fragment[:site.end()]+hook+fragment[site.end():]
    text=text[:methods[0].start()]+new_fragment+text[methods[0].end():]
    f.write_text(text,encoding="utf-8")
    manifest=game/"AndroidManifest.xml"
    xml=manifest.read_text()
    if "android:usesCleartextTraffic=" not in xml:
        xml=xml.replace("<application ","<application android:usesCleartextTraffic=\"true\" ",1)
    if "android:extractNativeLibs=" not in xml:
        xml=xml.replace("<application ","<application android:extractNativeLibs=\"true\" ",1)
    manifest.write_text(xml)
    print("Hooked",f)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--donor",type=Path,required=True)
    p.add_argument("--apktool",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix="tsto-merge-") as tmp:
        game=Path(tmp)/"game";donor=Path(tmp)/"donor"
        run("java","-jar",str(a.apktool.resolve()),"d",str(a.original.resolve()),"-f","-o",str(game))
        run("java","-jar",str(a.apktool.resolve()),"d",str(a.donor.resolve()),"-f","-o",str(donor))
        merge_dex(game,donor)
        merge_payloads(game,donor)
        patch_java(game);patch_native(game);hook_lifecycle(game)
        run("java","-jar",str(a.apktool.resolve()),"b",str(game),"-o",str(a.out.resolve()))
        print("Wrote experimental UNSIGNED APK",a.out.resolve())
if __name__=="__main__":main()
