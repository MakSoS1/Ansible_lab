#!/usr/bin/env python3
"""Identify downloadable archive chunks useful for the first playable TSTO session.
No copyrighted archives are saved to GitHub artifacts.
"""
import json, os, pathlib, sys, urllib.parse, urllib.request
URL="https://archive.org/metadata/tsto-dlc-cdn-oct2024"
HEADERS={"User-Agent":"TSTO-private-restoration-ci/1.0"}
with urllib.request.urlopen(urllib.request.Request(URL,headers=HEADERS),timeout=45) as response:
    metadata=json.load(response)
files=[{"name":x["name"],"bytes":int(x.get("size",0) or 0),"sha1":x.get("sha1","")} for x in metadata.get("files",[]) if x["name"].endswith(".7z")]
candidates=[]
for x in files:
    name=x["name"].lower()
    pri=10
    for word,score in [("tutorial",0),("base",1),("languagepack",2),("4_69",3),("4_68",4),("4_0",5),("4_1_",6),("0r0",7)]:
        if word in name:pri=min(pri,score)
    if pri<10:candidates.append({**x,"priority":pri})
candidates.sort(key=lambda x:(x["priority"],x["bytes"]))
pathlib.Path("reports").mkdir(exist_ok=True)
pathlib.Path("reports/archive-inventory.json").write_text(json.dumps({"count":len(files),"total_bytes":sum(x["bytes"] for x in files),"files":files},indent=2))
print("ARCHIVES",len(files),"total bytes",sum(x["bytes"] for x in files))
print("MATCHING BASE/INITIAL/LANGUAGE CHUNKS")
for x in candidates[:35]:print(x)
sample=next((x for x in candidates if 100_000 < x["bytes"] < 15_000_000),None)
if sample is None:
    sample=next((x for x in files if 100_000 < x["bytes"] < 10_000_000),None)
if sample is None:
    print("NO_SMALL_SAMPLE");sys.exit(0)
url="https://archive.org/download/tsto-dlc-cdn-oct2024/"+urllib.parse.quote(sample["name"],safe="/")
print("SAMPLE_URL",url,flush=True)
p=pathlib.Path("sample.7z")
with urllib.request.urlopen(urllib.request.Request(url,headers=HEADERS),timeout=120) as response:
    with p.open("wb") as output:
        n=0
        while (data:=response.read(256*1024)):
            n+=len(data)
            if n>20_000_000:raise RuntimeError("Archive sample exceeded 20 MB cap")
            output.write(data)
print("SAMPLE_DOWNLOADED_BYTES",p.stat().st_size)
