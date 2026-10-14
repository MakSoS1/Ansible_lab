#!/usr/bin/env python3
"""Probe archived game DLC sources without downloading licensed DLC files."""
import json, urllib.request, pathlib, traceback
ITEMS=("tsto-dlc-cdn-oct2024","tsto-cdn-content")
out={"archives":[]}
for item in ITEMS:
    url=f"https://archive.org/metadata/{item}"
    row={"identifier":item,"url":url}
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"TSTO-asset-capacity-check/1.0"})
        with urllib.request.urlopen(req,timeout=30) as res:
            raw=res.read(25_000_001)
        if len(raw)>25_000_000:raise ValueError("metadata larger than 25MB")
        meta=json.loads(raw)
        assets=[x for x in meta.get("files",[]) if x.get("name")]
        files=[{"name":x["name"],"size_bytes":int(x.get("size",0) or 0),"source":x.get("source")} for x in assets]
        extensions={}
        for f in files:
            ext=pathlib.PurePosixPath(f["name"]).suffix.lower()
            extensions[ext]=extensions.get(ext,0)+1
        row.update({"success":True,"files_count":len(files),
                    "total_bytes":sum(x["size_bytes"] for x in files),
                    "top_files":sorted(files,key=lambda x:x["size_bytes"],reverse=True)[:15],
                    "extensions":extensions,
                    "dlc_indexes": [x for x in files if "DLCIndex" in x["name"]][:30]})
    except Exception as e:
        row.update({"success":False,"error":str(e)})
    out["archives"].append(row)
pathlib.Path("reports").mkdir(exist_ok=True)
pathlib.Path("reports/dlc-archive-metadata.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
for row in out["archives"]:
    print(row["identifier"],"available",row.get("success"),"files",row.get("files_count"),
          "total_bytes",row.get("total_bytes"),"error",row.get("error"))
    print("LARGEST FILES",row.get("top_files",[])[:7])
