#!/usr/bin/env python3
"""Inspect historical DLC master indexes vs actual Flask served names.

Intended for temporary CI workdir only. No EA package content in reports.
"""
import argparse, json, pathlib, re, zipfile,xml.etree.ElementTree as ET
from werkzeug.utils import secure_filename
def check(root):
    root=pathlib.Path(root).resolve()
    main=root/"dlc"/"DLCIndex.zip"
    with zipfile.ZipFile(main) as f:
        name=next(n for n in f.namelist() if n.endswith(".xml") and not n.startswith("__MACOSX/"))
        index=ET.fromstring(f.read(name))
    result=[]
    for n in index.iter():
        v=n.attrib.get("index")
        if not v:continue
        dirname,fn=v.replace(":","/",1).split("/",1)
        direct=root/dirname/fn
        served=root/secure_filename(dirname)/secure_filename(fn)
        result.append({"index":v,"exists_on_disk":direct.is_file(),"exists_under_flask":served.is_file(),
                       "size":direct.stat().st_size if direct.is_file() else 0})
    mismatches=[x for x in result if not x["exists_under_flask"]]
    patch=[x for x in result if "4-69-5" in x["index"] or "UNTROUBLED" in x["index"].upper()]
    return {"total_master_refs":len(result),"flask_missing_count":len(mismatches),
            "missing":mismatches[:100],"version_469":patch,
            "top_level":sorted(x.name for x in (root/"dlc").iterdir())[:20]}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root");p.add_argument("--output",default="reports/dlc-path-contract.json")
    args=p.parse_args();result=check(args.root);pathlib.Path(args.output).parent.mkdir(exist_ok=True,parents=True)
    pathlib.Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ("top_level","missing")},indent=2))
    if result["flask_missing_count"]:raise SystemExit("Flask path contract mismatches: "+str(result["flask_missing_count"]))
