#!/usr/bin/env python3
"""Non-mutating coverage check for archived TSTO game assets.

Inputs are the restored user-owned personal test copy on a temporary runner.
Only aggregate counts and a bounded sample of missing filenames are reported.
"""
import argparse, json, zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

def load_xml(path):
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.lower().endswith(".xml")]
        if len(names)!=1:raise ValueError(f"Expected one XML in {path.name}: {names[:10]}")
        return ET.fromstring(z.read(names[0]))

def tags(node):
    return {n.tag.rsplit("}",1)[-1]:(n.get("name") or n.get("val") or (n.text or "").strip()) for n in node}

def check(root):
    root=Path(root)
    master=load_xml(root/"dlc"/"DLCIndex.zip")
    subindices=[node.get("index") for node in master.iter() if node.get("index")]
    totals=Counter()
    missing=[]
    entries={}
    for reference in subindices:
        path=root/reference.replace(":","/",1)
        if not path.is_file():
            totals["missing_index"]+=1
            continue
        totals["loaded_index"]+=1
        index=load_xml(path)
        for node in index.iter():
            if node.tag.rsplit("}",1)[-1]!="Package":continue
            a=tags(node)
            name=a.get("FileName")
            if not name:continue
            tier=(node.get("tier") or "").lower()
            language=(a.get("Language") or "").lower()
            key=name.replace(":","/",1)
            if key in entries:continue
            entries[key]=(tier,language)
    totals["unique_packages"]=len(entries)
    for name,(tier,lang) in entries.items():
        if (root/name).is_file():
            totals["present_all_languages"]+=1
        else:
            totals["missing_all_languages"]+=1
            if len(missing)<75:missing.append({"package":name,"tier":tier,"language":lang})
        if lang not in ("","all","ru","en"):continue
        if tier not in ("","100","retina","mp3","wav"):continue
        totals["ru_en_candidate_packages"]+=1
        if (root/name).is_file():
            totals["ru_en_present"]+=1
        else:
            totals["ru_en_missing"]+=1
    return {"counts":dict(totals),"missing_sample":missing[:75],
            "note":"Candidate language/tier filters are approximate; game requests are authoritative."}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--output",default="reports/restored-dlc-coverage.json")
    a=p.parse_args()
    output=check(a.root)
    Path(a.output).parent.mkdir(exist_ok=True,parents=True)
    Path(a.output).write_text(json.dumps(output,indent=2,ensure_ascii=False))
    print(json.dumps(output["counts"],indent=2))
