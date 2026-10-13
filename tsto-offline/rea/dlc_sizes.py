#!/usr/bin/env python3
"""Download *only* public DLC manifests; estimate feasible self-contained APK size."""
import collections,io,json, urllib.request,zipfile,xml.etree.ElementTree as ET
from pathlib import Path
BASE="http://oct2018-4-35-0-uam5h44a.tstodlc.eamobile.com/netstorage/gameasset/direct/simpsons/"
ALT="https://archive.org/download/tsto-dlc-cdn-oct2024/"
def load(url):
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"tsto-size-estimator/1.0"})
        with urllib.request.urlopen(req,timeout=35) as r:return r.read(12_000_000)
    except Exception as e:
        print("DOWNLOAD_ERROR",url,type(e).__name__,str(e)[:140]);return None
def unzip_xml(content):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names=[n for n in z.namelist() if n.lower().endswith(".xml")]
        if not names:names=z.namelist()
        return z.read(names[0])
def attrs(node):
    return {x.tag.split("}")[-1]:(x.get("val") or x.get("name") or (x.text or "")) for x in node}
def scan():
    raw=load(BASE+"dlc/DLCIndex.zip")
    if not raw:
        print("NO_MASTER_INDEX: full offline DLC cannot be confirmed without a verified archive");return
    root=ET.fromstring(unzip_xml(raw))
    subindices=[i.get("index").replace(":","/") for i in root.iter() if i.tag.endswith("IndexFile") and i.get("index")]
    rows=[];errors=[]
    for name in subindices:
        data=load(BASE+name)
        if not data: errors.append(name);continue
        try:
            root=ET.fromstring(unzip_xml(data))
            for item in root.iter():
                if item.tag.split("}")[-1]!="Package":continue
                a=attrs(item)
                if "FileName" not in a:continue
                rows.append({"file":a["FileName"],"language":a.get("Language","").lower(),"tier":item.get("tier","").lower(),
                    "bytes":int(a.get("FileSize","0") or 0)})
        except Exception as e:errors.append(name+" "+str(e)[:80])
    dedup={x["file"]:x for x in rows}
    rows=list(dedup.values())
    summary={}
    for label,languages,tiers in [
      ("ru_en_all_tiers",{"","ru","en","all"},None),
      ("ru_en_100_retina_audio",{"","ru","en","all"},{"","all","100","retina","mp3","wav"}),
      ("ru_en_100_audio",{"","ru","en","all"},{"","all","100","mp3","wav"}),
      ("all_languages_all_tiers",None,None)]:
        subset=[x for x in rows if (languages is None or x["language"] in languages) and (tiers is None or x["tier"] in tiers)]
        summary[label]={"packages":len(subset),"reported_bytes":sum(x["bytes"] for x in subset)}
    out={"downloaded_indices":len(subindices)-len(errors),"total_indices":len(subindices),"index_errors":errors,
          "total_packages":len(rows),
          "languages":dict(collections.Counter(x["language"] for x in rows)),
          "tiers":dict(collections.Counter(x["tier"] for x in rows)), "size_estimates":summary}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/dlc-size-summary.json").write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=="__main__":scan()
