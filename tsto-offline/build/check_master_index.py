#!/usr/bin/env python3
"""Audit historical DLCIndex references without publishing game data."""
import json, sys, zipfile, xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
master=Path(sys.argv[1]);folder=Path(sys.argv[2])
with zipfile.ZipFile(master) as z:
    xml=[n for n in z.namelist() if n.lower().endswith(".xml")]
    if not xml:raise SystemExit("No DLCIndex XML in archive")
    root=ET.fromstring(z.read(xml[0]))
found=[];missing=[]
for node in root.iter():
    value=node.get("index")
    if value is None:continue
    name=value.replace(":", "/")
    t=folder/name
    (found if t.exists() else missing).append({"index":value,"tag":node.tag,"attrs":dict(node.attrib)})
summary={"root":root.tag,"root_attrs":dict(root.attrib),
         "master_zip_members":xml,"index_count":len(found)+len(missing),
         "present":found[-15:],"missing":missing,
         "found_count":len(found),"missing_count":len(missing)}
Path("reports").mkdir(exist_ok=True)
Path("reports/master-index-audit.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps(summary,indent=2,ensure_ascii=False))
