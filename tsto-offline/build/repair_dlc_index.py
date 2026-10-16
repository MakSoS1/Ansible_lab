"""Build a DLCIndex.zip which references only existing historical subindexes.

A backup of the original index remains untouched. This does not invent assets:
missing events/quests remain missing and are reported for the user.
"""
from io import BytesIO
from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET

def filter_master_index(master_zip: Path, dlc_root: Path):
    """Return (bytes of new ZIP, missing names, preserved count)."""
    with zipfile.ZipFile(master_zip) as source:
        xmls=[n for n in source.namelist() if n.lower().endswith(".xml")]
        if len(xmls)!=1:raise ValueError("Expected exactly one DLC master XML")
        name=xmls[0]
        root=ET.fromstring(source.read(name))
        missing=[]
        preserved=0
        for parent in root.iter():
            for item in list(parent):
                key=item.attrib.get("index")
                if key is None:continue
                normalized=key.replace(":","/",1)
                if not normalized.startswith("dlc/") or ".." in Path(normalized).parts:
                    raise ValueError("Unsafe DLC subindex path: "+normalized)
                if not (dlc_root/normalized).is_file():
                    parent.remove(item)
                    missing.append(key)
                else:
                    preserved+=1
        patched_xml=ET.tostring(root,encoding="utf-8",xml_declaration=True)
        output=BytesIO()
        with zipfile.ZipFile(output,"w") as result:
            for member in source.infolist():
                data=patched_xml if member.filename==name else source.read(member.filename)
                result.writestr(member,data)
        return output.getvalue(),missing,preserved

def prepare_private_index(root: Path, log=print):
    master=root/"dlc"/"DLCIndex.zip"
    if not master.is_file():return None
    data,missing,preserved=filter_master_index(master,root)
    log(f"Master DLC references {preserved} installed indexes; excludes {len(missing)} unavailable indexes")
    if missing:
        log("Missing DLC index files: "+", ".join(missing[:40]))
    return BytesIO(data)

def private_master_xml(root: Path):
    """Uncompressed XML from the same filtered master ZIP served to Scorpio.

    The bare /gameassets/ route and /gameassets/dlc/DLCIndex.zip must
    reference the same available subindexes; otherwise startup can request
    subindexes that the ZIP endpoint deliberately excludes.
    """
    master = Path(root) / "dlc" / "DLCIndex.zip"
    if not master.is_file():
        return None
    packed, _, _ = filter_master_index(master, Path(root))
    with zipfile.ZipFile(BytesIO(packed)) as source:
        xmls = [n for n in source.namelist() if n.lower().endswith(".xml")]
        if len(xmls) != 1:
            raise ValueError("Expected exactly one filtered master XML")
        return source.read(xmls[0])
