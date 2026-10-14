#!/usr/bin/env python3
"""Reconstruct a private, temporary DLC tree from the public IA 2024 backup.

Downloaded EA assets MUST NOT be committed or uploaded as public Actions
artifacts. Reports contain filenames, hashes, and sizes only.
"""
import argparse,hashlib,json,os,subprocess,urllib.request,urllib.parse,shutil,time,zipfile
from pathlib import Path

ITEM="tsto-dlc-cdn-oct2024"
AGENT="TSTO-private-restore-test/1.0"
def priority(row):
    n=row["name"].lower()
    if n=="cdn/dlc.7z":return 0
    if "4_69" in n:return 1
    if "4_0_" in n or "0r0" in n:return 2
    if "language" in n or "database" in n:return 3
    return 4
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",default="_dlc_stage")
    p.add_argument("--max-download-gb",type=float,default=30.)
    p.add_argument("--min-free-gb",type=float,default=16.)
    args=p.parse_args()
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    temp=root.parent/("_temporary_tsto_7z");temp.mkdir(exist_ok=True)
    reports=Path("reports");reports.mkdir(exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(
        f"https://archive.org/metadata/{ITEM}",headers={"User-Agent":AGENT}),timeout=50) as res:
        catalog=json.load(res)
    items=[{"name":f["name"],"size":int(f.get("size",0) or 0),
             "sha1":f.get("sha1")} for f in catalog["files"] if f["name"].endswith(".7z")]
    items.sort(key=lambda f:(priority(f),f["size"]))
    planned=sum(f["size"] for f in items)
    if planned>args.max_download_gb*10**9:
        raise SystemExit(f"Download exceeds max budget: {planned} bytes")
    status={"planned_archive_count":len(items),"planned_download_bytes":planned,
            "started_at_utc":int(time.time()),"completed":[],"failed":[],
            "output_dir":str(root),"note":"No copyrighted data uploaded as artifacts."}
    def report():
        status["current_uncompressed_files"]=sum(1 for _ in root.rglob("*") if _.is_file())
        status["current_uncompressed_size_bytes"]=sum(_.stat().st_size for _ in root.rglob("*") if _.is_file())
        (reports/"full-dlc-progress.json").write_text(json.dumps(status,indent=2,ensure_ascii=False))
    for index,item in enumerate(items,1):
        name=item["name"]
        if shutil.disk_usage(".").free < args.min_free_gb*1024**3:
            status["failed"].append({"name":name,"error":"Low disk space; stop to preserve runner"})
            print("DISK_GUARD",flush=True);break
        url="https://archive.org/download/"+ITEM+"/"+urllib.parse.quote(name,safe="/")
        local=temp/"current.7z"
        try:
            local.unlink(missing_ok=True)
            for attempt in range(1,5):
                try:
                    result=subprocess.run(["curl","-fL","--retry","3","--retry-delay","4",
                        "--connect-timeout","45","--max-time","2100","-sS",
                        "-o",str(local),url],timeout=2160)
                    if result.returncode:return_code=result.returncode;raise RuntimeError(f"curl exit {return_code}")
                    if local.stat().st_size!=item["size"]:
                        raise ValueError(f"Size mismatch {local.stat().st_size} != {item['size']}")
                    if item["sha1"]:
                        with local.open("rb") as f:
                            digest=hashlib.file_digest(f,"sha1").hexdigest()
                        if digest!=item["sha1"]:raise ValueError("SHA1 mismatch")
                    break
                except Exception as e:
                    if attempt==4:raise
                    print("RETRY",name,attempt,str(e),flush=True);time.sleep(8)
            p=subprocess.run(["7z","x","-y","-aoa","-o"+str(root),str(local)],
                stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=900)
            if p.returncode:
                raise RuntimeError("7z extraction exit "+str(p.returncode)+" "+
                    p.stderr.decode(errors="replace")[-300:])
            status["completed"].append({"name":name,"bytes":item["size"]})
            print("COMPLETE",index,len(items),name,item["size"],
                  "freeGB",round(shutil.disk_usage(".").free/1024**3,1),flush=True)
        except Exception as e:
            status["failed"].append({"name":name,"error":str(e)})
            print("FAILED",index,name,str(e)[:350],flush=True)
        finally:
            local.unlink(missing_ok=True)
            if index%8==0 or index==len(items):report()
    # The September 2024 CDN backup is missing 27 master-linked indexes.
    # The earlier-history collection supplies all 151 of 151 (655 index files).
    # Add ONLY its compact index bundle; don't redownload 41.6GB of assets.
    if not status["failed"]:
        secondary="https://archive.org/download/tsto-cdn-content/cdn/dlc.7z"
        lookup="https://archive.org/metadata/tsto-cdn-content"
        with urllib.request.urlopen(urllib.request.Request(
                lookup,headers={"User-Agent":AGENT}),timeout=55) as res:
            alternate=json.load(res)
        record=next((x for x in alternate.get("files",[])
                     if x.get("name")=="cdn/dlc.7z"),None)
        if not record:raise RuntimeError("Secondary archive has no dlc.7z")
        local=temp/"complete-indexes.7z"
        result=subprocess.run(["curl","-fLsS","--retry","3",
            "--max-time","480","-o",str(local),secondary],timeout=520)
        if result.returncode:raise RuntimeError("Supplemental index download failed")
        with local.open("rb") as f:
            sha=hashlib.file_digest(f,"sha1").hexdigest()
        if sha != record.get("sha1"):
            raise RuntimeError("Supplemental index SHA-1 mismatch")
        unpack=subprocess.run(["7z","x","-y","-aoa","-o"+str(root),str(local)],
                              stdout=subprocess.DEVNULL,timeout=180)
        if unpack.returncode:raise RuntimeError("Supplemental index extraction failed")
        local.unlink()
        # Master index is a separate network file absent from the archived
        # index collection. Without this the client always gets a 404.
        master=root/"dlc"/"DLCIndex.zip"
        master.parent.mkdir(parents=True,exist_ok=True)
        master_url=("http://oct2018-4-35-0-uam5h44a.tstodlc.eamobile.com"
                    "/netstorage/gameasset/direct/simpsons/dlc/DLCIndex.zip")
        result=subprocess.run(["curl","-fLsS","--retry","3","--max-time","180",
                 "-o",str(master),master_url],timeout=200)
        if result.returncode or not zipfile.is_zipfile(master):
            raise RuntimeError("Could not fetch and verify original DLCIndex.zip")
        status["supplemental_index_count"]=sum(1 for x in (root/"dlc").glob("DLCIndex*.zip"))
        status["master_index_available"]=True
        print("MASTER_READY: supplemental indexes",
              status["supplemental_index_count"],"master",master.stat().st_size,flush=True)
        try:
            from verify_asset_coverage import check
            coverage=check(root)
            (reports/"restored-dlc-coverage.json").write_text(
                json.dumps(coverage,ensure_ascii=False,indent=2))
            status["coverage_counts"]=coverage["counts"]
            print("DLC_COVERAGE",coverage["counts"],flush=True)
        except Exception as e:
            status["coverage_error"]=str(e)
            print("DLC_COVERAGE_ERROR",str(e),flush=True)
    report()
    print("FINAL",len(status["completed"]),"of",len(items),
          "failed",len(status["failed"]),"bytes",status["current_uncompressed_size_bytes"])
    if status["failed"]:raise SystemExit(2)
if __name__=="__main__":main()
