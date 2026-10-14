#!/usr/bin/env python3
"""Reconstruct a private, temporary DLC tree from the public IA 2024 backup.

Downloaded EA assets MUST NOT be committed or uploaded as public Actions
artifacts. Reports contain filenames, hashes, and sizes only.
"""
import argparse,hashlib,json,os,subprocess,urllib.request,urllib.parse,shutil,time
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
    report()
    print("FINAL",len(status["completed"]),"of",len(items),
          "failed",len(status["failed"]),"bytes",status["current_uncompressed_size_bytes"])
    if status["failed"]:raise SystemExit(2)
if __name__=="__main__":main()
