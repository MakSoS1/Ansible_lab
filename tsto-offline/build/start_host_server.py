#!/usr/bin/env python3
"""Host-only Flask server for Android emulator full-DLC testing via adb reverse.

Never used for the final phone build: the actual single Android APK embeds a
Chaquopy backend instead. Source server stays on the ephemeral runner.
"""
import json, os, sys
from pathlib import Path
from flask import Response, abort, send_file

def main():
    if len(sys.argv)!=3:
        raise SystemExit("Usage: start_host_server.py /server/repo /full/dlc/root")
    origin=Path(sys.argv[1]).resolve()
    dlc=Path(sys.argv[2]).resolve()
    if not (dlc/"dlc"/"DLCIndex.zip").is_file():
        raise SystemExit("Missing complete DLCIndex.zip")
    towns=origin/"towns"
    towns.mkdir(exist_ok=True)
    conf_path=origin/"config.json"
    c=json.loads(conf_path.read_text(encoding="utf-8"))
    c.update({"listening_ip":"127.0.0.1","listening_port":9000,
              "server_url":"http://127.0.0.1:9000",
              "dlc_dir":str(dlc),"towns_dir":str(towns),
              "active_town":"mytown","run_tutorial":True,"debug":False,
              "http_debug":False,"server_debug_mode":False,
              "reverse_proxy":False})
    conf_path.write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding="utf-8")
    os.chdir(origin)
    sys.path.insert(0,str(origin))
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import tsto_server
    import repair_dlc_index
    srv=tsto_server.TheSimpsonsTappedOutLocalServer()
    srv.dlc_dir=str(dlc)
    srv.towns_dir=str(towns)
    old_dl=srv.dlc_download
    def guarded_dlc(directory,filename):
        if directory=="dlc" and filename=="DLCIndex.zip":
            f=repair_dlc_index.prepare_private_index(dlc)
            if f is not None:
                return send_file(f,mimetype="application/zip",download_name="DLCIndex.zip")
        return old_dl(directory,filename)
    for ep in ("dlc_download","dlc_download_netstorage"):
        if ep in srv.app.view_functions:
            srv.app.view_functions[ep]=guarded_dlc

    # Scorpio requests the bare root URL for UNCOMPRESSED master XML, not
    # zipped DLCIndex.zip. Previously the hosted-server control could
    # return a 404 here even though all 151 subindexes were present.
    def master_xml():
        import zipfile
        with zipfile.ZipFile(dlc/"dlc"/"DLCIndex.zip") as z:
            names=[x for x in z.namelist() if x.lower().endswith(".xml")
                   and not x.startswith("__MACOSX/") and not Path(x).name.startswith("._")]
            if len(names)!=1:
                abort(500,"Unexpected master DLC index XML count")
            return Response(z.read(names[0]),mimetype="application/xml")

    # Optional address-length-preserving patch: original DLC URL size is
    # 89 bytes. The path prefix pads the localhost URL without changing its
    # length; this lets us test the string-allocation hypothesis in native code.
    raw_old = "http://oct2018-4-35-0-uam5h44a.tstodlc.eamobile.com/netstorage/gameasset/direct/simpsons/"
    raw_new = "http://127.0.0.1:9000/gameassets/"
    pad_prefix = "/" + "p" * (len(raw_old) - len(raw_new) - 1)
    srv.app.add_url_rule(pad_prefix + "/gameassets/<string:directory>/<string:filename>",
                         endpoint="padded_dlc_download", view_func=guarded_dlc)
    for route,endpoint in (
        (pad_prefix + "/gameassets/", "master_xml_at_padded_gameassets"),
        ("/gameassets/","master_xml_at_gameassets"),
        ("/netstorage/gameasset/direct/simpsons/","master_xml_at_original_root")
    ):
        if route not in {rule.rule for rule in srv.app.url_map.iter_rules()}:
            srv.app.add_url_rule(route,endpoint=endpoint,view_func=master_xml)
    srv.run()
if __name__=="__main__":main()
