#!/usr/bin/env python3
"""Local-only Pixel 10 ARM64 diagnostic: authentic game + external host server.

Run on a PC with a USB-attached Android 16 Pixel and ADB installed.
The original game's client APK must be supplied by the user. For diagnostics,
it must be the (--external-server) variant pointing at http://127.0.0.1:9000.
No EA binaries/DLC or private device logs are uploaded by this script.

This DOES NOT validate an all-in-one APK; it isolates native Scorpio execution
on the actual phone from host-provided server and archive resources.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

PKG="com.ea.game.simpsons4_row"
def call(cmd,*,out=None,check=True,timeout=40):
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    if out:Path(out).write_bytes(p.stdout)
    if check and p.returncode:raise RuntimeError(f"{cmd[0]} exit={p.returncode}: {p.stdout[-500:].decode(errors='replace')}")
    return p.stdout.decode(errors="replace")
def getserial(preferred):
    devices=[row.split()[0] for row in call(["adb","devices"]).splitlines()[1:] if re.search(r"\sdevice(?:\s|$)",row)]
    if preferred:
        if preferred not in devices:raise RuntimeError(f"ADB device {preferred!r} not authorized/connected: {devices}")
        return preferred
    if len(devices)!=1:raise RuntimeError(f"Exactly one authorized device needed, got {devices}; pass --serial")
    return devices[0]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--apk",type=Path,required=True,help="your personal patched external-server test APK")
    ap.add_argument("--server-root",type=Path,required=True,help="source checkout of tsto-springfield")
    ap.add_argument("--dlc-root",type=Path,required=True,help="extracted historical DLC root containing dlc/DLCIndex.zip")
    ap.add_argument("--serial")
    ap.add_argument("--seconds",type=int,default=120)
    ap.add_argument("--output",type=Path,default=Path("pixel10-test-results"))
    ap.add_argument("--interactive",action="store_true",help="wait for user to finish tutorial manually before final screenshot")
    args=ap.parse_args()
    for field in (args.apk,args.server_root,args.dlc_root):
        if not field.exists():raise SystemExit(f"Not found: {field}")
    if not (args.dlc_root/"dlc"/"DLCIndex.zip").is_file():raise SystemExit("Missing master DLC index")
    if shutil.which("adb") is None:raise SystemExit("ADB not in PATH. Install Android SDK platform-tools.")
    args.output.mkdir(exist_ok=True,parents=True)
    serial=getserial(args.serial)
    adb=["adb","-s",serial]
    info={}
    for key,cmd in [("model",["shell","getprop","ro.product.model"]),
                    ("version",["shell","getprop","ro.build.version.release"]),
                    ("sdk",["shell","getprop","ro.build.version.sdk"]),
                    ("supported_abis",["shell","getprop","ro.product.cpu.abilist"]),
                    ("page_size",["shell","getconf","PAGE_SIZE"])]:
        info[key]=call(adb+cmd).strip()
    (args.output/"device.json").write_text(json.dumps(info,indent=2,ensure_ascii=False))
    print("Device",info)
    if int(info["sdk"])<36:raise SystemExit("Expected Android 16/API 36 or newer")
    if "arm64-v8a" not in info["supported_abis"]:raise SystemExit("ARM64 unsupported")
    root=Path(__file__).resolve().parent
    try:
        import flask, google.protobuf
    except ImportError:
        raise SystemExit("Install server dependencies in local Python first (Flask, protobuf, PyJWT, flask-inflate)")
    with tempfile.TemporaryDirectory(prefix="tsto-test-") as temp:
        source=Path(temp)/"server"
        shutil.copytree(args.server_root,source,ignore=shutil.ignore_patterns(".git","__pycache__"))
        log=(args.output/"host-server.log").open("wb")
        server=subprocess.Popen([sys.executable,"-u",str(root/"start_host_server.py"),str(source),str(args.dlc_root.resolve())],
              stdout=log,stderr=subprocess.STDOUT)
        try:
            for attempt in range(80):
                if server.poll() is not None:raise RuntimeError(f"Server exited: {server.returncode}. See host-server.log.")
                try:
                    urllib.request.urlopen("http://127.0.0.1:9000/",timeout=1).close()
                    break
                except Exception:
                    time.sleep(.5)
            else:raise RuntimeError("Local server did not start")
            call(adb+["reverse","tcp:9000","tcp:9000"])
            call(adb+["install","-r",str(args.apk.resolve())],out=args.output/"install.txt",timeout=180)
            call(adb+["logcat","-c"])
            call(adb+["shell","monkey","-p",PKG,"-c","android.intent.category.LAUNCHER","1"],out=args.output/"launch.txt")
            time.sleep(max(5,min(600,args.seconds)))
            (args.output/"online-screen.png").write_bytes(subprocess.check_output(adb+["exec-out","screencap","-p"],timeout=25))
            if args.interactive:
                input("On Pixel, try to enter Springfield and complete the first task; press ENTER to capture results.")
            (args.output/"after-interaction.png").write_bytes(subprocess.check_output(adb+["exec-out","screencap","-p"],timeout=25))
            logs=subprocess.check_output(adb+["logcat","-d","-v","time"],timeout=60)
            (args.output/"device-logcat-private.txt").write_bytes(logs)
            raw=logs.decode(errors="replace")
            fatal=bool(re.search(r"Fatal signal.*\b(?:SIGSEGV|SIGABRT)\b",raw))
            alive=PKG in call(adb+["shell","ps","-A"])
            svlog=(args.output/"host-server.log").read_text(errors="replace")
            result={"device":info,"game_alive":alive,"fatal_signal_in_log":fatal,
                    "gameplay_config_200":'GET /mh/gameplayconfig HTTP/1.1" 200' in svlog,
                    "town_get_200":bool(re.search(r'GET /mh/games/bg_gameserver_plugin/protoland/[^"]* HTTP/1.1" 200',svlog)),
                    "town_post_200":bool(re.search(r'POST /mh/games/bg_gameserver_plugin/protoland/[^"]* HTTP/1.1" 200',svlog)),
                    "true_standalone_offline_test":False}
            (args.output/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False))
            print("RESULT",json.dumps(result,indent=2,ensure_ascii=False))
            if fatal or not alive:return 2
            return 0
        finally:
            call(adb+["reverse","--remove","tcp:9000"],check=False)
            server.terminate()
            try:server.wait(timeout=6)
            except subprocess.TimeoutExpired:server.kill()
            log.close()
if __name__=="__main__":
    try:sys.exit(main())
    except (RuntimeError,subprocess.TimeoutExpired) as e:
        print("ERROR",e,file=sys.stderr);sys.exit(2)
