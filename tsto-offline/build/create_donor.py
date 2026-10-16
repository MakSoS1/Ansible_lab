#!/usr/bin/env python3
"""Create a Chaquopy Android donor APK for embedding an offline game server.
Does not package or distribute EA's proprietary game.
"""
from pathlib import Path
import argparse, shutil

def write(path, data):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(data,encoding="utf-8")

SETTINGS="""pluginManagement {
    repositories { google(); mavenCentral(); gradlePluginPortal() }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories { google(); mavenCentral() }
}
rootProject.name = 'TappedOutLocalSidecar'
include ':app'
"""
TOP="""plugins {
    id 'com.android.application' version '8.10.1' apply false
    id 'com.chaquo.python' version '17.1.0' apply false
}
"""
APP="""plugins {
    id 'com.android.application'
    id 'com.chaquo.python'
}
android {
    namespace 'com.tsto.offline'
    compileSdk 35
    defaultConfig {
        applicationId 'com.tsto.offline.sidecarprobe'
        minSdk 26
        targetSdk 34
        versionCode 1
        versionName '0.0.1'
        ndk { abiFilters 'arm64-v8a', 'x86_64' }
    }
}
chaquopy {
    defaultConfig {
        version = '3.13'
        pip {
            install 'flask==3.1.2'
            install 'flask-inflate'
            install 'protobuf==6.33.0'
            install 'PyJWT==2.10.1'
        }
    }
}
"""
MANIFEST="""<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
  <uses-permission android:name="android.permission.INTERNET" />
  <application android:label="TSTO Offline Server Probe" android:usesCleartextTraffic="true">
    <service android:name="com.tsto.offline.ServerService" android:process=":tsto_backend" android:exported="false" android:stopWithTask="true" />
    <activity android:name="com.tsto.offline.ProbeActivity" android:exported="true">
      <intent-filter><action android:name="android.intent.action.MAIN" />
        <category android:name="android.intent.category.LAUNCHER" />
      </intent-filter>
    </activity>
  </application>
</manifest>
"""
ACTIVITY="""package com.tsto.offline;
import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

public class ProbeActivity extends Activity {
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        TextView view = new TextView(this);
        view.setText("Experimental loopback TSTO game server. No game client included.");
        view.setTextSize(20);
        setContentView(view);
        ServerBootstrap.start(this);
    }
}
"""
BOOTSTRAP="""package com.tsto.offline;
import android.content.Context;
import android.content.Intent;
import android.util.Log;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.FileOutputStream;

public final class ServerBootstrap {
    private static volatile boolean running = false;
    private static final String TAG = "TSTO_OFFLINE_SERVER";
    public static void start(Context context) {
        if (android.os.Build.VERSION.SDK_INT >= 28) {
            String runningProcess = android.app.Application.getProcessName();
            if (runningProcess != null && runningProcess.endsWith(":tsto_backend")) {
                Log.i(TAG, "Avoid recursive start from embedded backend process");
                return;
            }
        }
        Context ctx = context.getApplicationContext();
        Intent intent = new Intent(ctx, ServerService.class);
        ctx.startService(intent);
    }
    public static boolean isBackendProcess() {
        if (android.os.Build.VERSION.SDK_INT < 28) return false;
        String name = android.app.Application.getProcessName();
        return name != null && name.endsWith(":tsto_backend");
    }
    public static synchronized void runInBackend(Context context) {
        Log.i(TAG, "ENTER runInBackend");
        if (running) return;
        running = true;
        final Context ctx = context.getApplicationContext();
        Thread thread = new Thread(() -> {
            try {
                File root = new File(ctx.getFilesDir(), "offline_tsto");
                root.mkdirs();
                File config = new File(root, "config.json");
                if (!config.isFile()) {
                    try (InputStream input = ctx.getAssets().open("server_config.json");
                         OutputStream output = new FileOutputStream(config)) {
                        byte[] buffer = new byte[8192];
                        int n;
                        while ((n = input.read(buffer)) != -1) output.write(buffer, 0, n);
                    }
                }
                new File(root, "dlc").mkdirs();
                new File(root, "towns").mkdirs();
                // Python is isolated inside this app and only listens on 127.0.0.1.
                Log.i(TAG, "BEFORE Python.start");
                if (!Python.isStarted()) {
                    Python.start(new AndroidPlatform(ctx));
                }
                Log.i(TAG, "AFTER Python.start");
                Python.getInstance().getModule("offline_server").callAttr("serve", root.getAbsolutePath());
            } catch (Throwable e) {
                running = false;
                Log.e(TAG, "Offline backend could not start", e);
            }
        }, "TstoLocalServer");
        thread.setDaemon(true);
        thread.start();
    }
}
"""
SERVICE="""package com.tsto.offline;
import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;

public final class ServerService extends Service {
    @Override public void onCreate() {
        super.onCreate();
        Log.i("TSTO_OFFLINE_SERVER", "Separate Android backend process started");
        ServerBootstrap.runInBackend(this);
    }
    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        return START_NOT_STICKY;
    }
    @Override public IBinder onBind(Intent intent) {
        return null;
    }
}
"""
PY="""def serve(root):
    import json
    import os
    from pathlib import Path
    data = Path(root)
    os.chdir(root)
    config_file = data / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config.update(debug=False, http_debug=False, server_debug_mode=False,
                  listening_ip="127.0.0.1", listening_port=9000,
                  server_url="http://127.0.0.1:9000", reverse_proxy=False,
                  run_tutorial=True, dlc_dir="dlc", towns_dir="towns", active_town="mytown")
    config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
    # The original Android client requests the bare assets root first, and
    # otherwise receives a 404 before asking for DLCIndex.zip. Support this
    # compatibility alias without weakening specific per-package paths.
    import tsto_server
    from flask import send_file, abort
    srv = tsto_server.TheSimpsonsTappedOutLocalServer()
    # Flask send_file resolves relative names against app.root_path rather than
    # the current working directory. Its packaged app.root_path is inside
    # Chaquopy, so the DLC path MUST be absolute in the installed Android app.
    srv.dlc_dir = str(data / "dlc")
    srv.towns_dir = str(data / "towns")
    # The native client can request an empty persona ID. Werkzeug normally
    # merges /pids//personas and returns 404; match the tested host server.
    srv.app.url_map.merge_slashes = False
    srv.app.add_url_rule("/proxy/identity/pids//personas",
                         endpoint="local_empty_pid_personas",
                         view_func=lambda: srv.pid_personas(str(srv.user_user_id)))
    # The live master index still references 27 updates absent from the
    # preserved 2024 snapshot. Serve a ZIP retaining only available index
    # entries: this permits loading preserved content but cannot recreate
    # missing event quests and should never be described as complete DLC.
    import repair_dlc_index
    upstream_dl = srv.dlc_download
    def available_dlc(directory, filename):
        if directory == "dlc" and filename == "DLCIndex.zip":
            repaired = repair_dlc_index.prepare_private_index(data / "dlc")
            if repaired is not None:
                return send_file(repaired, mimetype="application/zip",
                                 download_name="DLCIndex.zip")
        return upstream_dl(directory, filename)
    for endpoint in ("dlc_download", "dlc_download_netstorage"):
        if endpoint in srv.app.view_functions:
            srv.app.view_functions[endpoint] = available_dlc
    def master_index_at_root():
        # The Scorpio master-index network handler expects the uncompressed
        # XML document on this bare assets URL (not ZIP bytes).
        from flask import Response
        contents = repair_dlc_index.private_master_xml(data / "dlc")
        if contents is None:
            return abort(404, description="Required DLCIndex.zip not provisioned")
        return Response(contents, mimetype="application/xml")
    srv.app.add_url_rule("/gameassets/", endpoint="master_index_at_root",
                         view_func=master_index_at_root)
    srv.app.add_url_rule("/netstorage/gameasset/direct/simpsons/",
                         endpoint="master_index_at_original_root",
                         view_func=master_index_at_root)
    srv.run()
"""
def main():
    p=argparse.ArgumentParser();p.add_argument("server_root");p.add_argument("output");a=p.parse_args()
    root=Path(a.server_root);out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    write(out/"settings.gradle",SETTINGS);write(out/"build.gradle",TOP);write(out/"gradle.properties","org.gradle.jvmargs=-Xmx3g -XX:MaxMetaspaceSize=512m\nandroid.useAndroidX=true\n")
    write(out/"app/build.gradle",APP)
    write(out/"app/src/main/AndroidManifest.xml",MANIFEST)
    write(out/"app/src/main/java/com/tsto/offline/ProbeActivity.java",ACTIVITY)
    write(out/"app/src/main/java/com/tsto/offline/ServerBootstrap.java",BOOTSTRAP)
    write(out/"app/src/main/java/com/tsto/offline/ServerService.java",SERVICE)
    write(out/"app/src/main/python/offline_server.py",PY)
    shutil.copy2(Path(__file__).resolve().parent/"repair_dlc_index.py", out/"app/src/main/python/repair_dlc_index.py")
    # Personal offline build: preserve the original economic progression, not
    # the third-party server's default balance of 1,234,567 premium donuts.
    src = (root/"tsto_server.py").read_text(encoding="utf-8")
    cheat = 'getattr(self, "donut_balance", 1234567)'
    if src.count(cheat) != 1:
        raise RuntimeError("Upstream premium-currency logic changed; verify manually")
    write(out/"app/src/main/python/tsto_server.py",src.replace(
        cheat, 'getattr(self, "donut_balance", 0)'
    ))
    shutil.copy2(root/"auth_manager.py",out/"app/src/main/python/auth_manager.py")
    # The repository stores protobuf stubs as a proto package.
    for folder in ("proto",):
        src=root/folder
        if not src.is_dir():raise RuntimeError("Missing protobuf stubs: "+str(src))
        shutil.copytree(src,out/"app/src/main/python"/folder,dirs_exist_ok=True)
    (out/"app/src/main/assets").mkdir(parents=True, exist_ok=True)
    shutil.copy2(root/"config.json",out/"app/src/main/assets/server_config.json")
    print("Prepared Chaquopy embedded-server donor",out)
if __name__=="__main__":main()
