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
    public static synchronized void start(Context context) {
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
                // Python is isolated inside this app and only listens on 127.0.0.2.
                if (!Python.isStarted()) {
                    Python.start(new AndroidPlatform(ctx));
                }
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
PY="""def serve(root):
    import json
    import os
    from pathlib import Path
    data = Path(root)
    os.chdir(root)
    config_file = data / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    config.update(debug=False, http_debug=False, server_debug_mode=False,
                  listening_ip="127.0.0.2", listening_port=9000,
                  server_url="http://127.0.0.2:9000", reverse_proxy=False,
                  run_tutorial=True, dlc_dir="dlc", towns_dir="towns", active_town="mytown")
    config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
    # Run full original Python Flask+protobuf server, not a fake health-check app.
    import tsto_server
    tsto_server.start_server()
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
    write(out/"app/src/main/python/offline_server.py",PY)
    for name in ("tsto_server.py","auth_manager.py"):
        shutil.copy2(root/name,out/"app/src/main/python"/name)
    # The repository stores protobuf stubs as a proto package.
    for folder in ("proto",):
        src=root/folder
        if not src.is_dir():raise RuntimeError("Missing protobuf stubs: "+str(src))
        shutil.copytree(src,out/"app/src/main/python"/folder,dirs_exist_ok=True)
    shutil.copy2(root/"config.json",out/"app/src/main/assets/server_config.json")
    print("Prepared Chaquopy embedded-server donor",out)
if __name__=="__main__":main()
