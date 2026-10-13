# Experimental Pixel 10 local server

**NOT the requested one-APK offline game.** The tool does not contain or download the proprietary EA game client, licensed DLC, or upstream Python server code. The server is cloned directly onto the user's own phone on initial installation.

Install Termux; extract this folder to ~/tsto-mobile-kit and run:

~~~sh
cd ~/tsto-mobile-kit
bash install_termux.sh
~~~

After separately obtaining the original game data, provide it in ~/tsto-local/server/dlc, preserving folders and the DLC index at ~/tsto-local/server/dlc/dlc/DLCIndex.zip. This setup is not ready to play until DLC and a compatible patched game client are present. The APK client must address http://127.0.0.2:9000; success on a real Pixel 10 has NOT been verified.

Start the server with ~/tsto-local/start.sh and check with ~/tsto-local/check.sh.

The server binds to device loopback only, no Wi-Fi exposure. The script disables unnecessary debug logging, enables the original tutorial flow, and keeps saves in ~/tsto-local/server/towns. No background scheduled polling is added.

Tests: python -m unittest discover -s tests -v; bash -n install_termux.sh

References: https://github.com/sp00nznet/tsto-springfield and https://github.com/TappedOutReborn/Patch-Apk

Blockers for a single self-contained APK: unavailable embedded Android runtime for the complete game server; proprietary game client and full DLC; actual native runtime and client testing on Pixel 10.
