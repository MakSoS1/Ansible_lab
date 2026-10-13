#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
case "$PREFIX" in *com.termux*) ;; *) echo "Requires Termux" >&2; exit 2;; esac
ROOT="$HOME/tsto-local"
SERVER="$ROOT/server"
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
pkg update -y
pkg install -y python git clang make pkg-config curl
mkdir -p "$ROOT"
if [[ ! -d "$SERVER/.git" ]]; then
  git clone --depth 1 https://github.com/sp00nznet/tsto-springfield.git "$SERVER"
fi
mkdir -p "$SERVER/dlc" "$SERVER/towns"
python -m pip install --no-cache-dir flask flask-inflate protobuf PyJWT Werkzeug
python "$HERE/configure_server.py" "$SERVER/config.json"
cat > "$ROOT/start.sh" <<'START'
#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$HOME/tsto-local/server"
if [[ ! -f dlc/dlc/DLCIndex.zip ]]; then
  echo "DLC index missing: ~/tsto-local/server/dlc/dlc/DLCIndex.zip" >&2
  exit 3
fi
exec python tsto_server.py
START
chmod 700 "$ROOT/start.sh"
cat > "$ROOT/check.sh" <<'CHECK'
#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
curl --max-time 5 -sS -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.2:9000/
CHECK
chmod 700 "$ROOT/check.sh"
echo "Installed local server at $SERVER (no game client or assets included)."
echo "Start with $ROOT/start.sh after you supply licensed DLC."
