#!/usr/bin/env python3
"""Eight bounded GETs to the user-authorized CTF host; no guest execution.

All response bodies are encrypted to a one-session RSA public key before being
printed to public Actions logs. The private key is not stored in the repository.
No redirects, cookies, authentication, environment disclosure, or other hosts.
"""
from __future__ import annotations
import base64
import ipaddress
import json
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

HOST = 'fastcoll.chal.hitconctf.com'
PATHS = ('/', '/healthz', '/static/app.js', '/flag.txt', '/static/flag.txt',
         '/runner.py', '/app.py', '/.git/HEAD')
LIMIT = 65536
PUBLIC_KEY = '''-----BEGIN PUBLIC KEY-----
MIIBojANBgkqhkiG9w0BAQEFAAOCAY8AMIIBigKCAYEAxlD7Yb9RS8b3riOO7Cmf
LYse/n0c52WS3Kav81OZeDDgRbpj0IGMVt5Ch/AK4B+oDQEq7xWEVEuKz/k4aDD7
R9MFIA0p+GZDI0z1QfoyyZhmN7SRCZGc/NSD57iqhVm1EvFgEYGd9ALeQVk+iXJ6
DNQR1V9CrWvQLkH49kQN/ahSpD+XDAL5n5am4fxWTD1dkeUnpgemCmQ7hzaVxYdD
v8EI1NJhsE8g5iCKXru+5aRrkXQ5ymTyc3OIaV5PvbLZ0400vBwz41Gfokl2wjs2
pkCZd/Xnix1u2ObXPIeAXUYTO/rKg5WCfwoGjgAK8dZeDkITwO/hFWnsPHz/e3Hr
jkmINftpSeSG7jcCSPUYiL7QkvMQmu+c2CtQwVsJfCElvpiaExIEfUk3AyqFALax
dvbAelr2DV5XMsoluBSqWkNjG6d/iWa5VgVyPCbEpY0IpyfwYIbT9DTZIP3xCqJo
EcI/iZHF/ZYiHgzd4MHip7pjAa0ways9Tbeouw0PchMZAgMBAAE=
-----END PUBLIC KEY-----
'''

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def encrypt(data: bytes, public: Path) -> list[str]:
    compressed = zlib.compress(data, 9)
    output = []
    for offset in range(0, len(compressed), 256):
        process = subprocess.run(
            ['/usr/bin/openssl', 'pkeyutl', '-encrypt', '-pubin',
             '-inkey', str(public), '-pkeyopt', 'rsa_padding_mode:oaep',
             '-pkeyopt', 'rsa_oaep_md:sha256', '-pkeyopt', 'rsa_mgf1_md:sha256'],
            input=compressed[offset:offset + 256], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=10, check=True,
            env={'PATH': '/usr/bin:/bin'},
        )
        output.append(base64.b64encode(process.stdout).decode('ascii'))
    return output


def main() -> None:
    addresses = {entry[4][0] for entry in socket.getaddrinfo(HOST, 80, type=socket.SOCK_STREAM)}
    if not addresses or not all(ipaddress.ip_address(address).is_global for address in addresses):
        raise RuntimeError('Refusing non-public destination')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with tempfile.TemporaryDirectory(prefix='fastcoll-probe-') as folder:
        public = Path(folder) / 'public.pem'
        public.write_text(PUBLIC_KEY, encoding='ascii')
        print('PROBE_SELF_TEST=' + json.dumps(encrypt(b'fastcoll encrypted response transport', public)), flush=True)
        for path in PATHS:
            request = urllib.request.Request('http://' + HOST + path,
                headers={'User-Agent': 'CTF-local-review/1.0', 'Accept': '*/*'})
            try:
                try:
                    response = opener.open(request, timeout=12)
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    data = response.read(LIMIT + 1)
                    record = {'path': path, 'status': response.status,
                              'content_type': response.headers.get('Content-Type'),
                              'truncated': len(data) > LIMIT,
                              'body_b64': base64.b64encode(data[:LIMIT]).decode('ascii')}
            except (OSError, urllib.error.URLError, TimeoutError) as error:
                record = {'path': path, 'error_type': type(error).__name__}
            raw = json.dumps(record, sort_keys=True).encode('utf-8')
            print('PROBE_RECORD=' + json.dumps({'path': path, 'rsa_oaep_sha256_zlib': encrypt(raw, public)}), flush=True)
            time.sleep(1)
        print('PROBE_DONE: eight fixed GETs only; no WASM uploads; all bodies encrypted.', flush=True)

if __name__ == '__main__':
    main()
