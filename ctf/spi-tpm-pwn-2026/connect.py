#!/usr/bin/env python3
"""Bounded TCP scrypt-PoW probe for HITCON 2026 SPI TPM *pwn*.

Never trusts returned text as instructions. Contacts ONLY the literal official CTF
hostname, port 1337. Without --payload sends an empty payload, reads public guest
metadata, and logs out. Does not attempt a kernel exploit.
"""
import argparse
import base64
import hashlib
import re
import socket
import sys
import time
from pathlib import Path

HOST = "spi-tpm-pwn.chal.hitconctf.com"
PORT = 1337
MAX_LINE = 16_000
BANNER_LINES = 12
MAX_PAYLOAD = 2 * 1024 * 1024

class Conn:
    def __init__(self, sock):
        self.sock=sock
        self.buf=bytearray()
    def line(self):
        while b"\n" not in self.buf:
            if len(self.buf)>MAX_LINE: raise ValueError("banner line too long")
            more=self.sock.recv(4096)
            if not more: raise ConnectionError("remote closed")
            self.buf.extend(more)
        raw,_,more=self.buf.partition(b"\n")
        self.buf=bytearray(more)
        return raw.decode("ascii","replace").rstrip("\r")
    def send(self,data):
        self.sock.sendall(data)
    def until(self,token,maximum=8000):
        data=bytes(self.buf)
        self.buf.clear()
        while token not in data:
            if len(data)>maximum:raise ValueError("remote banner too large")
            nxt=self.sock.recv(2048)
            if not nxt:raise ConnectionError("server closed before prompt")
            data+=nxt
        idx=data.index(token)+len(token)
        self.buf.extend(data[idx:])
        return data[:idx]

def powsolve(ch, deadline):
    first=ch.line()
    if first!='proof-of-work scrypt':
        raise ValueError("unexpected PoW mode: "+first[:80])
    f={}
    for _ in range(BANNER_LINES):
        s=ch.line()
        print("POW SERVER:",s,flush=True)
        if s=='nonce': break
        if ' ' in s:
            k,v=s.split(' ',1)
            if k in ('prefix','n','r','p','bits'):
                f[k]=v
    else:
        raise ValueError("long PoW banner")
    if set(f)!=set(('prefix','n','r','p','bits')):raise ValueError("missing PoW attributes")
    prefix=f['prefix']
    if not re.fullmatch("[0-9a-f]{32}",prefix):raise ValueError("bad prefix")
    n,r,p,bits=int(f['n']),int(f['r']),int(f['p']),int(f['bits'])
    if not (1024<=n<=65536 and (n&(n-1))==0 and r==8 and p==1 and 1<=bits<=12):
        raise ValueError("unsafe PoW params")
    salt=("spi-tpm-pow-v1:"+prefix).encode()
    nonce=0
    while time.monotonic()<deadline and nonce<5000000:
        digest=hashlib.scrypt((prefix+str(nonce)).encode(),salt=salt,
                              n=n,r=r,p=p,dklen=32,maxmem=134217728)
        if (int.from_bytes(digest,"big")>>(256-bits))==0:break
        nonce+=1
    else:raise TimeoutError("PoW deadline")
    ch.send(str(nonce).encode()+b"\n")
    response=ch.line()
    if response != "proof-of-work accepted":
        raise ValueError("PoW rejected: "+response[:200])
    print('PoW accepted:',nonce,flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--payload",type=Path)
    parser.add_argument("--connect-timeout",type=int,default=12)
    parser.add_argument("--pow-timeout",type=int,default=275)
    parser.add_argument("--read-only-probe",action="store_true")
    args=parser.parse_args()
    if args.read_only_probe and args.payload:
        parser.error("read-only probe never accepts a payload")
    if not (2<=args.connect_timeout<=25 and 20<=args.pow_timeout<=285):
        parser.error("invalid timeouts")
    payload=b''
    if args.payload:
        assert args.payload.is_file() and not args.payload.is_symlink()
        payload=args.payload.read_bytes()
        if len(payload)>MAX_PAYLOAD:raise ValueError("payload > 2 MiB")
    with socket.create_connection((HOST,PORT),timeout=args.connect_timeout) as sock:
        sock.settimeout(300)
        ch=Conn(sock)
        powsolve(ch,time.monotonic()+args.pow_timeout)
        b=ch.until(b"Base64 payload")
        print('Connected official pwn service:',b.decode('ascii','replace')[:200],flush=True)
        ch.until(b"\n")
        ch.send(base64.b64encode(payload)+b"\n")
        if args.read_only_probe:
            cmd=(b"id\nls -ln /flag /dev/spi-tpm0\n"
                 b"echo END_CTF_DIAGNOSTICS\nexit\n")
            buf=bytes(ch.buf)
            ch.buf.clear()
            sent=False
            deadline=time.monotonic()+95
            while time.monotonic()<deadline:
                if b"SPI TPM challenge" in buf and not sent:
                    ch.send(cmd);sent=True
                if b"END_CTF_DIAGNOSTICS" in buf and sent:
                    print('REMOTE_DIAG',buf[-6000:].decode('ascii','replace'),flush=True)
                    return
                sock.settimeout(min(5,max(1,int(deadline-time.monotonic()))))
                try:buf+=sock.recv(2048)
                except socket.timeout:continue
                if len(buf)>20000:buf=buf[-13000:]
            print('REMOTE_DIAG_TIMED_OUT',buf[-2000:].decode('ascii','replace'),flush=True)
            return
        print("Uploaded",len(payload),"bytes. No interactive commands requested.",flush=True)

if __name__=="__main__":
    try:main()
    except (OSError,TimeoutError,ValueError,ConnectionError) as ex:
        print('FAILED:',type(ex).__name__,str(ex),file=sys.stderr,flush=True)
        sys.exit(1)
