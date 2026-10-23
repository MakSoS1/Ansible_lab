#!/usr/bin/env python3
"""Firmware-backed offline crypto verifier/decryptor. EXECUTE ONLY INSIDE OFFLINE DOCKER.

Loads a SHA256-verified original CTF firmware .so, invokes previously reverse-
engineered PURE crypto subroutines by fixed relative offsets, and never makes
network requests. Do NOT execute this script directly on an Actions VM host.
"""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
import os
import random
import struct
import sys
from pathlib import Path

FIRMWARE=Path("/opt/challenge/libspi_tpm_firmware.so")
EXPECTED_SHA256="1bd76fdd713d7c1003e27e418813e0444c3a2a780bf895836f093048b5cc7a40"
if FIRMWARE.is_symlink() or hashlib.sha256(FIRMWARE.read_bytes()).hexdigest()!=EXPECTED_SHA256:
    raise RuntimeError("Refusing to load unverified firmware binary")

spec=importlib.util.spec_from_file_location(
    "recovery",Path(__file__).with_name("recover_experimental.py"))
assert spec and spec.loader
recovery=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=recovery
spec.loader.exec_module(recovery)

lib=C.CDLL(str(FIRMWARE), mode=os.RTLD_NOW|os.RTLD_LOCAL)
export=C.cast(lib.spi_tpm_fw_get_api_v1,C.c_void_p).value
assert export is not None and export >= 0x3200
base=export-0x3200

RouteType=C.CFUNCTYPE(C.c_uint32,C.c_void_p,C.c_uint32,C.c_uint32)
OuterType=C.CFUNCTYPE(C.c_uint32,C.c_uint32,C.c_uint32,C.c_int32,C.c_int32,C.c_int,C.c_int)
CoreType=C.CFUNCTYPE(C.c_uint32,C.c_void_p,C.c_uint32,C.c_uint32,C.c_int)
route=RouteType(base+0x3850)
outer=OuterType(base+0x17e0)
core=CoreType(base+0x3700)


def buf_key(k0,k1):
    assert 0 <= k0 < (1<<64) and 0<=k1<(1<<64)
    # Firmware accesses exactly 48 bytes, all other derived bytes ignored.
    return (C.c_uint64*6)(k0,k1,0,0,0,0)


def test_primitives():
    pairs=[(0x3dddf9a611e26f3f,0x563598bcddb67486),
           (0,0),(1,2),(0xffffffffffffffff,0xfffffffffffffffe)]
    rng=random.Random(16923)
    for k0,k1 in pairs:
        key=buf_key(k0,k1)
        for _ in range(20):
            nonce=rng.getrandbits(32)
            expect=recovery.public(nonce)^((sum(v*w for v,w in zip(recovery.coeff(nonce),(k0,k1,k0*k1)))%recovery.P)>>118)
            actual=route(C.cast(key,C.c_void_p),nonce,0)
            if actual!=expect:
                raise AssertionError(f"Routing mismatch nonce={nonce:#x}: got {actual:#x} vs {expect:#x}")
            for cost in [1,2,4,15,1024]:
                x=rng.getrandbits(32)
                idx=rng.randrange(10)
                for domain in [0,1]:
                    y=outer(x,nonce,idx,cost,domain,0)
                    z=outer(y,nonce,idx,cost,domain,1)
                    if x!=z:
                        raise AssertionError(f"Outer forward/reverse mismatch domain={domain} cost={cost}")
                y=core(C.cast(key,C.c_void_p),x,cost,0)
                z=core(C.cast(key,C.c_void_p),y,cost,1)
                if x!=z:
                    raise AssertionError(f"Feistel core forward/reverse mismatch cost={cost}")
    print("PASS: firmware routing matches Python bilinear leakage for 80 synthetic cases")
    print("PASS: public transformations and keyed Feistel inverse verified")


def decrypt_file(path, k0, k1, output):
    raw=json.loads(Path(path).read_text(encoding="utf8"))
    cipher=bytes.fromhex(raw["flag_hex"])
    if not 44<=len(cipher)<=4136:
        raise ValueError("bad ciphertext length")
    magic,version,hsize,nonce,cost,length,fingerprint,mac=struct.unpack(
        "<4sHHIIII16s", cipher[:40])
    if magic!=b"STP1" or version!=2 or hsize!=40 or not 1<=length<=1024 or not 1<=cost<=1000000:
        raise ValueError("unexpected signed envelope structure")
    if 40+((length+3)//4)*4!=len(cipher):
        raise ValueError("bad payload length")
    key=buf_key(k0,k1)
    route_value=route(C.cast(key,C.c_void_p),nonce,0)
    if fingerprint!=route_value:
        raise ValueError("recovered seed fails firmware routing fingerprint verification")
    samples=raw.get("samples",[])
    if len(samples)!=27:
        raise ValueError("not enough fingerprints from the same session")
    for s in samples:
        if recovery.public(s["nonce"])^((sum(v*w for v,w in zip(recovery.coeff(s["nonce"]),
                                           (k0,k1,k0*k1)))%recovery.P)>>118)!=s["fingerprint"]:
            raise ValueError("a fingerprint sample conflicts with recovered key")
    plain=bytearray()
    for i in range((length+3)//4):
        word,=struct.unpack_from("<I",cipher,40+i*4)
        w=outer(word,nonce,i,cost,1,1)
        w=core(C.cast(key,C.c_void_p),w,cost,1)
        w=outer(w,nonce,i,cost,0,1)
        plain.extend(struct.pack("<I",w))
    result=bytes(plain[:length])
    try: text=result.decode("ascii").rstrip("\\r\\n")
    except UnicodeError as exc: raise ValueError("decryption is not ASCII") from exc
    if not text.startswith("hitcon{") or not text.endswith("}"):
        raise ValueError("decrypted bytes do not match competition flag format")
    outfile=Path(output)
    if not str(outfile).startswith("/opt/out/") or outfile.is_symlink() or outfile.exists():
        raise ValueError("invalid output file")
    outfile.write_text(text,encoding="ascii")
    print("VERIFIED: recovered plaintext has competition flag syntax; held in isolated output file")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--observations",type=Path)
    p.add_argument("--key0",type=lambda x:int(x,0))
    p.add_argument("--key1",type=lambda x:int(x,0))
    p.add_argument("--out",type=Path)
    opts=p.parse_args()
    if opts.self_test:
        test_primitives()
    else:
        if opts.observations is None or opts.key0 is None or opts.key1 is None or opts.out is None:
            p.error("all of --observations, --key0, --key1 are required")
        decrypt_file(opts.observations,opts.key0,opts.key1,opts.out)
