#!/usr/bin/env python3
"""Print only public synthetic deterministic test vectors for firmware routines.

This tool MUST be executed inside a network-disabled sandbox, never on host.
No key from any CTF session is used; constants below are invented test data.
"""
import ctypes as C
import hashlib
import os
from pathlib import Path

p = Path("/opt/challenge/libspi_tpm_firmware.so")
assert not p.is_symlink()
assert hashlib.sha256(p.read_bytes()).hexdigest()=="1bd76fdd713d7c1003e27e418813e0444c3a2a780bf895836f093048b5cc7a40"
d = C.CDLL(str(p),mode=os.RTLD_NOW|os.RTLD_LOCAL)
exp=C.cast(d.spi_tpm_fw_get_api_v1,C.c_void_p).value
assert exp is not None and exp>=0x3200
base=exp-0x3200
outer=C.CFUNCTYPE(C.c_uint32,C.c_uint32,C.c_uint32,C.c_int32,C.c_int32,C.c_int,C.c_int)(base+0x17e0)
core=C.CFUNCTYPE(C.c_uint32,C.c_void_p,C.c_uint32,C.c_uint32,C.c_int)(base+0x3700)
key=(C.c_uint64*6)(0x3dddf9a611e26f3f,0x563598bcddb67486,0,0,0,0)
for x,nonce,index,cost in [
 (0x01234567,0xfeedbeef,0,1),
 (0xdeadbeef,0x12345678,5,548787),
 (0x01234567,0,4,10),
 (0,0x12345678,12,19),
]:
    vals=[outer(x,nonce,index,cost,domain,0) for domain in (0,1)]
    back=[outer(val,nonce,index,cost,domain,1) for domain,val in enumerate(vals)]
    for domain,val in enumerate(vals): assert back[domain]==x
    c=core(C.cast(key,C.c_void_p),x,cost,0)
    rev=core(C.cast(key,C.c_void_p),c,cost,1)
    assert rev==x
    print("CASE x=%08x nonce=%08x idx=%d cost=%d outer0=%08x outer1=%08x core=%08x"%(x,nonce,index,cost,vals[0],vals[1],c),flush=True)
