#!/usr/bin/env python3
"""Construct FULL 128-bit XXH3-128 collisions across consecutive KNOWN seeds.

Does NOT solve the live CTF: the remote judge keeps its 64-bit seed secret.
For each known s, creates a distinct 1024-byte message M(s) satisfying

    XXH3_128(M(s), s) = XXH3_128(M(s+1), s+1)

with canonical full digest dbfd52f5f5a41471dbbb3ad8d9bd8268.
No bundled challenge code is imported or executed. Pure Python + system libxxhash.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import sys

MASK = (1 << 64) - 1
MASK32 = (1 << 32) - 1
MESSAGE_SIZE = 1024
EXPECTED = "dbfd52f5f5a41471dbbb3ad8d9bd8268"
DEFAULT_SECRET = bytes.fromhex(
    "b8fe6c3923a44bbe7c01812cf721ad1c"
    "ded46de9839097db7240a4a4b7b3671f"
    "cb79e64eccc0e578825ad07dccff7221"
    "b8084674f743248ee03590e6813a264c"
    "3c2852bb91c300cb88d0658b1b532ea3"
    "71644897a20df94e3819ef46a9deacd8"
    "a8fa763fe39c343ff9dcbbc7c70b4f1d"
    "8a51e04bcdb45931c89f7ec9d9787364"
    "eac5ac8334d3ebc3c581a0fffa1363eb"
    "170ddd51b7f0da49d316552629d4689e"
    "2b16be587d47a1fc8ff8b8d17ad031ce"
    "45cb3a8f95160428afd7fbcabb4b407e"
)
assert len(DEFAULT_SECRET) == 192
INIT_ACC = [
    0xC2B2AE3D,
    0x9E3779B185EBCA87,
    0xC2B2AE3D27D4EB4F,
    0x165667B19E3779F9,
    0x85EBCA77C2B2AE63,
    0x85EBCA77,
    0x27D4EB2F165667C5,
    0x9E3779B1,
]


class Hash128(ctypes.Structure):
    _fields_ = [("low64", ctypes.c_uint64), ("high64", ctypes.c_uint64)]


def get_hasher():
    path = ctypes.util.find_library("xxhash")
    if not path:
        raise RuntimeError("System libxxhash is required")
    lib = ctypes.CDLL(path)
    lib.XXH3_128bits_withSeed.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64
    ]
    lib.XXH3_128bits_withSeed.restype = Hash128
    return lib


def hash128(lib, data: bytes, seed: int) -> str:
    buffer = ctypes.create_string_buffer(data)
    value = lib.XXH3_128bits_withSeed(buffer, len(data), seed & MASK)
    return f"{value.high64:016x}{value.low64:016x}"


def read64(data, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 8], "little")


def write64(data: bytearray, offset: int, value: int):
    data[offset:offset + 8] = (value & MASK).to_bytes(8, "little")


def derive_secret(seed: int) -> bytearray:
    secret = bytearray(DEFAULT_SECRET)
    for j in range(24):
        base = read64(DEFAULT_SECRET, 8 * j)
        adjusted = (base + seed * (1 if j % 2 == 0 else -1)) & MASK
        write64(secret, 8 * j, adjusted)
    return secret


def compute_acc(message: bytes, secret: bytes) -> list[int]:
    acc = INIT_ACC.copy()
    for stripe in range(16):
        secret_offset = 8 * stripe if stripe < 15 else 121
        for lane in range(8):
            data = read64(message, 64 * stripe + 8 * lane)
            key = read64(secret, secret_offset + 8 * lane)
            x = data ^ key
            acc[lane] = (acc[lane] + (x & MASK32) * (x >> 32)) & MASK
            acc[lane ^ 1] = (acc[lane ^ 1] + data) & MASK
    return acc


def forge(seed: int) -> bytes:
    """Build 1024-byte message whose 8 accumulators hit chosen target values."""
    secret = derive_secret(seed & MASK)
    message = bytearray(MESSAGE_SIZE)
    # Put derived secret at every lane: all products are now exactly zero.
    for stripe in range(16):
        off = stripe * 8 if stripe < 15 else 121
        for lane in range(8):
            write64(message, stripe * 64 + 8 * lane, read64(secret, off + 8 * lane))

    before = compute_acc(message, secret)
    # For low output pairs, zero even multiplicands.
    # For high output pairs, zero odd multiplicands.
    targets = [
        read64(secret, (11 if i % 2 == 0 else 117) + i * 8)
        for i in range(8)
    ]

    for lane, target in enumerate(targets):
        # The inverse lane's raw data contributes to this accumulator.
        other = lane ^ 1
        key0 = read64(secret, 8 * other)       # full stripe 0
        key1 = read64(secret, 8 + 8 * other)   # full stripe 1
        delta = (target - before[lane]) & MASK

        # Keep g(D xor K) == 0 in both modified words:
        # word at stripe 0 retains K.low32 and has arbitrary high32;
        # word at stripe 1 retains K.high32 and has arbitrary low32.
        new_low = ((key1 & MASK32) + (delta & MASK32)) & MASK32
        low_delta = (new_low - (key1 & MASK32)) & MASK
        remaining = (delta - low_delta) & MASK
        assert (remaining & MASK32) == 0
        new_high = ((key0 >> 32) + (remaining >> 32)) & MASK32
        write64(message, 8 * other, (new_high << 32) | (key0 & MASK32))
        write64(message, 64 + 8 * other, (key1 & ~MASK32) | new_low)

    assert compute_acc(message, secret) == targets
    return bytes(message)


def main() -> None:
    lib = get_hasher()
    cases = [0, 1, 123, 0x0123456789ABCDEF, MASK]
    for seed in cases:
        following = (seed + 1) & MASK
        left = forge(seed)
        right = forge(following)
        hash_left = hash128(lib, left, seed)
        hash_right = hash128(lib, right, following)
        assert len(left) == len(right) == MESSAGE_SIZE
        assert left != right
        assert hashlib.sha256(left).digest() != hashlib.sha256(right).digest()
        assert hash_left == hash_right == EXPECTED, (seed, hash_left, hash_right)
        print(json.dumps({
            "seed": f"0x{seed:016x}",
            "seed_next": f"0x{following:016x}",
            "digest": hash_left,
            "distinct_messages": True,
            "collision": True,
            "sha256_left": hashlib.sha256(left).hexdigest(),
            "sha256_right": hashlib.sha256(right).hexdigest(),
        }, sort_keys=True))
    print("PASS: full-width cross-seed collisions for explicitly known seeds")
    print("NOT A LIVE FLAG: remote random 64-bit base seed is unknown to guest")


if __name__ == "__main__":
    main()
