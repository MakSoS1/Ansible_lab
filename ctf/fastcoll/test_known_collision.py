#!/usr/bin/env python3
"""Reproduce and NEGATIVELY test the published 256-byte XXH3-128 weak-key pair.

This is NOT a remote FastColl solver: the target hashes each subsequent
message with a different seed. No network, shell, tar extraction, or WASM.
The host's preinstalled libxxhash is used through Python's ctypes.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json

MASK64 = (1 << 64) - 1
# XXH3 0.8.3 built-in secret word 7, byte offset 56, little endian.
K7 = 0x4C263A81E69035E0
# Source: Thomas Ahle's September 2026 long-input multicollision analysis.
DATA_WORD = 0x000000007FFFFFFF
WEAK_KEY_WORD = 0xFFFFFFFE7FFFFFFF
# Secret word 7 is odd, hence custom-secret[7] = built-in-secret[7] - seed.
WEAK_SEED = (K7 - WEAK_KEY_WORD) & MASK64
EXPECTED_FULL_DIGEST = "283a0e4119a2f5b7dc4720d53e0bc205"


class Hash128(ctypes.Structure):
    _fields_ = [("low64", ctypes.c_uint64), ("high64", ctypes.c_uint64)]


def library() -> ctypes.CDLL:
    name = ctypes.util.find_library("xxhash")
    if not name:
        raise RuntimeError("system library libxxhash is missing")
    lib = ctypes.CDLL(name)
    lib.XXH_versionNumber.argtypes = []
    lib.XXH_versionNumber.restype = ctypes.c_uint
    if lib.XXH_versionNumber() < 803:
        raise RuntimeError("XXH3 0.8.3 or newer is required")
    lib.XXH3_128bits_withSeed.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64
    ]
    lib.XXH3_128bits_withSeed.restype = Hash128
    return lib


def xxh128(lib: ctypes.CDLL, value: bytes, seed: int) -> str:
    buf = ctypes.create_string_buffer(value, len(value))
    d = lib.XXH3_128bits_withSeed(buf, len(value), seed)
    return f"{d.high64:016x}{d.low64:016x}"


def messages() -> tuple[bytes, bytes]:
    first = bytearray(256)
    second = bytearray(256)
    for position in (56, 112):
        first[position:position + 8] = DATA_WORD.to_bytes(8, "little")
        second[position:position + 8] = (DATA_WORD ^ MASK64).to_bytes(8, "little")
    return bytes(first), bytes(second)


def main() -> None:
    lib = library()
    a, b = messages()
    assert a != b and len(a) == len(b) == 256
    same_seed_a = xxh128(lib, a, WEAK_SEED)
    same_seed_b = xxh128(lib, b, WEAK_SEED)
    next_seed_b = xxh128(lib, b, (WEAK_SEED + 1) & MASK64)

    # This proves the published same-seed full 128-bit collision really works.
    assert same_seed_a == same_seed_b == EXPECTED_FULL_DIGEST
    # This independently demonstrates WHY that pair does not solve the CTF.
    assert same_seed_a != next_seed_b
    assert hashlib.sha256(a).digest() != hashlib.sha256(b).digest()

    output = {
        "test": "full-width 128-bit XXH3 long-input collision",
        "message_lengths": [len(a), len(b)],
        "distinct_messages": a != b,
        "weak_seed_hex": f"{WEAK_SEED:016x}",
        "same_seed_collision": same_seed_a == same_seed_b,
        "different_adjacent_seed_collision": same_seed_a == next_seed_b,
        "same_seed_digest": same_seed_a,
        "adjacent_seed_digest": next_seed_b,
        "conclusion": "Published same-seed pair is NOT valid for this CTF",
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
