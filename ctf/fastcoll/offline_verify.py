#!/usr/bin/env python3
"""Offline, read-only reproduction of the FastColl judge's digest/seed checks.

This is NOT a collision exploit. No archive, Docker image, or guest module is
executed; only the installed system xxHash library is used through ctypes.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
from pathlib import Path

MIN_SIZE = 256
MAX_SIZE = 1024
MASK64 = (1 << 64) - 1


class Digest128(ctypes.Structure):
    _fields_ = [("low64", ctypes.c_uint64), ("high64", ctypes.c_uint64)]


class XXH3:
    def __init__(self) -> None:
        try:
            self._library = ctypes.CDLL("libxxhash.so.0")
        except OSError as exc:
            raise RuntimeError(
                "Install the system libxxhash.so.0 (libxxhash0) to run this verifier"
            ) from exc
        self._hash = self._library.XXH3_128bits_withSeed
        self._hash.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64)
        self._hash.restype = Digest128

    def digest(self, message: bytes, seed: int) -> bytes:
        if not 0 <= seed <= MASK64:
            raise ValueError("Seed outside uint64 range")
        buffer = ctypes.create_string_buffer(message)
        value = self._hash(buffer, len(message), seed)
        # Python xxhash.xxh3_128_digest uses canonical big-endian encoding.
        return value.high64.to_bytes(8, "big") + value.low64.to_bytes(8, "big")


def verify_pair(first: bytes, second: bytes, base_seed: int) -> dict[str, str | bool]:
    if not MIN_SIZE <= len(first) <= MAX_SIZE or not MIN_SIZE <= len(second) <= MAX_SIZE:
        raise ValueError("Both messages must contain 256–1024 bytes")
    hasher = XXH3()
    first_digest = hasher.digest(first, base_seed)
    second_digest = hasher.digest(second, (base_seed + 1) & MASK64)
    different = hashlib.sha256(first).digest() != hashlib.sha256(second).digest()
    return {
        "first_seed": f"0x{base_seed:016x}",
        "second_seed": f"0x{(base_seed + 1) & MASK64:016x}",
        "first_digest": first_digest.hex(),
        "second_digest": second_digest.hex(),
        "different_messages": different,
        "won": different and first_digest == second_digest,
    }


def self_test() -> None:
    a = b"A" * 256
    b = b"B" * 256
    result = verify_pair(a, b, 0)
    assert result["different_messages"] is True
    assert result["won"] is False
    assert result["first_seed"] == "0x0000000000000000"
    assert result["second_seed"] == "0x0000000000000001"
    assert len(result["first_digest"]) == 32
    assert len(result["second_digest"]) == 32
    wrapped = verify_pair(a, b, MASK64)
    assert wrapped["second_seed"] == "0x0000000000000000"
    assert verify_pair(a, a, 0)["different_messages"] is False
    try:
        verify_pair(b"short", b, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Short message should be rejected")
    print("PASS: XXH3-128, incremental seeds, wraparound, distinctness, and bounds")
    print("Illustrative non-collision:")
    for key, value in result.items():
        print(f"  {key}: {value}")


def parse_seed(value: str) -> int:
    number = int(value, 0)
    if not 0 <= number <= MASK64:
        raise argparse.ArgumentTypeError("Must be an unsigned 64-bit integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--first", type=Path, help="Path to first binary message")
    parser.add_argument("--second", type=Path, help="Path to second binary message")
    parser.add_argument("--seed", type=parse_seed, default=0,
                        help="Seed of first call, e.g. 0 or 0xdeadbeef (default: 0)")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        return
    if arguments.first is None or arguments.second is None:
        parser.error("Provide --self-test or both --first and --second")
    for path in (arguments.first, arguments.second):
        if not path.is_file():
            parser.error(f"Not a regular file: {path}")
        if not MIN_SIZE <= path.stat().st_size <= MAX_SIZE:
            parser.error(f"Message file must be 256–1024 bytes: {path}")
    outcome = verify_pair(arguments.first.read_bytes(),
                          arguments.second.read_bytes(), arguments.seed)
    for key, value in outcome.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
