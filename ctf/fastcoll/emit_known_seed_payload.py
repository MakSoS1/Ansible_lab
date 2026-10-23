#!/usr/bin/env python3
"""Emit a standalone WASM C guest containing a real known-seed collision.

The generated guest has exactly one imported function: env.xx_hash.
The 64-bit known seed here is a local test fixture only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from forge_known_seed import forge, get_hasher, hash128, EXPECTED, MASK


def render(seed: int) -> str:
    a = forge(seed)
    b = forge((seed + 1) & MASK)
    assert a != b
    lib = get_hasher()
    assert hash128(lib, a, seed) == hash128(lib, b, (seed + 1) & MASK) == EXPECTED
    joined = a + b
    rows = [", ".join(f"0x{v:02x}" for v in joined[i:i + 16])
            for i in range(0, len(joined), 16)]
    # Static payload fully owns its data, no WASI, no extra imports.
    return (
        '#include <stdint.h>\n'
        '__attribute__((import_module("env"), import_name("xx_hash")))\n'
        'extern int host_xxh3(uint32_t ptr, uint32_t len, uint32_t out);\n'
        'static const uint8_t messages[2048] __attribute__((aligned(8))) = {\n'
        + ',\n'.join(rows) +
        '\n};\n'
        'static uint8_t digest[32];\n'
        '__attribute__((export_name("xx_hash"))) void xx_hash(void) {\n'
        '  host_xxh3((uint32_t)(uintptr_t)&messages[0], 1024, (uint32_t)(uintptr_t)&digest[0]);\n'
        '  host_xxh3((uint32_t)(uintptr_t)&messages[1024], 1024, (uint32_t)(uintptr_t)&digest[16]);\n'
        '}\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outfile", type=Path)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0)
    args = parser.parse_args()
    assert 0 <= args.seed <= MASK
    args.outfile.write_text(render(args.seed), encoding="utf-8")
    print(f"Generated reviewed two-call wasm source for known seed {args.seed:#x}, {args.outfile}")


if __name__ == "__main__":
    main()
