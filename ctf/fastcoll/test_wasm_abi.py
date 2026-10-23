#!/usr/bin/env python3
"""Test a generated WASM payload with a self-contained host matching judge ABI.

This is our own small harness, not execution of an untrusted tarball script.
The secret seed is explicitly set for a controlled unit test only.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import wasmtime
import xxhash

MASK = (1 << 64) - 1


def evaluate(wasm_bytes: bytes, seed: int) -> dict[str, object]:
    cfg = wasmtime.Config()
    cfg.consume_fuel = True
    cfg.wasm_threads = False
    cfg.wasm_memory64 = False
    cfg.wasm_multi_memory = False
    engine = wasmtime.Engine(cfg)
    module = wasmtime.Module(engine, wasm_bytes)
    store = wasmtime.Store(engine)
    store.set_fuel(100_000_000)
    linker = wasmtime.Linker(engine)
    count = 0
    seen: dict[bytes, bytes] = {}
    won = False

    def imported_hash(caller: wasmtime.Caller, ptr: int, length: int, out_ptr: int) -> int:
        nonlocal count, won
        if count >= 1_000_000:
            return -2
        current_seed = (seed + count) & MASK
        count += 1
        mem = caller.get("memory")
        if not isinstance(mem, wasmtime.Memory):
            return -1
        size = mem.data_len(caller)
        if not (256 <= length <= 1024 and ptr >= 0 and out_ptr >= 0 and ptr + length <= size and out_ptr + 16 <= size):
            return -1
        data = bytes(mem.read(caller, ptr, ptr + length))
        digest = xxhash.xxh3_128_digest(data, current_seed)
        mem.write(caller, digest, out_ptr)
        fp = hashlib.sha256(data).digest()
        previous = seen.get(digest)
        if previous is None:
            seen[digest] = fp
        elif previous != fp:
            won = True
            return 1
        return 0

    linker.define_func(
        "env", "xx_hash",
        wasmtime.FuncType(
            [wasmtime.ValType.i32(), wasmtime.ValType.i32(), wasmtime.ValType.i32()],
            [wasmtime.ValType.i32()],
        ),
        imported_hash,
        access_caller=True,
    )
    instance = linker.instantiate(store, module)
    exports = instance.exports(store)
    assert isinstance(exports["memory"], wasmtime.Memory)
    exports["xx_hash"](store)
    return {"seed": f"0x{seed:016x}", "calls": count, "won": won}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wasm", type=Path)
    args = parser.parse_args()
    payload = args.wasm.read_bytes()
    assert len(payload) <= 1024 * 1024
    yes = evaluate(payload, 0)
    no = evaluate(payload, 123)
    print("Known seed (0, 1):", yes)
    print("Different seeded deployment (123, 124):", no)
    assert yes["won"] is True and yes["calls"] == 2
    assert no["won"] is False and no["calls"] == 2
    print("PASS: target ABI accepts known-seed collision, random seed remains obstacle")


if __name__ == "__main__":
    main()
