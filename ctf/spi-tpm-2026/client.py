#!/usr/bin/env python3
"""Read the official HITCON SPI TPM *sealed* flag; not a solved exploit.

No shell commands, arbitrary destinations, external modules or downloaded code.
Handles the official challenge's documented scrypt proof-of-work, bounded IO,
validates STP1 formatting, and exits on unexpected server messages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HOST = "spi-tpm-rev.chal.hitconctf.com"
PORT = 1337
LINE_LIMIT = 12288
BANNER_LIMIT = 14


@dataclass(frozen=True)
class Envelope:
    key_id: int
    cost: int
    plaintext_length: int
    nonce: int
    key_fingerprint: int
    auth_tag: bytes
    ciphertext: bytes


class Channel:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.unread = bytearray()

    def line(self) -> str:
        while b"\n" not in self.unread:
            if len(self.unread) > LINE_LIMIT:
                raise ValueError("server sent an oversized line")
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("remote closed the connection")
            self.unread.extend(chunk)
        raw, _, rest = self.unread.partition(b"\n")
        self.unread = bytearray(rest)
        if len(raw) > LINE_LIMIT:
            raise ValueError("server line length exceeded")
        return raw.decode("ascii", errors="strict").rstrip("\r")

    def command(self, value: str) -> None:
        if len(value) > LINE_LIMIT or not re.fullmatch(r"[a-zA-Z0-9_ ]+", value):
            raise ValueError("invalid command")
        self.sock.sendall((value + "\n").encode("ascii"))


def solve_scrypt_pow(ch: Channel, timeout: float) -> int:
    fields: dict[str, str] = {}
    for _ in range(BANNER_LIMIT):
        line = ch.line()
        print("POW:", line)
        if line == "nonce":
            break
        if " " in line:
            k, v = line.split(" ", 1)
            if k in {"prefix", "n", "r", "p", "bits"}:
                if k in fields:
                    raise ValueError("duplicate proof-of-work field")
                fields[k] = v
    else:
        raise ValueError("proof-of-work handshake is too long")

    if set(fields) != {"prefix", "n", "r", "p", "bits"}:
        raise ValueError("missing proof-of-work parameters")
    prefix = fields["prefix"]
    if not re.fullmatch(r"[0-9a-f]{32}", prefix):
        raise ValueError("unexpected challenge prefix")
    n = int(fields["n"])
    r = int(fields["r"])
    p = int(fields["p"])
    bits = int(fields["bits"])
    if not (n >= 1024 and n <= 65536 and n & (n - 1) == 0):
        raise ValueError("unsafe scrypt N parameter")
    if (r, p) != (8, 1) or not 1 <= bits <= 12:
        raise ValueError("unsafe proof-of-work parameters")

    deadline = time.monotonic() + timeout
    salt = ("spi-tpm-pow-v1:" + prefix).encode("ascii")
    for nonce in range(1, 10_000_000):
        if time.monotonic() >= deadline:
            raise TimeoutError("local scrypt PoW computation exceeded time limit")
        digest = hashlib.scrypt(
            (prefix + str(nonce)).encode("ascii"),
            salt=salt, n=n, r=r, p=p,
            maxmem=134_217_728, dklen=32
        )
        if (int.from_bytes(digest, "big") >> (256 - bits)) == 0:
            ch.command(str(nonce))
            answer = ch.line()
            if answer != "proof-of-work accepted":
                raise ValueError("PoW response: " + answer[:200])
            return nonce
    raise RuntimeError("unable to find PoW solution within the search bound")


def parse_envelope(line: str) -> Envelope:
    match = re.fullmatch(r"flag ([1-9][0-9]{0,8}) ([0-9a-fA-F]+)", line)
    if not match:
        raise ValueError("unexpected flag response: " + line[:200])
    key_id = int(match.group(1))
    binary = bytes.fromhex(match.group(2))
    if len(binary) < 44 or len(binary) > 4136:
        raise ValueError("unexpected STP1 envelope size")
    magic, version, header_size, nonce, cost, length, fingerprint, tag = struct.unpack(
        "<4sHHIIII16s", binary[:40]
    )
    if magic != b"STP1" or version != 2 or header_size != 40:
        raise ValueError("invalid STP1 version or header")
    if not (1 <= cost <= 1_000_000 and 1 <= length <= 1024):
        raise ValueError("invalid STP1 cost or plaintext length")
    padded = (length + 3) & ~3
    if len(binary) != 40 + padded:
        raise ValueError("STP1 payload length mismatch")
    return Envelope(key_id, cost, length, nonce, fingerprint, tag, binary[40:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect-timeout", type=float, default=9.0)
    parser.add_argument("--pow-timeout", type=float, default=285.0)
    parser.add_argument("--collect-samples", type=Path, help="Store 27 sample fingerprints in a file")
    args = parser.parse_args()
    if not 1 <= args.connect_timeout <= 30 or not 10 <= args.pow_timeout <= 290:
        parser.error("timeouts outside safe ranges")

    print(f"Connecting ONLY to {HOST}:{PORT}", flush=True)
    with socket.create_connection((HOST, PORT), timeout=args.connect_timeout) as sock:
        sock.settimeout(320)
        ch = Channel(sock)
        greeting = ch.line()
        if greeting == "proof-of-work scrypt":
            print("Authentic-looking scrypt PoW handshake detected")
            nonce = solve_scrypt_pow(ch, args.pow_timeout)
            print("PoW accepted with decimal nonce", nonce)
            greeting = ch.line()
        if greeting != "SPI-TPM reverse service":
            raise ValueError("unexpected server greeting: " + greeting[:200])
        opening = [ch.line(), ch.line(), ch.line()]
        if not (re.fullmatch(r"budget [0-9]+", opening[0])
                and re.fullmatch(r"queries [0-9]+", opening[1])
                and opening[2] == "ready"):
            raise ValueError("unexpected service banner")
        print(*opening, sep="\n")
        ch.command("get_flag")
        result = ch.line()
        sealed = parse_envelope(result)
        print("Encrypted flag is sealed, NOT decrypted.")
        print("key_id:", sealed.key_id)
        print("nonce:", sealed.nonce)
        print("cost:", sealed.cost)
        print("plaintext_length:", sealed.plaintext_length)
        print("key_fingerprint:", sealed.key_fingerprint)
        print("auth_tag_hex:", sealed.auth_tag.hex())
        if not args.collect_samples:
            print("ciphertext_hex:", sealed.ciphertext.hex())
        print("decryption_budget_needed:", sealed.cost * ((sealed.plaintext_length + 3) // 4 + 256))
        if not args.collect_samples:
            print("original_stp1_hex:", result.split()[2])
        if args.collect_samples:
            if sealed.key_id != 1:
                raise ValueError("unexpected flag key ID")
            sample_list=[{"nonce":sealed.nonce,"fingerprint":sealed.key_fingerprint}]
            for i in range(26):
                ch.command("encrypt 1 1 41")
                response=ch.line()
                if not re.fullmatch(r"ciphertext [0-9a-fA-F]{82,256}",response):
                    raise ValueError(f"unexpected encrypt response #{i}: {response[:200]}")
                env=parse_envelope("flag 1 "+response.split()[1])
                if env.cost != 1 or env.plaintext_length != 1:
                    raise ValueError("unexpected cost or length in chosen-plaintext ciphertext")
                sample_list.append({"nonce":env.nonce,"fingerprint":env.key_fingerprint})
                print(f"fingerprint sample {i+1}/26: {env.nonce:#010x}",flush=True)
            if len({item["nonce"] for item in sample_list}) != len(sample_list):
                raise ValueError("repeated nonce reduces independent sample count")
            result={"samples":sample_list, "flag_hex": result.split()[2],
                    "key_id":sealed.key_id, "flag_length":sealed.plaintext_length}
            args.collect_samples.write_text(json.dumps(result,indent=2),encoding="utf-8")
            print("Saved verified fingerprint samples:",len(sample_list))
            print("Saved sealed flag with cost:",sealed.cost)
        ch.command("quit")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, TimeoutError, RuntimeError, ConnectionError) as exc:
        print(f"CTF probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
