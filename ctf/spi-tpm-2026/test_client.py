"""Offline tests for the SPI TPM challenge probe; no external connections."""
from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import socket
import unittest

CLIENT_FILE = pathlib.Path(__file__).resolve().with_name("client.py")
spec = importlib.util.spec_from_file_location("spi_tpm_probe", CLIENT_FILE)
assert spec and spec.loader
probe = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)

LOCAL_FAKE_ENVELOPE = (
    "flag 1 "
    "5354503102002800dd1c9a64b35f080022000000d54828d4"
    "2437b9355d69497384df1e189d9780ad"
    "b16e396ec95dc0f9b709ed13dd8f5dcc15130ce6fb28f12d07d9eaef660b9973cd57a4ee"
)


class ProbeTests(unittest.TestCase):
    def test_parse_known_local_fake_flag(self):
        envelope = probe.parse_envelope(LOCAL_FAKE_ENVELOPE)
        self.assertEqual(envelope.key_id, 1)
        self.assertEqual(envelope.cost, 548_787)
        self.assertEqual(envelope.plaintext_length, 34)
        self.assertEqual(len(envelope.ciphertext), 36)
        self.assertEqual(envelope.auth_tag.hex(), "2437b9355d69497384df1e189d9780ad")
        self.assertEqual(envelope.cost * (9 + 256), 145_428_555)

    def test_reject_malformed_flag(self):
        with self.assertRaises(ValueError):
            probe.parse_envelope("flag 1 nothex")
        with self.assertRaises(ValueError):
            probe.parse_envelope(LOCAL_FAKE_ENVELOPE[:-2])
        with self.assertRaises(ValueError):
            probe.parse_envelope(LOCAL_FAKE_ENVELOPE.replace("53545031", "42414421", 1))

    def test_channel_line_and_command_limits(self):
        left, right = socket.socketpair()
        try:
            ch = probe.Channel(left)
            right.sendall(b"ready\r\nbudget 1024\n")
            self.assertEqual(ch.line(), "ready")
            self.assertEqual(ch.line(), "budget 1024")
            ch.command("get_flag")
            self.assertEqual(right.recv(64), b"get_flag\n")
            with self.assertRaises(ValueError):
                ch.command("get_flag; env")
        finally:
            left.close()
            right.close()

    def test_offline_scrypt_pow(self):
        class SimulatedChannel:
            def __init__(self):
                self.messages = iter([
                    "prefix " + "a" * 32,
                    "n 1024",
                    "r 8",
                    "p 1",
                    "bits 3",
                    'find a decimal nonce such that scrypt(prefix || nonce, salt="spi-tpm-pow-v1:" || prefix) has the requested leading zero bits',
                    "nonce",
                    "proof-of-work accepted",
                ])
                self.sent = []
            def line(self):
                return next(self.messages)
            def command(self, cmd):
                self.sent.append(cmd)

        ch = SimulatedChannel()
        nonce = probe.solve_scrypt_pow(ch, 5.0)
        self.assertEqual(ch.sent, [str(nonce)])
        digest = hashlib.scrypt(
            ("a" * 32 + str(nonce)).encode(),
            salt=("spi-tpm-pow-v1:" + "a" * 32).encode(),
            n=1024, r=8, p=1, maxmem=134217728, dklen=32
        )
        self.assertEqual(digest[0] >> 5, 0)


if __name__ == "__main__":
    unittest.main()
