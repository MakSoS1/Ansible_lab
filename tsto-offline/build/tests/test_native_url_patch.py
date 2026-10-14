"""Regression test for the original 4.69.5 ARM64 CDN URL buffer bug."""
import sys
import unittest
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merge_game import A64_CDN_IMMEDIATES, A64_TEMPLATES, CDN, DLC, patch_a64_url_lengths


class TestScorpioLengthPatch(unittest.TestCase):
    def make_elf_stub(self, name):
        url_offset, instructions = A64_CDN_IMMEDIATES[name]
        length = max(x[0] for x in instructions) + 4
        b=bytearray(length)
        b[url_offset:url_offset+len(CDN)]=CDN
        for offset,kind,kind_len in instructions:
            orig_kind = "add_x9_x0" if kind == "add_x0_x0" else kind
            value = len(CDN)+1 if kind_len=="alloc" else len(CDN)
            base,mask,shift,maximum=A64_TEMPLATES[orig_kind]
            struct.pack_into("<I", b, offset, base | (value<<shift))
        return b

    def test_native_neon_and_scalar(self):
        for name,(url_offset,instructions) in A64_CDN_IMMEDIATES.items():
            with self.subTest(name=name):
                code=self.make_elf_stub(name)
                patch_a64_url_lengths(code,name)
                for offset,kind,kind_len in instructions:
                    got=struct.unpack_from("<I",code,offset)[0]
                    base,mask,shift,maximum=A64_TEMPLATES[kind]
                    self.assertEqual(got&mask,base)
                    expected=len(DLC)+1 if kind_len=="alloc" else len(DLC)
                    self.assertEqual((got>>shift)&maximum,expected)

    def test_wrong_opcode_rejected(self):
        name="lib/arm64-v8a/libscorpio.so"
        code=self.make_elf_stub(name)
        struct.pack_into("<I",code,A64_CDN_IMMEDIATES[name][1][0][0],0)
        with self.assertRaisesRegex(RuntimeError,"Unexpected AArch64 instruction"):
            patch_a64_url_lengths(code,name)

    def test_wrong_original_url_location_rejected(self):
        name="lib/arm64-v8a/libscorpio-neon.so"
        code=self.make_elf_stub(name)
        i=code.find(CDN)
        code[i:i+len(CDN)]=b"\0"*len(CDN)
        with self.assertRaisesRegex(RuntimeError,"Unexpected CDN address"):
            patch_a64_url_lengths(code,name)

if __name__=="__main__":
    unittest.main()
