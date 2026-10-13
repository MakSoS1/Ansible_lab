import io
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from tsto_prepare import elf_alignment, audit_apk, parse_dlc_index, plan_dlc

INDEX = b'''<Root>
<Package tier="100"><FileName name="images:common.zip"/><Language val=""/><FileSize val="100"/></Package>
<Package tier="100"><FileName name="lang:ru.zip"/><Language val="ru"/><FileSize val="20"/></Package>
<Package tier="100"><FileName name="lang:en.zip"/><Language val="en"/><FileSize val="30"/></Package>
<Package tier="100"><FileName name="lang:de.zip"/><Language val="de"/><FileSize val="40"/></Package>
<Package tier="iphone"><FileName name="images:iphone.zip"/><Language val="en"/><FileSize val="50"/></Package>
</Root>'''

def fake_elf(alignment=0x4000):
    data = bytearray(128)
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", data, 32, 64)
    struct.pack_into("<HH", data, 54, 56, 1)
    struct.pack_into("<I", data, 64, 1)
    struct.pack_into("<Q", data, 112, alignment)
    return bytes(data)

class PreflightTests(unittest.TestCase):
    def test_elf(self):
        self.assertEqual(elf_alignment(fake_elf()), [16384])
        self.assertEqual(elf_alignment(fake_elf(4096)), [4096])
        self.assertEqual(elf_alignment(b"bad"), [])

    def test_apk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.apk"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("AndroidManifest.xml", b"manifest")
                z.writestr("lib/arm64-v8a/libscorpio.so", fake_elf())
            report = audit_apk(path)
            self.assertTrue(report["arm64_engine"])
            self.assertTrue(report["libraries"][0]["supports_16kb_elf"])
            self.assertTrue(report["warnings"])

    def test_dlc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.xml"
            path.write_bytes(INDEX)
            p = plan_dlc([path])
            self.assertEqual(p["selected_count"], 3)
            self.assertEqual(p["compressed_bytes_estimate"], 150)
            self.assertEqual(p["excluded_count"], 2)

    def test_zip_index(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as z:
            z.writestr("index.xml", INDEX)
        self.assertEqual(len(parse_dlc_index(data.getvalue())), 5)

if __name__ == "__main__":
    unittest.main()
