import tempfile, unittest, sys, zipfile, struct
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quick_recon import audit, alignment
class TestQuick(unittest.TestCase):
    def test_elf(self):
        b=bytearray(128);b[:6]=b'\x7fELF\x02\x01';struct.pack_into('<Q',b,32,64);struct.pack_into('<HH',b,54,56,1);struct.pack_into('<I',b,64,1);struct.pack_into('<Q',b,112,0x4000)
        self.assertEqual(alignment(b),[0x4000]);self.assertIsNone(alignment(b'bad'))
    def test_apk(self):
        with tempfile.TemporaryDirectory() as d:
            apk=Path(d)/'f.apk'
            with zipfile.ZipFile(apk,'w') as z:
                z.writestr('AndroidManifest.xml','x');z.writestr('classes.dex','x');z.writestr('lib/arm64-v8a/libscorpio.so','https://prod.simpsons-ea.com')
            r=audit(apk);self.assertTrue(r['manifest_present']);self.assertEqual(r['dex_count'],1);self.assertEqual(r['hosts'][0]['hits'],1)
if __name__=='__main__':unittest.main()
