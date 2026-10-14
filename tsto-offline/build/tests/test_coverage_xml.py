import sys,unittest,tempfile,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_asset_coverage import load_xml

class TestCorruptMacMetadata(unittest.TestCase):
    def test_apple_double_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"DLCIndex-v4_1_StPatricksDay-r21336_ANDROID.zip"
            with zipfile.ZipFile(p,"w") as z:
                z.writestr("DLCIndex-v4_1_StPatricksDay-r21336_ANDROID.xml","<Items/>")
                z.writestr("__MACOSX/._DLCIndex-v4_1_StPatricksDay-r21336_ANDROID.xml",b"AppleDouble")
            self.assertEqual(load_xml(p).tag,"Items")
    def test_no_xml_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"empty.zip"
            with zipfile.ZipFile(p,"w") as z:z.writestr("__MACOSX/._bad.xml","x")
            with self.assertRaises(ValueError):load_xml(p)
if __name__=="__main__":unittest.main()
