import sys,zipfile,tempfile, unittest
from pathlib import Path
from xml.etree import ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repair_dlc_index import filter_master_index, private_master_xml

class TestMasterIndex(unittest.TestCase):
    def test_only_existing_and_zip_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"dlc").mkdir()
            (root/"dlc"/"A.zip").write_bytes(b"present")
            master=root/"master.zip"
            with zipfile.ZipFile(master,"w") as z:
                z.writestr("DLCIndex.xml",
                    '<MasterDLCIndex><IndexFile index="dlc:A.zip" version="1"/>'
                    '<IndexFile index="dlc:B.zip" version="2"/></MasterDLCIndex>')
            packed,missing,count=filter_master_index(master,root)
            self.assertEqual(count,1)
            self.assertEqual(missing,["dlc:B.zip"])
            with zipfile.ZipFile(__import__("io").BytesIO(packed)) as z:
                xml=z.read("DLCIndex.xml")
                self.assertIn(b'dlc:A.zip',xml)
                self.assertNotIn(b'dlc:B.zip',xml)
    def test_bare_root_xml_matches_repaired_zip(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "dlc").mkdir()
            (root / "dlc" / "present.zip").write_bytes(b"asset")
            master = root / "dlc" / "DLCIndex.zip"
            with zipfile.ZipFile(master, "w") as z:
                z.writestr("DLCIndex.xml",
                    '<MasterDLCIndex><IndexFile index="dlc:present.zip"/>'
                    '<IndexFile index="dlc:absent.zip"/></MasterDLCIndex>')
            xml = private_master_xml(root)
            self.assertIn(b"dlc:present.zip", xml)
            self.assertNotIn(b"dlc:absent.zip", xml)
            packed, _, _ = filter_master_index(master, root)
            with zipfile.ZipFile(__import__("io").BytesIO(packed)) as z:
                self.assertEqual(xml, z.read("DLCIndex.xml"))
            self.assertIsNone(private_master_xml(root / "no-such-assets"))

    def test_untrusted_path_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with zipfile.ZipFile(root/"m.zip","w") as z:
                z.writestr("DLCIndex.xml",'<MasterDLCIndex><IndexFile index="dlc:../secret"/></MasterDLCIndex>')
            with self.assertRaises(ValueError):
                filter_master_index(root/"m.zip",root)
if __name__=="__main__":unittest.main()
