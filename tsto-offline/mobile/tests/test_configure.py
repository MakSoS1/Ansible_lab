from pathlib import Path
from tempfile import TemporaryDirectory
import json, sys, unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configure_server import configure
class ConfigTests(unittest.TestCase):
    def test_loopback(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"config.json"
            p.write_text(json.dumps({"secret_key":"original-key","client_secret":"existing-key","extra":42}))
            r=configure(p)
            self.assertEqual(r["server_url"],"http://127.0.0.2:9000")
            self.assertEqual(r["secret_key"],"original-key")
            self.assertEqual(r["extra"],42)
            self.assertFalse(r["debug"])
            self.assertTrue(r["run_tutorial"])
            self.assertEqual(json.loads(p.read_text()),r)
    def test_reject_bind(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"config.json";p.write_text("{}")
            with self.assertRaises(ValueError): configure(p,"0.0.0.0")
            with self.assertRaises(ValueError): configure(p,"127.0.0.2",80)
    def test_keep_generated_secret(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"config.json";p.write_text("{}")
            a=configure(p);b=configure(p)
            self.assertEqual(a["secret_key"],b["secret_key"])
            self.assertNotEqual(a["secret_key"],a["client_secret"])
if __name__=="__main__": unittest.main()
