"""Regression contract for the anonymous persona URL used by Scorpio."""
import unittest
from pathlib import Path

class HostIdentityContract(unittest.TestCase):
    def test_empty_pid_is_mapped_to_local_identity(self):
        source = (Path(__file__).resolve().parents[1] / "start_host_server.py").read_text()
        self.assertIn('"/proxy/identity/pids//personas"', source)
        self.assertIn('srv.pid_personas(str(srv.user_user_id))', source)
        self.assertIn('srv.app.url_map.merge_slashes = False', source)

    def test_embedded_android_server_has_same_empty_pid_fix(self):
        from importlib.util import spec_from_file_location, module_from_spec
        source = (Path(__file__).resolve().parents[1] / "create_donor.py")
        spec = spec_from_file_location("tsto_donor_contract", source)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        compile(mod.PY, "<embedded-offline-server>", "exec")
        self.assertIn('srv.app.url_map.merge_slashes = False', mod.PY)
        self.assertIn('srv.app.add_url_rule("/proxy/identity/pids//personas"', mod.PY)
        self.assertIn('repair_dlc_index.private_master_xml(data / "dlc")', mod.PY)

if __name__ == "__main__":
    unittest.main()
