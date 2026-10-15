"""Regression contract for the anonymous persona URL used by Scorpio."""
import unittest
from pathlib import Path

class HostIdentityContract(unittest.TestCase):
    def test_empty_pid_is_mapped_to_local_identity(self):
        source = (Path(__file__).resolve().parents[1] / "start_host_server.py").read_text()
        self.assertIn('"/proxy/identity/pids//personas"', source)
        self.assertIn('srv.pid_personas(str(srv.user_user_id))', source)

if __name__ == "__main__":
    unittest.main()
