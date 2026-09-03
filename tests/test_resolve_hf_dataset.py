import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "resolve_hf_dataset.py"


class ResolveHfDatasetTest(unittest.TestCase):
    def test_uses_authenticated_hf_owner_instead_of_github_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "huggingface_hub.py").write_text(
                "class HfApi:\n"
                "    def __init__(self, token):\n"
                "        assert token == 'test-token'\n"
                "    def whoami(self):\n"
                "        return {'name': 'actual-hf-user'}\n",
                encoding="utf-8",
            )
            github_env = tmp_path / "github-env"
            env = os.environ.copy()
            env.update(
                {
                    "HF_TOKEN": "test-token",
                    "GITHUB_ENV": str(github_env),
                    "PYTHONPATH": str(tmp_path),
                }
            )

            result = subprocess.run(
                [sys.executable, str(SCRIPT)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                github_env.read_text(encoding="utf-8"),
                "HF_DATASET_ID=actual-hf-user/aios-track2-runs\n",
            )
            self.assertIn("actual-hf-user/aios-track2-runs", result.stdout)
            self.assertNotIn("test-token", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
