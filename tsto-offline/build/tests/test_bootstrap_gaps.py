"""Coverage for bounded DLC 404 parser and archive budget planner."""
import importlib.util
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "report_bootstrap_gaps.py"
spec = importlib.util.spec_from_file_location("tsto_bootstrap_gaps", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestBootstrapGaps(unittest.TestCase):
    def test_real_padded_urls_multiple_retries_and_prior_ok(self):
        data = (
            '127.0.0.1 "GET /pppp/gameassets/4_7_TEST/Shadows.zip HTTP/1.1" 404 -\n'
            '127.0.0.1 "GET /pppp/gameassets/4_7_TEST/Shadows.zip HTTP/1.1" 404 -\n'
            '127.0.0.1 "GET /pppp/gameassets/4_7_TEST/BG.zip HTTP/1.1" 200 -\n'
            '127.0.0.1 "GET /pppp/gameassets/dlc/DLCIndexMissing.zip HTTP/1.1" 404 -\n'
        )
        metadata={"completed":[{"name":"cdn/4_7_TEST.7z","bytes":718001}]}
        native=("LogErrorToServer cause: Too Many Download Retries, message: "
                "ScorpioDLCIndex::AddPackageToDownload failed file: "
                "simpsons4:4_7_TEST:Shadows.zip")
        output = mod.audit(data, native, metadata)
        self.assertEqual(output["missing_unique_paths"], 2)
        self.assertEqual(output["missing_requests"][0]["http_404"], 2)
        self.assertEqual(output["selected_total_bytes"], 718001)
        self.assertEqual(output["selected_archives"][0]["name"], "cdn/4_7_TEST.7z")
        self.assertEqual(output["scorpio_package_failures"][0]["filename"], "Shadows.zip")

    def test_strict_download_budget(self):
        data='GET /gameassets/4_7_TEST/a.zip HTTP/1.1" 404 -'
        catalog={"files":[{"name":"cdn/4_7_TEST.7z","size":"1000"}]}
        result = mod.audit(data, metadata=catalog, budget=999)
        self.assertEqual(result["selected_archives"], [])
        self.assertEqual(len(result["deferred_archives"]), 1)
        with self.assertRaises(ValueError):
            mod.audit(data, budget=-1)

    def test_ignore_non_game_assets_and_bogus_missing_http_success(self):
        data=('GET /mh/gameplayconfig HTTP/1.1" 404 -\n'
              'GET /gameassets/4_7_TEST/a.zip HTTP/1.1" 404 -\n'
              'GET /gameassets/4_7_TEST/a.zip HTTP/1.1" 200 -\n')
        self.assertEqual(mod.audit(data)["missing_unique_paths"], 0)


if __name__ == "__main__":
    unittest.main()
