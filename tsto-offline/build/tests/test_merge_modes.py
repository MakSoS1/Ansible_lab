"""Regression tests: external-server game control must not carry Python runtime."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE = Path(__file__).resolve().parents[1] / "merge_game.py"
spec = importlib.util.spec_from_file_location("tsto_merge_game", MODULE)
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)


class MergeModesTest(unittest.TestCase):
    def run_mode(self, args):
        with mock.patch.object(sys, "argv", ["merge_game.py", "--original", "/tmp/ea.apk",
                                              "--apktool", "/tmp/apktool.jar", "--out",
                                              "/tmp/game.apk"] + args), \
             mock.patch.object(merge, "run") as run, \
             mock.patch.object(merge, "merge_dex") as dex, \
             mock.patch.object(merge, "merge_payloads") as native, \
             mock.patch.object(merge, "patch_java"), \
             mock.patch.object(merge, "patch_native"), \
             mock.patch.object(merge, "allow_external_loopback") as external, \
             mock.patch.object(merge, "hook_lifecycle") as embedded:
            merge.main()
            return (run.call_args_list, dex.call_count, native.call_count,
                    external.call_count, embedded.call_count, network.call_count)

    def test_external_server_control_never_loads_donor(self):
        runs, dex, native, external, embedded, network = self.run_mode(["--external-server"])
        self.assertEqual(len(runs), 2, "Game decode and APK rebuild only")
        self.assertEqual(dex, 0)
        self.assertEqual(native, 0)
        self.assertEqual(external, 1)
        self.assertEqual(embedded, 0)
        self.assertEqual(network, 1)

    def test_legacy_offline_guard_can_be_retained_for_ab_testing(self):
        runs, dex, native, external, embedded, network = self.run_mode(
            ["--external-server", "--retain-legacy-network-gates"])
        self.assertEqual(network, 0)

    def test_embedded_server_includes_donor(self):
        runs, dex, native, external, embedded = self.run_mode(["--donor", "/tmp/donor.apk"])
        self.assertEqual(len(runs), 3, "Decode game/donor, rebuild game")
        self.assertEqual(dex, 1)
        self.assertEqual(native, 1)
        self.assertEqual(external, 0)
        self.assertEqual(embedded, 1)
        self.assertEqual(network, 1)

    def test_embedded_server_rejects_missing_donor(self):
        with mock.patch.object(sys, "argv", ["merge_game.py", "--original", "ea.apk",
                    "--apktool", "apktool.jar", "--out", "game.apk"]):
            with self.assertRaises(SystemExit) as result:
                merge.main()
            self.assertEqual(result.exception.code, 2)



class LoopbackNetworkPatchTest(unittest.TestCase):
    def test_legacy_wifi_mobile_guards_are_patched_for_localhost(self):
        targets = (
            ("smali/com/bight/android/app/BGActivity.smali",
             (("isConnectivityOn", True), ("isAirplaneModeOn", False))),
            ("smali/com/ea/simpsons/NetworkStatus.smali",
             (("HasInternetAccess", True),)),
            ("smali/com/ea/nimble/NetworkImpl.smali",
             (("reachabilityCheck", True),)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            game = Path(tmp)
            for path, methods in targets:
                f = game / path
                f.parent.mkdir(parents=True, exist_ok=True)
                text = ".class public LTest;\n"
                for name, value in methods:
                    signature = ("private" if name == "reachabilityCheck"
                                 else "public static")
                    text += (f".method {signature} {name}()Z\n"
                             "    .locals 1\n    const/4 v0, 0x0\n"
                             "    return v0\n.end method\n")
                f.write_text(text, encoding="utf-8")
            merge.patch_android_loopback_connectivity(game)
            for path, methods in targets:
                contents = (game / path).read_text()
                for name, value in methods:
                    needle = ("0x1" if value else "0x0")
                    self.assertIn(f"{name}()Z\n    .locals 1\n"
                                  f"    const/4 v0, {needle}\n"
                                  "    return v0\n.end method", contents)

if __name__ == "__main__":
    unittest.main()
