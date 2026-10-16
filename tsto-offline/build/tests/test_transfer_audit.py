"""Deterministic WSGI streaming diagnostics, including range and premature close."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transfer_audit import StreamAudit


class StreamAuditTest(unittest.TestCase):
    def invoke(self, method="GET", path="/gameassets/dlc/part.zip",
               chunks=(b"abc", b"de"), headers=None, request_range=None, close_early=False):
        headers = headers if headers is not None else [("Content-Length", "5")]
        messages = []
        status_lines = []

        def app(environ, start_response):
            start_response("206 Partial Content" if request_range else "200 OK", headers)
            return (chunk for chunk in chunks)

        wrapped = StreamAudit(app, logger=messages.append)
        environ = {"PATH_INFO": path, "REQUEST_METHOD": method}
        if request_range is not None:
            environ["HTTP_RANGE"] = request_range
        response = wrapped(environ, lambda status, h, exc_info=None: status_lines.append(status))
        if close_early:
            next(iter(response))
            response.close()
        else:
            list(response)
        self.assertEqual(len(messages), 1)
        self.assertTrue(messages[0].startswith("TSTO_STREAM "))
        return json.loads(messages[0].removeprefix("TSTO_STREAM ")), status_lines

    def test_complete_transfer(self):
        result, lines = self.invoke()
        self.assertEqual(lines, ["200 OK"])
        self.assertEqual(result["bytes_sent"], 5)
        self.assertEqual(result["expected_bytes"], 5)
        self.assertTrue(result["complete"])

    def test_incomplete_transfer_is_detected(self):
        result, _ = self.invoke(close_early=True)
        self.assertEqual(result["bytes_sent"], 3)
        self.assertFalse(result["complete"])

    def test_head_is_not_a_false_incomplete_transfer(self):
        result, _ = self.invoke(method="HEAD", chunks=())
        self.assertEqual(result["bytes_sent"], 0)
        self.assertTrue(result["complete"])

    def test_partial_range_content_headers_recorded(self):
        result, _ = self.invoke(
            chunks=(b"abc",), headers=[("Content-Length", "3"),
                                     ("Content-Range", "bytes 10-12/100"),
                                     ("Content-Encoding", "identity")],
            request_range="bytes=10-12")
        self.assertEqual(result["status"], "206 Partial Content")
        self.assertEqual(result["request_range"], "bytes=10-12")
        self.assertEqual(result["content_range"], "bytes 10-12/100")
        self.assertEqual(result["content_encoding"], "identity")
        self.assertTrue(result["complete"])

    def test_original_length_padded_native_dlc_url_is_logged(self):
        padded = "/" + "p" * 56 + "/gameassets/4_69/real-payload.zip"
        result, status_lines = self.invoke(path=padded)
        self.assertEqual(status_lines, ["200 OK"])
        self.assertEqual(result["path"], padded)
        self.assertEqual(result["bytes_sent"], 5)
        self.assertTrue(result["complete"])

    def test_unrelated_path_does_not_emit_game_dlc_log(self):
        def app(environ, start_response):
            start_response("200 OK", [("Content-Length", "2")])
            return [b"ok"]
        messages = []
        wrapped = StreamAudit(app, logger=messages.append)
        self.assertEqual(b"".join(wrapped({"PATH_INFO": "/mh/gameplayconfig"},
                                            lambda *_: None)), b"ok")
        self.assertEqual(messages, [])


if __name__ == "__main__":
    unittest.main()
