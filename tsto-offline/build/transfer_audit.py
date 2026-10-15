"""WSGI response-stream audit for local test servers.

Unlike Flask's after_request hook, this measures bytes actually yielded
by the response iterator, including premature client disconnects.
Only call it for routes that are safe to log (e.g. local gameasset paths).
"""
import json
import time


class StreamAudit:
    def __init__(self, app, logger=None, path_prefix="/gameassets/"):
        self.app = app
        self.logger = logger or (lambda message: print(message, flush=True))
        self.path_prefix = path_prefix

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if not path.startswith(self.path_prefix):
            return self.app(environ, start_response)
        started = time.monotonic()
        state = {"path": path, "method": environ.get("REQUEST_METHOD", "GET"),
                 "status": None, "expected_bytes": None}

        def audited_start(status, headers, exc_info=None):
            state["status"] = status
            for name, value in headers:
                if name.lower() == "content-length":
                    try:
                        state["expected_bytes"] = int(value)
                    except ValueError:
                        pass
                    break
            return start_response(status, headers, exc_info)

        upstream = self.app(environ, audited_start)

        def stream():
            sent = 0
            completed = False
            failure = None
            try:
                for block in upstream:
                    sent += len(block)
                    yield block
                completed = True
            except Exception as exc:
                failure = type(exc).__name__
                raise
            finally:
                try:
                    if hasattr(upstream, "close"):
                        upstream.close()
                finally:
                    expected = state["expected_bytes"]
                    result = dict(state, bytes_sent=sent,
                                  complete=completed and (expected is None or sent == expected),
                                  elapsed_ms=round((time.monotonic() - started) * 1000, 1))
                    if failure is not None:
                        result["error"] = failure
                    self.logger("TSTO_STREAM " + json.dumps(result, sort_keys=True))

        return stream()
