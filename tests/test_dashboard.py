#!/usr/bin/env python3
"""Smoke tests for the stdlib-only dashboard (run: python3 -m pytest tests/
or python3 tests/test_dashboard.py). Starts a real server on an ephemeral
port and checks the security/token paths that used to be untested."""

import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from threading import Thread

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dashboard  # noqa: E402

# never interpolate a real token into requests for these
TOKEN = "opensesame-123"


class DashboardServeMixin:
    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.log_path = os.path.join(tmp, "events.jsonl")
        self.state_path = os.path.join(tmp, "state.json")
        self.handler = dashboard.make_handler(
            self.log_path, None, active_timeout=120.0,
            state_file=self.state_path, token=self.token,
            heartbeat_timeout=20.0)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), self.handler)
        self.port = self.srv.server_address[1]
        self.thread = Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join(timeout=5)

    def get(self, path, token=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        if token:
            url += ("&" if "?" in url else "?") + "t=" + urllib.parse.quote(token)
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def get_bearer(self, token):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/events")
        req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()


class TokenDashboardTest(DashboardServeMixin, unittest.TestCase):
    token = TOKEN

    def test_page_always_served_but_token_gated(self):
        status, _ = self.get("/")
        self.assertEqual(status, 401)

    def test_correct_query_token_serves_page(self):
        status, body = self.get("/", token=TOKEN)
        self.assertEqual(status, 200)
        self.assertNotIn(b"__TOKEN_JS__", body, "placeholder leaked into page")
        self.assertIn("opensesame-123".encode(), body)

    def test_bearer_token_serves_api(self):
        status, body = self.get_bearer(TOKEN)
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["status"], "ARMED")

    def test_non_ascii_query_value_rejected_not_crash(self):
        # ?t=%C3%A9 hits hmac.compare_digest; must yield 401 not a traceback
        status, _ = self.get("/", token="\u00e9")
        self.assertEqual(status, 401)
        # the server must still be alive and answering afterwards
        status, _ = self.get("/", token=TOKEN)
        self.assertEqual(status, 200)

    def test_snapshot_path_traversal_rejected(self):
        status, _ = self.get("/snapshots/..%2F..%2Fetc%2Fpasswd", token=TOKEN)
        self.assertEqual(status, 400)


class NoTokenDashboardTest(DashboardServeMixin, unittest.TestCase):
    token = None

    def test_no_token_means_open(self):
        status, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertNotIn(b"__TOKEN_JS__", body)


class ReadStateTest(unittest.TestCase):
    def test_json_list_is_not_a_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            with open(path, "w") as fh:
                json.dump([1, 2, 3], fh)
            self.assertIsNone(dashboard.read_state(path))

    def test_corrupt_json_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            with open(path, "w") as fh:
                fh.write("{not json")
            self.assertIsNone(dashboard.read_state(path))

    def test_good_state_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            with open(path, "w") as fh:
                json.dump({"armed": True, "ts": 1.0}, fh)
            self.assertEqual(dashboard.read_state(path)["armed"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)