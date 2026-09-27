#!/usr/bin/env python3
"""Smoke tests for the stdlib-only dashboard (run: python3 -m pytest tests/
or python3 tests/test_dashboard.py). Starts a real server on an ephemeral
port and checks the security/token paths that used to be untested."""

import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from threading import Thread
from unittest import mock

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


class TokenEscapeTest(DashboardServeMixin, unittest.TestCase):
    token = 'x</script><script>alert(1)</script>y'

    def test_token_cannot_close_the_script_element(self):
        status, body = self.get("/", token=self.token)
        self.assertEqual(status, 200)
        self.assertNotIn(b"TOKEN = \"x</", body)   # raw closer would be parseable
        self.assertNotIn(b"</script><script>", body)


class BindSafetyTest(unittest.TestCase):
    def test_open_nonloopback_bind_refused(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        out = subprocess.run(
            [sys.executable, "-m", "dashboard", "--token", "",
             "--bind", "0.0.0.0"],
            cwd=root, capture_output=True, text=True)
        self.assertEqual(out.returncode, 2)
        self.assertIn("refusing", out.stderr)

    def test_loopback_without_token_starts(self):
        # a loopback no-token bind is legal: main() must reach serve_forever()
        # and stay up (killed here) — not refuse like the non-loopback case
        root = os.path.join(os.path.dirname(__file__), "..")
        proc = subprocess.Popen(
            [sys.executable, "-m", "dashboard", "--token", "",
             "--bind", "127.0.0.1", "--port", "0"],
            cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True)
        try:
            _out, err = proc.communicate(timeout=3)
            self.fail(f"loopback no-token bind refused by main(): {err}")
        except subprocess.TimeoutExpired:
            proc.kill()   # still listening after 3 s == legal bind accepted
        proc.wait()


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


class BuildArgsTest(unittest.TestCase):
    """The argparse layer: --token from DASH_TOKEN, loopback bind gate."""

    def test_token_defaults_from_env(self):
        with mock.patch.dict(os.environ, {"DASH_TOKEN": "envtok-1"}):
            args = dashboard.build_args(["--bind", "127.0.0.1", "--port", "0"])
        self.assertEqual(args.token, "envtok-1")

    def test_explicit_token_overrides_env(self):
        with mock.patch.dict(os.environ, {"DASH_TOKEN": "envtok-1"}):
            args = dashboard.build_args(["--token", "cli-tok",
                                         "--bind", "127.0.0.1", "--port", "0"])
        self.assertEqual(args.token, "cli-tok")

    def test_ipv6_loopback_bind_accepted_without_token(self):
        # the loopback gate used to require version==4 and wrongly refused
        # ::1, pushing IPv6-only setups into --allow-open. ::1 must pass.
        args = dashboard.build_args(["--token", "", "--bind", "::1",
                                     "--port", "0"])
        self.assertEqual(args.bind, "::1")
        args = dashboard.build_args(["--token", "", "--bind",
                                     "::ffff:127.0.0.1", "--port", "0"])
        self.assertEqual(args.bind, "::ffff:127.0.0.1")

    def test_nonloopback_without_token_refused(self):
        with self.assertRaises(SystemExit) as cm:
            dashboard.build_args(["--token", "", "--bind", "0.0.0.0"])
        self.assertEqual(cm.exception.code, 2)


class LogChainTest(unittest.TestCase):
    """read_events must span the logrotate chain, and derive_status must not
    resurrect events orphaned by a restart."""

    def _write_log(self, path, events):
        with open(path, "wb") as fh:
            for e in events:
                fh.write((json.dumps(e) + "\n").encode())

    def test_rotation_chain_merged_in_chronological_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "events.jsonl")
            with gzip.open(base + ".2.gz", "wb") as fh:   # oldest, compressed
                for e in [{"type": "end", "ts": 100.0},
                          {"type": "end", "ts": 101.0}]:
                    fh.write((json.dumps(e) + "\n").encode())
            self._write_log(base + ".1", [{"type": "start", "ts": 200.0},
                                          {"type": "end", "ts": 210.0}])
            self._write_log(base, [{"type": "start", "ts": 300.0}])
            events = dashboard.read_events(base, limit=10)
            self.assertEqual([e["ts"] for e in events],
                             [100.0, 101.0, 200.0, 210.0, 300.0])

    def test_limit_cuts_across_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "events.jsonl")
            self._write_log(base + ".1", [{"ts": i} for i in range(5)])
            self._write_log(base, [{"ts": 10 + i} for i in range(10)])
            # limit within the live file -> all from the live file
            events = dashboard.read_events(base, limit=8)
            self.assertEqual([e["ts"] for e in events],
                             [12, 13, 14, 15, 16, 17, 18, 19])
            # limit larger than the live file -> spills into the rotated
            # chain (older file first, chronologically)
            events = dashboard.read_events(base, limit=12)
            self.assertEqual([e["ts"] for e in events],
                             [3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    def test_rotation_gap_means_no_chain_files(self):
        # no .1/.2 files exist -> plain single-file behavior, unchanged
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "events.jsonl")
            self._write_log(base, [{"ts": 1.0}, {"ts": 2.0}])
            events = dashboard.read_events(base, limit=10)
            self.assertEqual([e["ts"] for e in events], [1.0, 2.0])

    def test_boot_record_breaks_orphaned_start_scan(self):
        # a crash mid-event usually leaves an open START behind; the boot
        # record from the restart must stop the scan so that old event
        # cannot pin the pill ACTIVE/STALE (the new process's state.json is
        # the source of truth after a reboot)
        now = time.time()
        events = [{"type": "start", "ts": now - 3600},
                  {"type": "boot", "ts": now}]
        status, age = dashboard.derive_status(events, active_timeout=120.0)
        self.assertEqual(status, "ARMED")
        self.assertIsNone(age)

    def test_boot_alone_is_armed(self):
        now = time.time()
        status, _ = dashboard.derive_status([{"type": "boot", "ts": now}],
                                            active_timeout=120.0)
        self.assertEqual(status, "ARMED")

    def test_orphaned_start_without_boot_still_stale(self):
        # the boot-break is for the restart case only; a plain open START
        # with no reboot must still go STALE exactly as before
        now = time.time()
        status, age = dashboard.derive_status(
            [{"type": "start", "ts": now - 300}], active_timeout=120.0)
        self.assertEqual(status, "STALE")
        self.assertGreater(age, 170.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)