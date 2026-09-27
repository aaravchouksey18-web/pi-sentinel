#!/usr/bin/env python3
"""Tests for intrusion.py's control/alert plumbing
(run: python3 tests/test_intrusion.py).

intrusion.py imports cv2 + numpy at module level, and neither is guaranteed
on a dev machine — both are stubbed before the import so the control/alert
paths can be exercised without a camera, a broker, or OpenCV. Only pure
control-flow behavior is verified here; model/decode behavior is verified
on the Pi.
"""

import json
import os
import sys
import tempfile
import time
import types
import unittest
from types import SimpleNamespace
from unittest import mock

for _name in ("cv2", "numpy"):          # never installed on this Mac
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import intrusion  # noqa: E402


def make_args(**over):
    base = dict(start_disarmed=False, log=None, mqtt_broker=None,
                mqtt_port=1883, mqtt_topic="intrusion/events",
                mqtt_control_topic="intrusion/control",
                mqtt_status_topic="intrusion/status",
                state_file=None, stream="0", image=None,
                mqtt_username=None, mqtt_password=None)
    base.update(over)
    return SimpleNamespace(**base)


class FireEndCaptionTest(unittest.TestCase):
    """fire_end() must never text a false "All clear" (loop-6 B1)."""

    def _notifier(self):
        return intrusion.Notifier(make_args(), labels=["person"])

    def _text(self, n, **kw):
        with mock.patch.object(intrusion.Notifier, "telegram_send_text") as send:
            n.fire_end([], ts=kw.pop("ts", 1000.0),
                       start_ts=kw.pop("start_ts", 990.0), **kw)
        send.assert_called_once()
        return send.call_args[0][0]

    def test_genuine_quiet_end_says_all_clear(self):
        text = self._text(self._notifier())
        self.assertIn("✅ All clear", text)
        self.assertIn("10.0s", text)

    def test_disarm_end_is_emphatically_not_an_all_clear(self):
        text = self._text(self._notifier(), reason="disarmed")
        self.assertIn("Watch disarmed", text)
        self.assertIn("NOT confirmed clear", text)
        self.assertNotIn("All clear", text)

    def test_stream_lost_end_is_emphatically_not_an_all_clear(self):
        text = self._text(self._notifier(), reason="stream_lost")
        self.assertIn("CAMERA LOST", text)
        self.assertIn("NOT an all-clear", text)
        self.assertNotIn("All clear", text)

    def test_duration_uses_unrounded_seconds(self):
        text = self._text(self._notifier(), ts=1001.234, start_ts=990.0)
        self.assertIn("11.2s", text)

    def test_end_record_carries_the_reason(self):
        # the JSONL/MQTT record must keep stamping why the event ended even
        # though the caption changed — a disarmed event is auditable
        n = self._notifier()
        with mock.patch.object(intrusion.Notifier, "telegram_send_text"):
            with mock.patch.object(n, "log_line") as log:
                n.fire_end([], ts=1000.0, start_ts=990.0, reason="disarmed")
        rec = log.call_args[0][0]
        self.assertEqual(rec["reason"], "disarmed")


class ArmedPersistenceTest(unittest.TestCase):
    """state.json armed restore across restarts (loop-6 S6)."""

    def test_restores_disarmed_from_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = os.path.join(tmp, "state.json")
            with open(st, "w") as fh:
                json.dump({"armed": False, "ts": time.time()}, fh)
            n = intrusion.Notifier(make_args(state_file=st), labels=["person"])
            self.assertFalse(n.armed)

    def test_restores_armed_from_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = os.path.join(tmp, "state.json")
            with open(st, "w") as fh:
                json.dump({"armed": True, "ts": time.time()}, fh)
            n = intrusion.Notifier(make_args(state_file=st), labels=["person"])
            self.assertTrue(n.armed)

    def test_start_disarmed_wins_over_state(self):
        # --start-disarmed is the explicit cold-boot override: it must beat
        # a persisted armed=true
        with tempfile.TemporaryDirectory() as tmp:
            st = os.path.join(tmp, "state.json")
            with open(st, "w") as fh:
                json.dump({"armed": True, "ts": time.time()}, fh)
            n = intrusion.Notifier(make_args(state_file=st,
                                             start_disarmed=True),
                                   labels=["person"])
            self.assertFalse(n.armed)

    def test_corrupt_or_missing_state_falls_back_to_default(self):
        for content in ("{not json", "[1,2,3]", '{"no_armed_key": true}'):
            with tempfile.TemporaryDirectory() as tmp:
                st = os.path.join(tmp, "state.json")
                with open(st, "w") as fh:
                    fh.write(content)
                n = intrusion.Notifier(make_args(state_file=st),
                                       labels=["person"])
                self.assertTrue(n.armed)   # default = starts armed


class TelegramThreadTest(unittest.TestCase):
    """telegram_send_text must not run on the detection thread (loop-6 S1)."""

    def test_send_happens_off_thread(self):
        n = intrusion.Notifier(make_args(), labels=["person"])
        n.telegram_token, n.telegram_chat = "tok", "chat"
        calls = []
        n._telegram_send_text = lambda text: calls.append(text)
        n.telegram_send_text("hello")
        deadline = time.time() + 2.0
        while not calls and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(calls, ["hello"])

    def test_disabled_when_tokens_unset(self):
        n = intrusion.Notifier(make_args(), labels=["person"])
        n.telegram_token, n.telegram_chat = None, None
        t0 = time.time()
        n.telegram_send_text("x")
        self.assertLess(time.time() - t0, 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)