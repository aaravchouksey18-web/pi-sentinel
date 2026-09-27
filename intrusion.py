#!/usr/bin/env python3
"""pi-intrusion-detection — YOLOv5 (TFLite) person/intrusion detector.

Consolidated from the original testcode-v18/19/20.py and spruced up into an
event-driven sentry:

  * events       - alerts fire on NEW intrusions (not every frame), with a
                   START/END lifecycle (quiet-after + cooldown)
  * alerts       - Telegram (text + annotated photo) via env vars, MQTT
                   payloads on intrusion/events, and a JSONL event log
  * deployment   - deploy/pi-intrusion-detection.service turns this into an
                   always-on headless service
  * pipeline     - letterboxed inference, real YOLOv5 single-output decode,
                   numpy NMS, class filter, boxes mapped back to frame coords

Run modes:
    python3 intrusion.py --image intrusion.jpg --model best-fp16.tflite
    python3 intrusion.py --stream 0
    python3 intrusion.py --stream http://<camera-ip>:81/stream \
        --headless --mqtt-broker 127.0.0.1 --snapshot-dir snapshots

Telegram alerts need environment vars (never passed as args):
    TELEGRAM_BOT_TOKEN=<bot token from @BotFather>
    TELEGRAM_CHAT_ID=<your chat id>
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# image / decode helpers
# --------------------------------------------------------------------------- #

def load_labels(path):
    """Read one class name per line (skipping blank + '#' comment lines)
    into a list, or None if unset/missing."""
    if not path or not os.path.exists(path):
        return None
    names = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            names.append(s)
    return names or None


def letterbox(img, size=640):
    """YOLO-style resize-and-pad to a square. Returns padded image, scale, pads."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img, (nw, nh))
    pad_w, pad_h = size - nw, size - nh
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, left, top


def decode(raw, input_size, threshold):
    """Decode the single YOLOv5 TFLite output [1, 25200, 5+C].

    Every row is an already-decoded anchor: xywh (fractions of the model
    input), objectness, then one score per class. Returns detections in
    MODEL-SPACE pixels with score >= threshold.
    """
    det = raw[0].astype(np.float32)
    obj = det[:, 4]
    cls_conf = det[:, 5:].max(axis=1)
    cls_id = det[:, 5:].argmax(axis=1)
    score = obj * cls_conf
    keep = np.where(score >= threshold)[0]

    out = []
    for i in keep:
        cx, cy, w, h = det[i, :4] * input_size
        out.append({
            "x1": float(cx - w / 2),
            "y1": float(cy - h / 2),
            "x2": float(cx + w / 2),
            "y2": float(cy + h / 2),
            "score": float(score[i]),
            "class_id": int(cls_id[i]),
        })
    return out


def nms(dets, iou_threshold=0.45):
    """Greedy non-maximum suppression (numpy)."""
    if not dets:
        return []
    arr = np.array([[d["x1"], d["y1"], d["x2"], d["y2"], d["score"]]
                    for d in dets])
    order = np.argsort(-arr[:, 4])
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(arr[i, 0], arr[rest, 0])
        yy1 = np.maximum(arr[i, 1], arr[rest, 1])
        xx2 = np.minimum(arr[i, 2], arr[rest, 2])
        yy2 = np.minimum(arr[i, 3], arr[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (arr[i, 2] - arr[i, 0]) * (arr[i, 3] - arr[i, 1])
        area_r = (arr[rest, 2] - arr[rest, 0]) * (arr[rest, 3] - arr[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou < iou_threshold]
    return [dets[j] for j in keep]


def resolve_wanted_ids(wanted, labels):
    """Map `wanted` class names/ids to model indices (empty = nothing matched).

    With a labels file, names are resolved through it. Without one,
    "person" falls back to class id 2 — measured on this model: class
    index 2 fires on a person (see docs/build-log.md).
    """
    ids = set()
    for raw in wanted:
        name = raw.strip().lower()
        if labels:
            for i, lab in enumerate(labels):
                if lab.strip().lower() == name:
                    ids.add(i)
        elif name == "person":
            ids.add(2)
        else:
            try:
                ids.add(int(name))
            except ValueError:
                pass
    return ids


def filter_classes(dets, wanted, labels):
    """Keep only detections whose class is in `wanted` (names or ids)."""
    if not wanted:                                  # empty list => any class
        return dets
    wanted_ids = resolve_wanted_ids(wanted, labels)
    return [d for d in dets if d["class_id"] in wanted_ids]


def map_to_frame(dets, scale, pad_x, pad_y):
    """Convert model-space box pixels back to original frame coordinates."""
    for d in dets:
        d["x1"] = (d["x1"] - pad_x) / scale
        d["y1"] = (d["y1"] - pad_y) / scale
        d["x2"] = (d["x2"] - pad_x) / scale
        d["y2"] = (d["y2"] - pad_y) / scale
    return dets


def class_name(d, labels):
    if labels and d["class_id"] < len(labels):
        return labels[d["class_id"]]
    return f"class-{d['class_id']}"


def draw(frame, dets, labels, active=False, armed=True):
    """Boxes + labels + score bars + status overlay."""
    if not armed:
        status, color = "DISARMED", (140, 140, 140)
    else:
        status, color = ("ACTIVE" if active else "ARMED",
                         (0, 0, 255) if active else (0, 255, 0))
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in (d["x1"], d["y1"], d["x2"], d["y2"]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # score bar beneath the box
        bw = int(round((x2 - x1) * d["score"]))
        cv2.rectangle(frame, (x1, y2), (x2, y2 + 4), (50, 50, 50), -1)
        cv2.rectangle(frame, (x1, y2), (x1 + bw, y2 + 4), color, -1)
        label = f"{class_name(d, labels)} {d['score']:.2f}"
        cv2.putText(frame, label, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    clock = time.strftime("%H:%M:%S")
    cv2.putText(frame, f"{status} | {len(dets)} det | {clock}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def dets_to_json(dets):
    return [{
        "class_id": d["class_id"],
        "score": round(d["score"], 3),
        "bbox": [round(v, 1) for v in (d["x1"], d["y1"], d["x2"], d["y2"])],
    } for d in dets]


def load_interpreter(model_path):
    try:
        from tensorflow import lite  # TF >= 2.x
    except ImportError:
        try:
            import tflite_runtime.interpreter as lite
        except ImportError:
            sys.exit("no TFLite runtime: pip install tensorflow (or tflite-runtime)")
    interp = lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()[0]


class Detector:
    def __init__(self, model_path, threshold=0.5, labels=None, wanted=None):
        self.interp, self.inp, self.out = load_interpreter(model_path)
        shape = list(self.inp["shape"])
        if len(shape) != 4 or shape[1] != shape[2] or shape[1] <= 0:
            sys.exit(f"model input must be [1, H, W, C] with square H == W; "
                     f"got {shape} — re-export with a square input")
        if self.inp.get("dtype") != np.float32:
            sys.exit(f"model input dtype must be float32 (got "
                     f"{self.inp.get('dtype')}); set_tensor requires an exact "
                     f"dtype match, and int8/uint8 quantized exports are not "
                     f"supported by this build — re-export float16/fp32")
        oshape = list(self.out["shape"])
        # The decoder assumes [1, anchors, 5+C] (anchors as ROWS). Modern
        # exports (YOLOv8/v11) are [1, num_classes, 8400] (anchors as
        # COLUMNS): on that layout this decoder would silently return ZERO
        # detections at the default threshold (a guarded, calm sentry that
        # detects nothing all night) or garbage boxes at lower thresholds.
        # Gate the output shape the same way the input is gated.
        if not (len(oshape) == 3 and oshape[0] == 1
                and oshape[1] > 6 and 6 < oshape[2] < 64):
            sys.exit(f"model output must be [1, anchors, 5+classes] (a "
                     f"YOLOv5 single-output head); got {oshape} — re-export "
                     f"with the v5-style head, this decoder cannot read "
                     f"column-major (v8/v11) layouts")
        if self.out.get("dtype") != np.float32:
            sys.exit(f"model output dtype must be float32 (got "
                     f"{self.out.get('dtype')}); an int8/uint8 output tensor "
                     f"would be scaled as if it were 0-1 and silently report "
                     f"garbage scores — re-export fp32")
        # keep input size tied to what the model actually declares
        self.input_size = int(shape[1])
        self.threshold = threshold
        self.labels = labels
        self.wanted = wanted

    def detect(self, frame_bgr):
        padded, scale, px, py = letterbox(frame_bgr, self.input_size)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.interp.set_tensor(self.inp["index"], blob[None])
        self.interp.invoke()
        raw = self.interp.get_tensor(self.out["index"])
        dets = nms(decode(raw, self.input_size, self.threshold))
        dets = filter_classes(dets, self.wanted, self.labels)
        return map_to_frame(dets, scale, px, py)


# --------------------------------------------------------------------------- #
# notifier: event log + MQTT + Telegram
# --------------------------------------------------------------------------- #

class Notifier:
    def __init__(self, args, labels):
        self.args = args
        self.labels = labels
        self.armed = not getattr(args, "start_disarmed", False)
        self.last_command = None
        self.want_status = False
        self._last_io_warn = 0.0

        # A restart (systemd Restart=on-failure, or the 30-read stream-death
        # exit) must never silently re-arm a sentry the user had DISARMED.
        # state.json is written every heartbeat by this same process, so its
        # "armed" is the most recent operator intent — reuse it unless
        # --start-disarmed explicitly says cold boot. (Loop-5 removed the
        # accidental retained-MQTT persistence path; this is the deliberate
        # one.)
        if (not getattr(args, "start_disarmed", False)
                and getattr(args, "state_file", None)):
            restored = self._restore_armed_state()
            if restored is not None:
                self.armed = restored
                print(f"[control] restored armed={self.armed} from "
                      f"{args.state_file} (restart, not cold boot)", flush=True)

        # JSONL event log
        if args.log:
            self.log_line({"ts": time.time(), "type": "boot",
                           "source": args.stream or args.image})

        # MQTT
        self.mqtt_client = None
        if args.mqtt_broker:
            try:
                import paho.mqtt.client as mqtt
            except ImportError:
                sys.exit("--mqtt-broker needs paho-mqtt: "
                         "pip install 'paho-mqtt>=2.0'")
            if hasattr(mqtt, "CallbackAPIVersion"):
                self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            else:                       # paho 1.x fallback (works the same)
                self.mqtt_client = mqtt.Client()

            if hasattr(self.mqtt_client, "max_queued_messages_set"):
                # QoS-1 messages queue while the broker is unreachable; with
                # no cap, a long outage queues thousands of retained heartbeat
                # publishes and the shutdown heartbeat sits behind all of
                # them. Successive heartbeats target the same retained topic,
                # so the newest one is all that matters.
                self.mqtt_client.max_queued_messages_set(2)

            def _on_connect(client, userdata, flags, *args):
                # paho calls this on refused CONNACKs too (2.x hands a
                # ReasonCode, 1.x an int) — say no before (re)subscribing
                rc = args[0] if args else 0
                if getattr(rc, "is_failure", rc != 0):
                    print(f"mqtt: connect refused (rc={rc}); not subscribed",
                          flush=True)
                    return
                # (re)subscribe here: paho's auto-reconnect does NOT restore
                # subscriptions, so without this a broker restart would
                # silently kill arm/disarm forever.
                client.subscribe(self.args.mqtt_control_topic, qos=1)
                print(f"mqtt: connected; subscribed to "
                      f"{self.args.mqtt_control_topic}", flush=True)

            self.mqtt_client.on_connect = _on_connect
            self.mqtt_client.on_message = self._on_control
            # connect_async + loop_start: paho keeps retrying in the
            # background until the broker appears, so a broker down at
            # startup cannot permanently disable MQTT control. QoS>0
            # publishes (the retained heartbeat) queue while offline and
            # flush on reconnect; QoS-0 event publishes are dropped while
            # offline — they are also JSONL + Telegram.
            self.mqtt_client.on_connect_fail = \
                lambda *_: print("mqtt: broker unavailable — retrying in the "
                                 "background", flush=True)
            if hasattr(self.mqtt_client, "suppress_exceptions"):
                self.mqtt_client.suppress_exceptions = True
            # LWT: if the process dies hard (kill -9, power cut) the broker
            # drops this retained OFFLINE heartbeat itself, so the dashboard
            # can never show a calm ARMED sentry that is actually gone.
            self.mqtt_client.will_set(
                args.mqtt_status_topic,
                json.dumps({"online": False,
                            "ts": round(time.time(), 3)}),
                qos=1, retain=True)
            self.mqtt_client.connect_async(args.mqtt_broker,
                                           args.mqtt_port, 30)
            self.mqtt_client.loop_start()

        # Telegram (env-driven; silently off unless both vars present)
        self.telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.telegram_chat = os.environ.get("TELEGRAM_CHAT_ID")
        if self.telegram_token and self.telegram_chat:
            print("telegram alerts enabled")
        elif self.telegram_token or self.telegram_chat:
            print("warning: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID must both be set")

    def _restore_armed_state(self):
        """Last persisted armed value from state.json; None if absent/corrupt."""
        try:
            with open(self.args.state_file, encoding="utf-8") as fh:
                st = json.load(fh)
        except (OSError, ValueError):
            return None
        if not isinstance(st, dict) or "armed" not in st:
            return None
        return bool(st["armed"])

    # -- event log -------------------------------------------------------- #
    def _warn_io(self, msg):
        now = time.time()
        if now - self._last_io_warn >= 30.0:        # don't spam the journal
            self._last_io_warn = now
            print(f"warning: {msg}", flush=True)

    def log_line(self, entry):
        if not self.args.log:
            return
        try:
            with open(self.args.log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as e:
            self._warn_io(f"event log write failed: {e}")

    # -- mqtt ------------------------------------------------------------- #
    def _check_mqtt_delivery(self, kind, info, qos=0):
        """Log (at most every 30 s) publishes that actually drop data.

        rc == MQTT_ERR_NO_CONN (4) with qos>0 means paho QUEUED the message
        and will flush it on reconnect — that is queued, not lost. QoS-0
        publishes are dropped while offline, and a full queue
        (MQTT_ERR_QUEUE_SIZE = 15) drops qos>0 messages; surface both.
        """
        rc = getattr(info, "rc", 0)
        if rc and (qos == 0 or rc == 15):
            self._warn_io(f"mqtt {kind} publish not delivered (rc={rc})")

    def mqtt_publish(self, payload):
        if not self.mqtt_client:
            return
        try:
            info = self.mqtt_client.publish(self.args.mqtt_topic,
                                            json.dumps(payload))
        except Exception as e:
            self._warn_io(f"mqtt event publish failed: {e}")
            return
        self._check_mqtt_delivery("event", info)

    # -- telegram --------------------------------------------------------- #
    def _telegram_url(self, method):
        return (f"https://api.telegram.org/bot{self.telegram_token}/{method}")

    def telegram_send_photo(self, caption, photo_path):
        """Send the photo alert on a daemon thread.

        Telegram can hang for the full urlopen timeout (20 s per call, and
        sendPhoto falls back to sendText on failure = up to 40 s). Running
        it on the side keeps a blackholed endpoint from stalling the
        detection loop — during a stall no frames are watched and
        state.json goes stale.
        """
        if not (self.telegram_token and self.telegram_chat):
            return
        threading.Thread(target=self._telegram_send_photo,
                         args=(caption, photo_path), daemon=True).start()

    def _telegram_send_photo(self, caption, photo_path):
        """Multipart sendPhoto (stdlib urllib, no deps)."""
        if not (self.telegram_token and self.telegram_chat):
            return
        try:
            with open(photo_path, "rb") as fh:
                img = fh.read()
        except OSError:
            self.telegram_send_text(caption)
            return
        boundary = "----piid"
        head = (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
                f"{self.telegram_chat}\r\n"
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="caption"\r\n\r\n'
                f"{caption}\r\n"
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="photo"; '
                f'filename="intrusion.jpg"\r\n'
                f"Content-Type: image/jpeg\r\n\r\n").encode()
        body = head + img + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            self._telegram_url("sendPhoto"), data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(req, timeout=20):
                print("telegram: photo alert sent")
        except Exception as e:
            print("telegram sendPhoto failed:", e)
            self.telegram_send_text("(photo failed) " + caption)

    def telegram_send_text(self, text):
        """Send the text alert on a daemon thread.

        The same stall hazard as the photo path: api.telegram.org can hang
        for the full 20 s urlopen timeout, and fire_end() runs on the
        detection thread — an inline send froze frame processing (10-40
        frames unprocessed at typical ~1-2 s/frame, i.e. an intruder walking
        through during the stall is never seen) and let state.json go stale,
        which briefly showed OFFLINE on the dashboard at every event end.
        """
        if not (self.telegram_token and self.telegram_chat):
            return
        threading.Thread(target=self._telegram_send_text,
                         args=(text,), daemon=True).start()

    def _telegram_send_text(self, text):
        if not (self.telegram_token and self.telegram_chat):
            return
        query = urllib.parse.urlencode({"chat_id": self.telegram_chat,
                                        "text": text})
        req = urllib.request.Request(self._telegram_url("sendMessage") + "?" + query)
        try:
            with urllib.request.urlopen(req, timeout=20):
                print("telegram: text alert sent")
        except Exception as e:
            print("telegram sendMessage failed:", e)

    # -- control plane ---------------------------------------------------- #
    def _on_control(self, client, userdata, msg):
        """Handle arm/disarm/status messages on the control topic."""
        raw = msg.payload.decode(errors="replace").strip()
        try:
            data = json.loads(raw)
            cmd = (str(data.get("command") or data.get("cmd") or "").strip()
                   .lower() if isinstance(data, dict) else str(data).strip()
                   .lower())
        except ValueError:
            cmd = raw.lower()
        if cmd in ("arm", "on", "enable", "resume"):
            action = "armed"
        elif cmd in ("disarm", "off", "disable", "pause"):
            action = "disarmed"
        else:
            action = None                       # "status" / "state" / "?"
        if action and getattr(msg, "retain", 0):
            # A retained arm/disarm on the control topic is re-delivered on
            # every broker (re)connect and on every restart; applying it
            # again lets a stale retained "disarm" disable the sentry for
            # ever. Retained control messages are always the old state, so
            # ignore them outright — send live commands with retain off.
            print(f"[control] ignored {action!r}: retained replay (stale "
                  f"control state on the broker)", flush=True)
            self.last_command = {"cmd": cmd, "ts": round(time.time(), 3),
                                 "ignored": "retained replay"}
            self.want_status = True
            return
        now = time.time()
        if action:
            if action == "armed":
                self.armed = True
            else:
                self.armed = False
            print(f"[control] {action} ({msg.topic})", flush=True)
        self.last_command = {"cmd": cmd or "status", "ts": round(now, 3)}
        self.want_status = True

    def publish_status(self, payload):
        if not self.mqtt_client:
            return None
        try:
            info = self.mqtt_client.publish(self.args.mqtt_status_topic,
                                            json.dumps(payload), retain=True,
                                            qos=1)
        except Exception as e:
            self._warn_io(f"mqtt status publish failed: {e}")
            return None
        self._check_mqtt_delivery("status", info, qos=1)
        return info

    # -- events ----------------------------------------------------------- #
    def fire_start(self, dets, ts, snapshot_path=None):
        payload = {"type": "start", "ts": round(ts, 3),
                   "detections": dets_to_json(dets),
                   "snapshot": snapshot_path}
        self.log_line(payload)          # write the record before MQTT
        self.mqtt_publish(payload)
        names = ", ".join(f"{class_name(d, self.labels)} {d['score']:.2f}"
                          for d in dets)
        caption = (f"🚨 Intrusion detected ({len(dets)}): {names}\n"
                   f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
        if snapshot_path and os.path.exists(snapshot_path):
            self.telegram_send_photo(caption, snapshot_path)
        else:
            self.telegram_send_text(caption)

    def fire_end(self, dets_last, ts, start_ts, reason=None):
        payload = {"type": "end", "ts": round(ts, 3),
                   "start": round(start_ts, 3),
                   "duration": round(ts - start_ts, 2),
                   "detections": dets_to_json(dets_last)}
        if reason:
            payload["reason"] = reason
        self.log_line(payload)          # write the record before MQTT
        self.mqtt_publish(payload)
        d = ts - start_ts
        if reason == "disarmed":
            # the user stepped in or a false alarm was silenced: the scene is
            # NOT confirmed clear — saying "All clear" would send them back
            # to bed while someone may still be on camera
            caption = (f"⚠️ Watch disarmed — alarm silenced after {d:.1f}s, "
                       "scene NOT confirmed clear")
        elif reason == "stream_lost":
            # camera went blind mid-event: there is no evidence the threat
            # left, so this is emphatically not an all-clear (the only trace
            # visible elsewhere is stream_ok:false in state.json)
            caption = (f"⚠️ CAMERA LOST — sentry blind after {d:.1f}s; "
                       "this is NOT an all-clear")
        else:
            caption = f"✅ All clear — alarm lasted {d:.1f}s"
        self.telegram_send_text(caption)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def prune_old_snapshots(snapshot_dir, keep_days):
    """Delete snapshot JPEGs not touched in the last keep_days, once at startup
    so a long-running sentry can't fill the SD card."""
    if keep_days <= 0:
        return 0
    cutoff = time.time() - keep_days * 86400
    removed = 0
    try:
        names = os.listdir(snapshot_dir)
    except OSError:
        return 0
    for name in names:
        path = os.path.join(snapshot_dir, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            pass
    if removed:
        print(f"pruned {removed} snapshot(s) older than {keep_days}d", flush=True)
    return removed


def build_parser():
    p = argparse.ArgumentParser(
        description="YOLOv5 TFLite intrusion sentry (display / headless / image).")
    p.add_argument("--model", default="best-fp16.tflite",
                   help="path to the .tflite model")
    p.add_argument("--labels", default=None,
                   help="optional labels file, one class name per line")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="min object*class score")
    p.add_argument("--classes", default="person",
                   help="comma-separated classes to keep (names via --labels; "
                        "empty for any)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--stream", default=None,
                     help="video source: 0 for webcam, else MJPEG/RTSP URL")
    src.add_argument("--image", default=None,
                     help="process one image file once (single-shot test mode)")
    p.add_argument("--output", default=None,
                   help="write annotated result here (image mode)")
    p.add_argument("--no-display", "--headless", dest="headless",
                   action="store_true", help="never open a window")
    p.add_argument("--snapshot-dir", default=None,
                   help="save an annotated snapshot per intrusion event")
    p.add_argument("--snapshot-keep-days", type=int, default=7,
                   help="prune snapshots older than this at startup (0 = keep all)")
    p.add_argument("--log", default="events.jsonl",
                   help="JSONL event log (default events.jsonl)")
    p.add_argument("--quiet-after", type=float, default=6.0,
                   help="no-detection grace before an EVENT ends (s)")
    p.add_argument("--event-cooldown", type=float, default=30.0,
                   help="min seconds between separate intrusion events")
    p.add_argument("--mqtt-broker", default=None, help="MQTT broker host (alerts)")
    p.add_argument("--mqtt-port", type=int, default=1883)
    p.add_argument("--mqtt-topic", default="intrusion/events")
    p.add_argument("--mqtt-control-topic", default="intrusion/control",
                   help="MQTT topic to arm/disarm the sentry")
    p.add_argument("--mqtt-status-topic", default="intrusion/status",
                   help="MQTT retained status heartbeat topic")
    p.add_argument("--status-interval", type=float, default=10.0,
                   help="seconds between state flush + status heartbeat")
    p.add_argument("--state-file", default="state.json",
                   help="write sentry state here as JSON (for the dashboard)")
    p.add_argument("--start-disarmed", action="store_true",
                   help="boot disarmed; arm later via MQTT control")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.stream and not args.image:
        sys.exit("give --stream <src> or --image <file>")

    # install the graceful-stop handler before the (slow) model load and
    # camera open so a systemd stop is never left without a clean path
    import signal
    signal.signal(signal.SIGTERM,
                  lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    labels = load_labels(args.labels)
    if labels:
        print(f"labels: {len(labels)} classes")
    wanted = [c for c in args.classes.split(",") if c.strip()] if args.classes else []
    if wanted:
        print(f"filter: {args.classes}")
        if not resolve_wanted_ids(wanted, labels):
            sys.exit(f"--classes {args.classes!r} matched nothing "
                     "(is --labels missing, or do the class names differ?) — "
                     "refusing to run a sentry that can never fire")

    notify = Notifier(args, labels)
    print(f"loading model {args.model} ...")
    det = Detector(args.model, args.threshold, labels, wanted)

    if args.snapshot_dir:
        try:
            os.makedirs(args.snapshot_dir, exist_ok=True)
        except OSError as e:
            sys.exit(f"cannot create snapshot dir {args.snapshot_dir}: {e}")
        prune_old_snapshots(args.snapshot_dir, args.snapshot_keep_days)

    # ------------------------------------------------------------------ #
    # single image mode
    # ------------------------------------------------------------------ #
    if args.image:
        if not os.path.exists(args.image):
            sys.exit(f"image not found: {args.image}")
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"cannot read image: {args.image}")
        ts = time.time()
        dets = det.detect(frame)
        annotated = draw(frame.copy(), dets, labels, active=bool(dets),
                         armed=notify.armed)
        if dets:
            print(f"[{ts:.3f}] {len(dets)} detection(s): "
                  + ", ".join(f"{class_name(d, labels)} {d['score']:.2f}"
                              for d in dets), flush=True)
        out_path = args.output or "annotated.jpg"
        cv2.imwrite(out_path, annotated)
        if dets and notify.armed:
            notify.fire_start(dets, ts, snapshot_path=out_path)
        print(f"wrote {out_path}")
        return

    # ------------------------------------------------------------------ #
    # live stream mode (event engine)
    # ------------------------------------------------------------------ #
    # "0" (or any digits) means a webcam index; anything else is a URL/file.
    src = (args.stream or "").strip()
    cap = cv2.VideoCapture(int(src) if src.lstrip("+-").isdigit() else src)
    if not cap.isOpened():
        sys.exit(f"cannot open stream: {src}")
    # don't hang forever inside cap.read() when the stream stalls
    if not cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000):
        print("warning: stream read timeout unsupported — a hung stream can "
              "block read() indefinitely (systemd Restart=on-failure is the "
              "only backstop)", flush=True)
    print(f"watching {src} (Ctrl+C to stop)")

    boot_ts = time.time()
    active = False
    event_start_ts = None
    last_event_ts = None
    last_seen_ts = 0.0
    last_event_end = -999999.0
    fps = 0.0
    prev = time.time()
    last_flush = 0.0
    prev_armed = None
    stream_ok = True
    stream_fails = 0

    def maybe_end(now):
        """Close the current event once the quiet grace period has passed."""
        nonlocal active, last_event_end, event_start_ts
        if active and notify.armed and (now - last_seen_ts) >= args.quiet_after:
            active = False
            print(f"[{now:.3f}] EVENT END — lasted "
                  f"{now - event_start_ts:.1f}s", flush=True)
            notify.fire_end([], now, event_start_ts)
            last_event_end = now

    def sentry_state(now):
        return {
            "ts": round(now, 3),
            "armed": bool(notify.armed),
            "active": bool(active),
            "stream_ok": stream_ok,
            "online": True,
            "fps": round(fps, 1),
            "uptime": round(now - boot_ts, 1),
            "last_event": last_event_ts,
            "last_command": notify.last_command,
        }

    def write_state_atomic(now, payload=None):
        if not args.state_file:
            return
        tmp = args.state_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(sentry_state(now) if payload is None else payload,
                          fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, args.state_file)
        except OSError as e:
            notify._warn_io(f"state write failed: {e}")

    def flush_state(now):
        write_state_atomic(now)
        if notify.mqtt_client:
            notify.publish_status(sentry_state(now))

    def process(frame, now):
        nonlocal active, event_start_ts, last_seen_ts, last_event_end, \
            prev_armed, last_event_ts, fps
        dt = now - prev
        if dt > 0:                                  # guard against clock steps
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        dets = det.detect(frame)
        armed = notify.armed        # read after inference: a disarm lands ~1 frame
        if prev_armed is not None and armed != prev_armed:
            if armed:
                print("[control] re-armed — fresh watch", flush=True)
                last_event_end = -999999.0
            else:
                if active:
                    # close the event on every channel (JSONL + MQTT + Telegram)
                    # so the dashboard doesn't stay ACTIVE after a disarm
                    notify.fire_end([], now, event_start_ts, reason="disarmed")
                    print(f"[{now:.3f}] EVENT END (disarmed) — lasted "
                          f"{now - event_start_ts:.1f}s", flush=True)
                print("[control] disarmed — events suppressed", flush=True)
            active = False
            event_start_ts = None
        prev_armed = armed

        if dets:
            if armed:
                last_seen_ts = now
                if not active and (now - last_event_end) >= args.event_cooldown:
                    # ----- new intrusion event ---------------------------- #
                    active = True
                    event_start_ts = now
                    last_event_ts = now
                    snap = None
                    if args.snapshot_dir:
                        # ms resolution so two starts in the same second
                        # can't silently overwrite each other
                        # int() truncates; round() of 999.6 -> 1000 would
                        # break the dashboard's 3-digit filename regex
                        millis = int((now - int(now)) * 1000) % 1000
                        snap = os.path.join(
                            args.snapshot_dir,
                            time.strftime("%Y%m%d-%H%M%S") + f"-{millis:03d}.jpg")
                        if not cv2.imwrite(snap, draw(frame.copy(), dets, labels,
                                                      active=True, armed=True)):
                            print("warning: failed to write snapshot", flush=True)
                            snap = None
                        else:
                            print(f"snapshot: {snap}", flush=True)
                    print(f"[{now:.3f}] EVENT START — {len(dets)} "
                          f"detection(s): "
                          + ", ".join(f"{class_name(d, labels)} "
                                      f"{d['score']:.2f}" for d in dets),
                          flush=True)
                    notify.fire_start(dets, now, snapshot_path=snap)
        else:
            maybe_end(now)

        annotated = draw(frame.copy(), dets, labels, active=active,
                         armed=armed)
        cv2.putText(annotated, f"{fps:.1f} fps",
                    (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return annotated

    try:
        while True:
            ok, frame = cap.read()
            now = time.time()
            if not ok:
                # stream hiccup / EOF: still close events gracefully, but
                # back off instead of spinning at 100% CPU — and if it stays
                # dead, exit so systemd (Restart=on-failure) brings it back.
                if active and notify.armed:
                    # camera died mid-event: log the end explicitly so a
                    # stream failure is never mistaken for a genuine end
                    active = False
                    print(f"[{now:.3f}] EVENT END (stream lost) — lasted "
                          f"{now - event_start_ts:.1f}s", flush=True)
                    notify.fire_end([], now, event_start_ts,
                                    reason="stream_lost")
                    last_event_end = now
                stream_ok = False
                stream_fails += 1
                if stream_fails == 1:
                    print(f"[{now:.3f}] stream error — camera not producing "
                          f"frames ({src})", flush=True)
                time.sleep(0.2)
                if stream_fails >= 30:
                    write_state_atomic(now)         # last state; stream_ok False
                    sys.exit("stream dead for 30 consecutive reads — "
                             "restarting (systemd Restart=on-failure)")
            else:
                stream_ok = True
                stream_fails = 0
                annotated = process(frame, now)
                prev = now
                if not args.headless:
                    cv2.imshow("Intrusion Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if notify.want_status or now - last_flush >= args.status_interval:
                flush_state(now)
                last_flush = now
                notify.want_status = False
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if notify.mqtt_client:
            # drop the retained heartbeat so the dashboard shows OFFLINE —
            # wait briefly so the QoS-1 publish flushes before the loop stops.
            st = sentry_state(time.time())
            st["online"] = False
            st["active"] = False
            write_state_atomic(time.time(), st)   # state.json too, not only MQTT
            # the 2-message cap (pass-4) can drop this exact heartbeat when
            # the broker was down and the queue is already full — lift it so
            # the OFFLINE announcement is what actually lands.
            if hasattr(notify.mqtt_client, "max_queued_messages_set"):
                notify.mqtt_client.max_queued_messages_set(0)
            info = notify.publish_status(st)
            if info is not None:
                try:
                    info.wait_for_publish(2)
                except Exception:
                    pass
            notify.mqtt_client.loop_stop()


if __name__ == "__main__":
    main()