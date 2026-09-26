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
    with open(path, "r") as fh:
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


def filter_classes(dets, wanted, labels):
    """Keep only detections whose class is in `wanted` (names or ids).

    With a labels file, names are resolved through it. Without one,
    "person" falls back to class id 2 — measured on this model: class
    index 2 fires on a person (see docs/build-log.md).
    """
    if not wanted:                                  # empty list => any class
        return dets
    wanted_ids = set()
    for raw in wanted:
        name = raw.strip().lower()
        if labels:
            for i, lab in enumerate(labels):
                if lab.strip().lower() == name:
                    wanted_ids.add(i)
        elif name == "person":
            wanted_ids.add(2)
        else:
            try:
                wanted_ids.add(int(name))
            except ValueError:
                pass
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
        # keep input size tied to what the model actually declares
        self.input_size = int(self.inp["shape"][1])
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

        # JSONL event log
        if args.log:
            with open(args.log, "a") as fh:
                fh.write(f'{{"ts": {time.time():.3f}, "type": "boot", '
                         f'"source": "{args.stream or args.image}"}}\n')

        # MQTT
        self.mqtt_client = None
        if args.mqtt_broker:
            try:
                import paho.mqtt.client as mqtt
            except ImportError:
                sys.exit("--mqtt-broker needs paho-mqtt: pip install paho-mqtt")
            self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.mqtt_client.connect(args.mqtt_broker, args.mqtt_port, 30)
            self.mqtt_client.loop_start()
            # remote arm/disarm + status queries ride on the control topic
            self.mqtt_client.on_message = self._on_control
            self.mqtt_client.subscribe(getattr(args, "mqtt_control_topic",
                                               "intrusion/control"))

        # Telegram (env-driven; silently off unless both vars present)
        self.telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.telegram_chat = os.environ.get("TELEGRAM_CHAT_ID")
        if self.telegram_token and self.telegram_chat:
            print("telegram alerts enabled")
        elif self.telegram_token or self.telegram_chat:
            print("warning: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID must both be set")

    # -- event log -------------------------------------------------------- #
    def log_line(self, entry):
        if self.args.log:
            with open(self.args.log, "a") as fh:
                fh.write(json.dumps(entry) + "\n")

    # -- mqtt ------------------------------------------------------------- #
    def mqtt_publish(self, payload):
        if self.mqtt_client:
            self.mqtt_client.publish(self.args.mqtt_topic, json.dumps(payload))

    # -- telegram --------------------------------------------------------- #
    def _telegram_url(self, method):
        return (f"https://api.telegram.org/bot{self.telegram_token}/{method}")

    def telegram_send_photo(self, caption, photo_path):
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

    def telegram_send_text(self, text):
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
            cmd = (str(data.get("command") or data.get("cmd") or "").lower()
                   if isinstance(data, dict) else str(data).lower())
        except ValueError:
            cmd = raw.lower()
        now = time.time()
        if cmd in ("arm", "on", "enable", "resume"):
            self.armed = True
            action = "armed"
        elif cmd in ("disarm", "off", "disable", "pause"):
            self.armed = False
            action = "disarmed"
        else:
            action = None                       # "status" / "state" / "?"
        self.last_command = {"cmd": cmd or "status", "ts": round(now, 3)}
        self.want_status = True
        if action:
            print(f"[control] {action} ({msg.topic})", flush=True)

    def publish_status(self, payload):
        if self.mqtt_client:
            self.mqtt_client.publish(self.args.mqtt_status_topic,
                                     json.dumps(payload), retain=True)

    # -- events ----------------------------------------------------------- #
    def fire_start(self, dets, ts, snapshot_path=None):
        payload = {"type": "start", "ts": round(ts, 3),
                   "detections": dets_to_json(dets),
                   "snapshot": snapshot_path}
        self.mqtt_publish(payload)
        self.log_line(payload)
        names = ", ".join(f"{class_name(d, self.labels)} {d['score']:.2f}"
                          for d in dets)
        caption = (f"🚨 Intrusion detected ({len(dets)}): {names}\n"
                   f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
        if snapshot_path and os.path.exists(snapshot_path):
            self.telegram_send_photo(caption, snapshot_path)
        else:
            self.telegram_send_text(caption)

    def fire_end(self, dets_last, ts, start_ts):
        payload = {"type": "end", "ts": round(ts, 3),
                   "start": round(start_ts, 3),
                   "duration": round(ts - start_ts, 2),
                   "detections": dets_to_json(dets_last)}
        self.mqtt_publish(payload)
        self.log_line(payload)
        self.telegram_send_text(
            f"✅ All clear — alarm lasted {ts - start_ts:.1f}s")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

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
    p.add_argument("--log", default="events.jsonl",
                   help="JSONL event log (default events.jsonl)")
    p.add_argument("--quiet-after", type=float, default=2.0,
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

    labels = load_labels(args.labels)
    if labels:
        print(f"labels: {len(labels)} classes")
    wanted = [c for c in args.classes.split(",") if c.strip()] if args.classes else []
    if wanted:
        print(f"filter: {args.classes}")

    notify = Notifier(args, labels)
    print(f"loading model {args.model} ...")
    det = Detector(args.model, args.threshold, labels, wanted)

    os.makedirs(args.snapshot_dir, exist_ok=True) if args.snapshot_dir else None

    # ------------------------------------------------------------------ #
    # single image mode
    # ------------------------------------------------------------------ #
    if args.image:
        if not os.path.exists(args.image):
            sys.exit(f"image not found: {args.image}")
        frame = cv2.imread(args.image)
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
    cap = cv2.VideoCapture(args.stream)
    if not cap.isOpened():
        sys.exit(f"cannot open stream: {args.stream}")
    print(f"watching {args.stream} (Ctrl+C to stop)")

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

    def maybe_end(now):
        """Close the current event once the quiet grace period has passed."""
        nonlocal active, last_event_end, event_start_ts
        if active and (now - last_seen_ts) >= args.quiet_after:
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
            "fps": round(fps, 1),
            "uptime": round(now - boot_ts, 1),
            "last_event": last_event_ts,
            "last_command": notify.last_command,
        }

    def flush_state(now):
        if args.state_file:
            tmp = args.state_file + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(sentry_state(now), fh)
            os.replace(tmp, args.state_file)
        if notify.mqtt_client:
            notify.publish_status(sentry_state(now))

    def process(frame, now):
        nonlocal active, event_start_ts, last_seen_ts, last_event_end, fps, prev_armed, last_event_ts
        dt = now - prev if now > prev else 1e-6
        fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)

        armed = notify.armed
        if prev_armed is not None and armed != prev_armed:
            if armed:
                print("[control] re-armed — fresh watch", flush=True)
                last_event_end = -999999.0
            else:
                print("[control] disarmed — events suppressed", flush=True)
            active = False
            event_start_ts = None
        prev_armed = armed

        dets = det.detect(frame)
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
                        snap = os.path.join(
                            args.snapshot_dir,
                            time.strftime("%Y%m%d-%H%M%S") + ".jpg")
                        cv2.imwrite(snap, draw(frame.copy(), dets, labels,
                                               active=True, armed=True))
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
                # stream hiccup / EOF: still close events gracefully
                maybe_end(now)
            else:
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
            notify.mqtt_client.loop_stop()


if __name__ == "__main__":
    main()