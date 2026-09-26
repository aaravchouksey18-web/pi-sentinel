#!/usr/bin/env python3
"""pi-intrusion-detection — YOLOv5 (TFLite) person/intrusion detector.

Consolidated and cleaned up from the project's earlier iterations
(testcode-v18/v19/v20.py): the rough single-output decoder is replaced with
a proper YOLOv5 decode + NVIDIA-free numpy NMS, frame handling uses
letterboxing instead of naive resize, streams/params are configurable via
args (no hardcoded IPs), and a headless mode publishes alerts to MQTT and
writes annotated snapshots without needing a display.

Run modes
---------
    # single image (self-test, no camera needed)
    python3 intrusion.py --image intrusion.jpg --model best-fp16.tflite

    # live stream (webcam or ESP32-CAM MJPEG URL)
    python3 intrusion.py --stream 0
    python3 intrusion.py --stream "http://<camera-ip>:81/stream"

    # headless intrusion watcher + MQTT alerts + snapshots
    python3 intrusion.py --stream "http://<camera-ip>:81/stream" \
        --headless --mqtt-broker 127.0.0.1 --mqtt-topic intrusion/detections \
        --snapshot-dir snapshots
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# helpers
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
    input), objectness, then one score per class.
    Returns detections in MODEL-SPACE pixels, score >= threshold.
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
    "person" falls back to class id 2 — measured on this model:
    class index 2 fires on a person (see docs/build-log.md).
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


def draw(frame, dets, labels):
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in (d["x1"], d["y1"], d["x2"], d["y2"]))
        name = (labels[d["class_id"]]
                if labels and d["class_id"] < len(labels)
                else f"class-{d['class_id']}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {d['score']:.2f}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


def dets_to_json(dets, ts):
    return {
        "ts": ts,
        "detections": [{
            "class_id": d["class_id"],
            "score": round(d["score"], 3),
            "bbox": [round(v, 1) for v in (d["x1"], d["y1"], d["x2"], d["y2"])],
        } for d in dets],
    }


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


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #

class Detector:
    def __init__(self, model_path, input_size=640, threshold=0.5,
                 labels=None, wanted=None):
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
# CLI
# --------------------------------------------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(
        description="YOLOv5 TFLite intrusion detector (display / headless / image).")
    p.add_argument("--model", default="best-fp16.tflite",
                   help="path to the .tflite model")
    p.add_argument("--labels", default=None,
                   help="optional labels file, one class name per line")
    p.add_argument("--input-size", type=int, default=640,
                   help="model input side (default 640)")
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
    p.add_argument("--no-display", "--headless", dest="headless", action="store_true",
                   help="never show a window")
    p.add_argument("--snapshot-dir", default=None,
                   help="save annotated snapshots of frames with detections here")
    p.add_argument("--snapshot-interval", type=float, default=5.0,
                   help="min seconds between snapshots (default 5)")
    p.add_argument("--mqtt-broker", default=None, help="MQTT broker host (alerts)")
    p.add_argument("--mqtt-port", type=int, default=1883)
    p.add_argument("--mqtt-topic", default="intrusion/detections")
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

    print(f"loading model {args.model} ...")
    det = Detector(args.model, args.input_size, args.threshold, labels, wanted)

    mqtt_client = None
    if args.mqtt_broker:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            sys.exit("--mqtt-broker needs paho-mqtt: pip install paho-mqtt")
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        mqtt_client.connect(args.mqtt_broker, args.mqtt_port, 30)
        mqtt_client.loop_start()

    os.makedirs(args.snapshot_dir, exist_ok=True) if args.snapshot_dir else None
    last_snap = 0.0

    def handle(frame, ts):
        nonlocal last_snap
        dets = det.detect(frame)
        annotated = draw(frame.copy(), dets, labels) if dets else frame
        if dets:
            print(f"[{ts}] {len(dets)} detection(s): " +
                  ", ".join(f"id={d['class_id']} {d['score']:.2f}" for d in dets),
                  flush=True)
        if dets and args.snapshot_dir and ts - last_snap >= args.snapshot_interval:
            path = os.path.join(args.snapshot_dir,
                                time.strftime("%Y%m%d-%H%M%S") + ".jpg")
            cv2.imwrite(path, annotated)
            print(f"snapshot: {path}")
            last_snap = ts
        if dets and mqtt_client:
            mqtt_client.publish(args.mqtt_topic,
                                json.dumps(dets_to_json(dets, ts)))
        return annotated

    # --- single image mode ------------------------------------------------ #
    if args.image:
        if not os.path.exists(args.image):
            sys.exit(f"image not found: {args.image}")
        frame = cv2.imread(args.image)
        out = handle(frame, time.time())
        out_path = args.output or "annotated.jpg"
        cv2.imwrite(out_path, out)
        print(f"wrote {out_path}")
        return

    # --- live stream mode ------------------------------------------------- #
    cap = cv2.VideoCapture(args.stream)
    if not cap.isOpened():
        sys.exit(f"cannot open stream: {args.stream}")
    print(f"watching {args.stream} (Ctrl+C to stop)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.5)
                continue
            annotated = handle(frame, time.time())
            if not args.headless:
                cv2.imshow("Intrusion Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if mqtt_client:
            mqtt_client.loop_stop()


if __name__ == "__main__":
    main()