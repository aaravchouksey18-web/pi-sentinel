# pi-intrusion-detection

Intrusion detection on a Raspberry Pi: a YOLOv5 (TFLite) model watches a
camera stream, flags people, and can run fully **headless** — publishing
alerts over MQTT and saving annotated snapshots with no display attached.

This is the consolidated, cleaned-up version of the project's original
iterations (`testcode-v18/19/20.py`). One script, no hardcoded camera IPs,
no secrets.

```
[ESP32-CAM / webcam] --MJPEG--> intrusion.py --YOLOv5 TFLite--> detections
                                                              |-- window (display mode)
                                                              |-- MQTT alerts (headless)
                                                              +-- annotated snapshots
```

## Quick start

1. **Get the model into place** (gitignored on purpose — it's a binary
   artifact): copy your YOLOv5 TFLite export to this directory and name it
   `best-fp16.tflite` (or pass `--model /path/to/model.tflite`).

2. **Self-test with a single image** (no camera needed):

   ```sh
   python3 intrusion.py --image test/intrusion.jpg
   ```

3. **Live stream** — webcam or ESP32-CAM MJPEG:

   ```sh
   python3 intrusion.py --stream 0                                  # webcam
   python3 intrusion.py --stream "http://<camera-ip>:81/stream"     # ESP32-CAM
   ```

4. **Headless watcher with MQTT + snapshots:**

   ```sh
   python3 intrusion.py --stream "http://<camera-ip>:81/stream" \
       --headless \
       --mqtt-broker 127.0.0.1 --mqtt-topic intrusion/detections \
       --snapshot-dir snapshots
   ```

MQTT payloads are JSON: `{"ts": ..., "detections": [{"class_id": 2,
"score": 0.97, "bbox": [x1, y1, x2, y2]}, ...]}` on each frame that has one.

## Options

| flag | default | meaning |
|---|---|---|
| `--model` | `best-fp16.tflite` | path to the TFLite model |
| `--labels` | — | labels file, one class name per line (see below) |
| `--threshold` | `0.5` | min object × class score |
| `--classes` | `person` | comma-separated classes to keep; empty = any |
| `--stream` | — | video source: integer index or MJPEG/RTSP URL |
| `--image` | — | single-shot mode: process one image file once |
| `--headless` | off | never open a window |
| `--snapshot-dir` | — | save annotated snapshots here |
| `--mqtt-broker` | — | enable MQTT alerts (needs `paho-mqtt`) |

## The model & class labels

The shipped model is a custom **5-class YOLOv5 export** (output shape
`[1, 25200, 10]` = 5 box/objectness + 5 class scores). Verified while
testing: **class index 2 fires on people** (intrusion.jpg → obj 0.60,
class 0.968). The decoder derives the class count from the output columns,
so this code works with any N-class YOLOv5 export — just supply a labels
file (`--labels labels.txt`, one name per line, line 0 = class 0, ...).
`labels.example.txt` ships with the observed mapping.

## Requirements

```sh
pip install -r requirements.txt        # numpy, opencv-python, tensorflow, paho-mqtt
```

`tensorflow` includes the TFLite interpreter; on a Pi you can substitute the
lighter `tflite-runtime`. `paho-mqtt` is only needed when `--mqtt-broker` is
used.

## Notes

- **No personal data in this repo** — camera IPs are arguments, the Twilio
  secrets from an earlier iteration were never carried over, and model
  binaries/snapshots are gitignored.
- Detection boxes are drawn in the original frame coordinates (letterbox
  scaling is undone); the decode handles the YOLOv5 single-output format
  directly, no external detector libraries.
- See `docs/build-log.md` for the consolidation history and model probes.