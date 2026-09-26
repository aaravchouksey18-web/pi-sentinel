# pi-intrusion-detection

Intrusion detection on a Raspberry Pi: a YOLOv5 (TFLite) model watches a
camera stream, flags people, and acts as an **event-driven sentry** — alerts
fire only when a *new* intrusion starts and ends, not on every frame. Runs
fully **headless** (MQTT alerts + annotated snapshots + JSONL event log), and
ships a systemd unit so it can run as an always-on service.

Consolidated from the project's original iterations (`testcode-v18/19/20.py`)
and spruced up: one script, no hardcoded camera IPs, no secrets in the repo.

```
[ESP32-CAM / webcam] --MJPEG--> intrusion.py --YOLOv5 TFLite--> detections
                                                              |-- window (display mode)
                                                              |-- MQTT events (headless)
                                                              |-- Telegram alert + photo
                                                              +-- snapshot / event log
```

## How events work

The watcher treats an intrusion as a **span**: while detections match the
class filter, the event is `active`; once the frame is quiet for
`--quiet-after` seconds the event `END`s; a new event can start only after
`--event-cooldown` seconds, so re-entering or a person walking past doesn't
spam alerts. MQTT, Telegram, snapshots, and the event log all fire on the
START/END transitions, not on individual frames.

## Quick start

1. **Get the model into place** (gitignored on purpose — it's a binary
   artifact): copy your YOLOv5 TFLite export here as `best-fp16.tflite` (or
   pass `--model /path/to/model.tflite`).

2. **Self-test with a single image** (no camera needed):

   ```sh
   python3 intrusion.py --image test/intrusion.jpg
   ```

3. **Live stream** — webcam or ESP32-CAM MJPEG:

   ```sh
   python3 intrusion.py --stream 0                                  # webcam
   python3 intrusion.py --stream "http://<camera-ip>:81/stream"     # ESP32-CAM
   ```

4. **Headless sentry** — MQTT events + snapshots, no display:

   ```sh
   python3 intrusion.py --stream "http://<camera-ip>:81/stream" \
       --headless \
       --mqtt-broker 127.0.0.1 --mqtt-topic intrusion/events \
       --snapshot-dir snapshots
   ```

5. **Telegram alerts** (text + annotated photo on START, "all clear" on END):

   ```sh
   export TELEGRAM_BOT_TOKEN="<token from @BotFather>"
   export TELEGRAM_CHAT_ID="<your chat id>"
   python3 intrusion.py --stream "http://<camera-ip>:81/stream" --headless
   ```

## Run as a service (systemd)

```sh
sudo cp deploy/pi-intrusion-detection.service /etc/systemd/system/
sudo cp deploy/pi-intrusion.env.example /etc/pi-intrusion.env
sudo nano /etc/pi-intrusion.env        # fill tokens
sudo nano /etc/systemd/system/pi-intrusion-detection.service   # set <CAMERA_URL>
sudo systemctl daemon-reload
sudo systemctl enable --now pi-intrusion-detection
```

Follow it with `journalctl -u pi-intrusion-detection -f`. The unit
`EnvironmentFile`s the tokens, keeps them out of the repo, and restarts the
watcher on failure.

## MQTT payloads

Topic: `intrusion/events` (default). JSON per event:

```json
{"type": "start", "ts": 1790421234.56, "detections": [
   {"class_id": 2, "score": 0.97, "bbox": [12.0, 40.0, 200.0, 235.0]}],
 "snapshot": "snapshots/20260926-150000.jpg"}
{"type": "end", "ts": 1790421238.12, "start": 1790421234.56,
 "duration": 3.56, "detections": []}
```

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
| `--snapshot-dir` | — | save an annotated snapshot per intrusion START |
| `--log` | `events.jsonl` | JSONL event log (append-only) |
| `--quiet-after` | `2.0` | no-detection grace before an event ends (s) |
| `--event-cooldown` | `30.0` | min seconds between separate events |
| `--mqtt-broker` | — | enable MQTT events (needs `paho-mqtt`) |

Telegram is configured exclusively through `TELEGRAM_BOT_TOKEN` /
`TELEGRAM_CHAT_ID` env vars (see `deploy/pi-intrusion.env.example`).

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
used; Telegram uses only the Python standard library.

## Notes

- **No personal data in this repo** — camera IPs are arguments, Telegram
  tokens live in env/`EnvironmentFile`, the Twilio secrets from an earlier
  iteration were never carried over, and model binaries/snapshots/event logs
  are gitignored.
- Detection boxes are drawn in the original frame coordinates (letterbox
  scaling is undone); the decode handles the YOLOv5 single-output format
  directly, no external detector libraries.
- Stream hiccups are handled: if the source drops frames, the current event
  still ends after `--quiet-after` instead of hanging.
- Throughput on the Pi: this float fp16 model runs at roughly **0.5–1 fps**
  on a Pi 4-class CPU (measured ≈1.9 s/frame with the TFLite XNNPACK
  delegate, plus an ≈18 s cold model load). That's fine for an intrusion
  watcher's START/END lifecycle. For higher rates, export an int8-quantized
  YOLOv5 model — several times faster, same code.
- See `docs/build-log.md` for the consolidation history, model probes, and
  the spruce pass.