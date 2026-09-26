# build log

## Consolidation — 2026-09-26

Found in `an archive of the original Pi camera test scripts` as `testcode-v18/19/20.py`,
three generations of the same detector:

- **v18** — TensorFlow SavedModel (SSD MobileNet COCO, 300×300), plain
  person detection, ESP32-CAM stream over MJPEG (`http://<camera-ip>:81/stream`).
- **v19** — moved to YOLOv5 TFLite (`best-fp16.tflite`, 640×640) and added
  Twilio WhatsApp alerts. Contained a **live Twilio account SID** and phone
  numbers in source — those were **never carried into this repo**.
- **v20** — dropped Twilio, cleaned the stream URL, but kept a guessed
  single-output decoder (`[1, 25200, 10]`, confidence at column 4).

### Model probe (ran on the Pi, TF 2.18)

- input `[1 640 640 3]` float32, output `[1 25200 10]` float32 — a custom
  **5-class YOLO export**. v20's shape guess was right.
- Ran `intrusion.jpg` through the model: top detections at ~(0.92, 0.49)
  with obj 0.60 and **class index 2 @ 0.968** → class 2 is the person class.
  (The 240×240 test frame is square, so it also confirmed the plain-resize
  calibration from the original scripts works on square feeds; letterboxing
  is used for odd aspect ratios and undone when drawing.)

### Cleanup decisions

- One script (`intrusion.py`) replaces all three versions:
  - args for stream/model/threshold/classes — nothing hardcoded, no IPs in
    the repo;
  - real YOLOv5 single-output decode + class filter + numpy NMS (no external
    detector libs);
  - letterbox instead of naive resize, boxes mapped back to frame coords;
  - **single-image mode** (`--image`) so it's testable without a camera;
  - **headless mode**: no window + MQTT alert payloads + annotated snapshots.
- `labels.example.txt` documents the observed class-2-person mapping; the
  decoder derives class count from output columns so any N-class export works.
- Models (`*.tflite`, `saved_model.pb`), snapshots, and output images are
  gitignored. Source artefacts (`research/`, `tensorflow_models/`) also
  never enter the repo — they ship with the TF models zoo, not this project.

## Spruce pass — 2026-09-26

Turned the single script into an event-driven sentry:

- **Event engine** — alerts/snapshots/MQTT/log fire on START/END transitions
  instead of every frame. `--quiet-after` (default 2.0 s) ends an event once
  the frame goes quiet; `--event-cooldown` (default 30.0 s) re-arms it so a
  person walking back past the camera doesn't re-trigger spam.
- **Telegram alerts** — free, stdlib-only (multipart `sendPhoto` via
  urllib): START sends text + the annotated snapshot, END sends a clear
  message. Configured purely through `TELEGRAM_BOT_TOKEN` /
  `TELEGRAM_CHAT_ID` env vars — the successor to v19's paid Twilio idea,
  minus the secrets.
- **JSONL event log** (`events.jsonl`, gitignored) — boot/start/end lines for
  an audit trail.
- **systemd unit** — `deploy/pi-intrusion-detection.service` +
  `deploy/pi-intrusion.env.example` document always-on headless operation.
- **Overlay polish** — ARMED/ACTIVE status, wall-clock, FPS, and a
  confidence bar under each box.
- **Robustness** — end-of-event logic also runs when the stream drops/EOF,
  so a camera crash ends the event rather than leaving it stuck active.
  Fixed during first test: missing `urllib.parse` import for the Telegram
  text fallback.
- Verified on the Pi: image mode still detects person id 2; synthetic
  video (black → real intrusion frames → black) produced a clean
  START → snapshot → END against the live broker, with matching JSONL rows.
- **Measured throughput** (worth planning around): cold model load ≈ 18 s,
  inference ≈ 1.9 s/frame (≈0.5 fps) with the float fp16 model via the
  XNNPACK delegate on this Pi. An int8-quantized export would restore
  real-time review rates. Debugging note: earlier "END never fired" runs
  turned out to be test-timing artifacts — the 55-frame test video needed
  ~100 s of frames but the runs were killed at 15–45 s, before the quiet
  segment was reached.