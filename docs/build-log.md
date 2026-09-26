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