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
                                                              |-- snapshot / event log
                                                              +-- dashboard.py web UI (:8001)
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
sudo cp deploy/pi-intrusion.env.example /etc/pi-intrusion.env && \
    sudo chown root:root /etc/pi-intrusion.env && sudo chmod 600 /etc/pi-intrusion.env
sudo nano /etc/pi-intrusion.env        # fill tokens
sudo nano /etc/systemd/system/pi-intrusion-detection.service   # set <CAMERA_URL>,
    # User= and the %h/... paths to your setup
sudo systemctl daemon-reload
sudo systemctl enable --now pi-intrusion-detection
```

Follow it with `journalctl -u pi-intrusion-detection -f`. The unit
`EnvironmentFile`s the tokens (owned by root, chmod 600), keeps them out of
the repo, and restarts the watcher on failure — including when the camera
stalls for 30 consecutive reads (~6 s) so a dead stream self-heals.

## Web dashboard

A tiny stdlib-only companion UI that reads the event log and serves the
snapshots — no framework, no CDN, no secrets:

```sh
python3 dashboard.py --log events.jsonl --snapshot-dir snapshots --port 8001
# open http://<pi-ip>:8001
```

Live status pill (**ARMED / ACTIVE / STALE / DISARMED / OFFLINE /
STREAM-ERR**), last event, a snapshot grid, and recent history —
auto-refreshes every 2.5 s. When run against the sentry's `state.json` it
also shows its uptime, fps and last remote command, a **DISARMED** pill
whenever the sentry has been told to go quiet, **STREAM-ERR** when the
detector reports the camera is not producing frames, and **OFFLINE** when
the heartbeat goes stale. Read-only by design. It serves camera imagery,
so pass `--token` (or put a reverse proxy in front) whenever it's reachable
beyond your own machines. Run it next to the detector; systemd unit:
`deploy/pi-intrusion-dashboard.service` (port 8001, right next to
presence-vigil's :8000).

| flag | default | meaning |
|---|---|---|
| `--log` | `events.jsonl` | event log to read |
| `--snapshot-dir` | — | where the detector saves snapshots |
| `--port` / `--bind` | `8001` / `0.0.0.0` | HTTP listen address |
| `--token` | — | access token; every request needs `?t=<token>` or `Authorization: Bearer <token>` |
| `--active-timeout` | `120` | an unresolved START older than this shows **STALE** (detector probably down) |
| `--heartbeat-timeout` | `20` | `state.json` older than this shows **OFFLINE** (sentry heartbeat runs every ~10 s) |
| `--state-file` | `state.json` | sentry `state.json` — makes the pill show **DISARMED/OFFLINE/STREAM-ERR** and fills uptime/fps/last-cmd |

## MQTT payloads

Topic: `intrusion/events` (default). JSON per event:

```json
{"type": "start", "ts": 1790421234.56, "detections": [
   {"class_id": 2, "score": 0.97, "bbox": [12.0, 40.0, 200.0, 235.0]}],
 "snapshot": "snapshots/20260926-150000-244.jpg"}
{"type": "end", "ts": 1790421238.12, "start": 1790421234.56,
 "duration": 3.56, "detections": []}
```

## MQTT control & status

The sentry listens on `intrusion/control` (default) for arm / disarm
commands — plain strings or JSON:

```sh
mosquitto_pub -h 127.0.0.1 -t intrusion/control -m disarm      # go quiet
mosquitto_pub -h 127.0.0.1 -t intrusion/control -m arm
mosquitto_pub -h 127.0.0.1 -t intrusion/control -m '{"command":"status"}'
```

Accepted commands: `arm | on | enable | resume`, `disarm | off | disable |
pause`, and `status | state | ?` (republish state immediately). While
disarmed no events or alerts fire, but the sentry keeps processing and
drawing; re-arming starts a fresh watch immediately. The subscription is
re-established on every MQTT connect via an `on_connect` handler, so a
broker restart (or WiFi blip) can never silently kill remote control.

Every `--status-interval` seconds it publishes a **retained** heartbeat to
`intrusion/status` and writes `state.json` (gitignored) so the dashboard
shows the true sentry state — including a **DISARMED** pill — instead of
guessing from the event log:

```json
{"ts": 1790421234.5, "armed": false, "active": false, "fps": 0.5,
 "uptime": 1234.5, "stream_ok": true, "online": true,
 "last_event": null, "last_command": {"cmd": "disarm", "ts": 1790421230.1}}
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
| `--output` | — | single-shot mode: write the annotated image here |
| `--headless` | off | never open a window |
| `--snapshot-dir` | — | save an annotated snapshot per intrusion START |
| `--log` | `events.jsonl` | JSONL event log (append-only) |
| `--quiet-after` | `6.0` | no-detection grace before an event ends (s) |
| `--event-cooldown` | `30.0` | min seconds between separate events |
| `--mqtt-broker` | — | enable MQTT events (needs `paho-mqtt>=2.0`) |
| `--mqtt-port` | `1883` | MQTT broker port |
| `--mqtt-control-topic` | `intrusion/control` | topic to arm/disarm the sentry |
| `--mqtt-status-topic` | `intrusion/status` | retained status heartbeat (MQTT) |
| `--status-interval` | `10.0` | seconds between state flush + heartbeat |
| `--state-file` | `state.json` | sentry state JSON (read by the dashboard) |
| `--start-disarmed` | off | boot disarmed until armed via MQTT |

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
lighter `tflite-runtime`. `paho-mqtt>=2.0` is only needed when
`--mqtt-broker` is used (the code uses the paho 2.x API);
Telegram uses only the Python standard library.

## Notes

- **No personal data in this repo** — camera IPs are arguments, Telegram
  tokens live in env/`EnvironmentFile`, the Twilio secrets from an earlier
  iteration were never carried over, and model binaries/snapshots/event logs
  are gitignored.
- Detection boxes are drawn in the original frame coordinates (letterbox
  scaling is undone); the decode handles the YOLOv5 single-output format
  directly, no external detector libraries.
- Stream failures are handled loudly: a dropped source closes an in-progress
  event with `"reason": "stream_lost"`, the loop backs off instead of
  spinning the CPU, `state.json` flips `stream_ok` to false (the dashboard
  briefly shows **STREAM-ERR**, then **OFFLINE** once the file goes stale),
  and 30 consecutive failed reads (~6 s) exit so systemd restarts the
  watcher.
- The detector publishes a **retained** heartbeat on `intrusion/status`
  (`online: true`) and flips it to `online: false` on graceful shutdown, so
  other MQTT consumers can watch for a dead sentry. The dashboard itself
  does not subscribe to MQTT — it reads `state.json`, and flags **OFFLINE**
  once that file is older than `--heartbeat-timeout`.
- Display mode needs a GUI-capable OpenCV: the `requirements.txt` build
  ships `opencv-python-headless` (for the headless service), which has no
  `cv2.imshow` — install `opencv-python` instead to use the live window.
- Throughput on the Pi: this float fp16 model runs at roughly **0.5–1 fps**
  on a Pi 4-class CPU (measured ≈1.9 s/frame with the TFLite XNNPACK
  delegate, plus an ≈18 s cold model load). That's fine for an intrusion
  watcher's START/END lifecycle. For higher rates, export an int8-quantized
  YOLOv5 model — several times faster, same code.
- See `docs/build-log.md` for the consolidation history, model probes, and
  the spruce pass.