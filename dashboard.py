#!/usr/bin/env python3
"""pi-intrusion-detection dashboard — a tiny web UI for the sentry.

Dependency-free (Python stdlib only): it reads the JSONL event log the
detector appends to, serves the annotated snapshots it saves, and exposes
a small JSON API that the single-page UI polls every couple of seconds.

    python3 dashboard.py --log events.jsonl --snapshot-dir snapshots \
        --port 8001

Open http://<pi-ip>:8001 in a browser. Read-only by design — no POST
endpoints and no secrets — but it DOES serve camera imagery, so a token
(--token) is required whenever the dashboard is reachable beyond your
own machines. Run it behind a reverse proxy if you want TLS.

Status is derived from the log plus the sentry's state.json:
    ARMED      - no unresolved START
    ACTIVE     - the last event is a START with no END yet
    STALE      - ACTIVE but that START is older than --active-timeout
    DISARMED   - the sentry says it is disarmed (state.json)
    OFFLINE    - state.json is older than --heartbeat-timeout
    STREAM-ERR - the sentry reports the camera is not producing frames
"""

import argparse
import glob
import hmac
import html
import ipaddress
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

_TOKEN_JS = "__TOKEN_JS__"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>intrusion — pi sentry</title>
<style>
  :root { --bg:#0d1117; --card:#161b22; --fg:#e6edf3; --muted:#8b949e;
          --ok:#3fb950; --bad:#f85149; --warn:#d29922; --brd:#30363d; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif;
         background:var(--bg); color:var(--fg); }
  header { display:flex; align-items:center; gap:14px; padding:14px 20px;
           border-bottom:1px solid var(--brd); background:var(--card); }
  header h1 { font-size:18px; margin:0; }
  #clock { color:var(--muted); margin-left:auto; }
  .pill { padding:5px 14px; border-radius:999px; font-weight:600;
          text-transform:uppercase; font-size:12px; letter-spacing:.08em; }
  .ARMED { background:#10291a; color:var(--ok); border:1px solid var(--ok); }
  .ACTIVE { background:#33120f; color:var(--bad); border:1px solid var(--bad);
            animation:pulse 1.1s ease-in-out infinite; }
  .STALE  { background:#2b2306; color:var(--warn); border:1px solid var(--warn); }
  .DISARMED { background:#1c2026; color:#9aa7b4; border:1px dashed #55606b; }
  .OFFLINE { background:#1f1412; color:#ff9b72; border:1px solid #ff9b72; }
  .STREAM-ERR { background:#2b2306; color:var(--warn); border:1px solid var(--warn); }
  @keyframes pulse { 50% { opacity:.45; } }
  main { display:grid; grid-template-columns: minmax(280px, 340px) 1fr;
         gap:16px; padding:16px 20px; }
  @media (max-width: 860px) { main { grid-template-columns: 1fr; } }
  .card { background:var(--card); border:1px solid var(--brd);
          border-radius:10px; padding:14px; }
  .card h2 { margin:0 0 10px; font-size:13px; color:var(--muted);
             text-transform:uppercase; letter-spacing:.08em; }
  .kv { display:grid; grid-template-columns: 92px 1fr; gap:4px 10px; }
  .kv b { color:var(--muted); font-weight:500; }
  #lastsev { font-size:15px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--brd); }
  th { color:var(--muted); font-weight:500; }
  tr.hot td { color:var(--bad); }
  .shots { display:grid; grid-template-columns:repeat(auto-fill,minmax(120px,1fr));
           gap:8px; margin-top:8px; }
  .shots a { display:block; border:1px solid var(--brd); border-radius:8px;
             overflow:hidden; }
  .shots img { width:100%; display:block; }
  .muted { color:var(--muted); }
  .det { margin:2px 0; font-size:12px; color:var(--muted); }
  footer { padding:10px 20px; color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>pi-intrusion <span class="muted">sentry</span></h1>
  <span id="pill" class="pill">…</span>
  <span id="clock"></span>
</header>
<main>
  <section>
    <div class="card">
      <h2>status</h2>
      <div class="kv">
        <b>state</b><span id="st">…</span>
        <b>uptime</b><span id="up" class="muted">—</span>
        <b>fps</b><span id="sfps" class="muted">—</span>
        <b>last cmd</b><span id="cmd" class="muted">—</span>
        <b>last seen</b><span id="lastseenv" class="muted">—</span>
        <b>events</b><span id="evcount">0</span>
        <b>snapshots</b><span id="shcount">0</span>
      </div>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>last event</h2>
      <div id="lastsev" class="muted">no events yet — watching…</div>
    </div>
  </section>
  <section>
    <div class="card">
      <h2>snapshots</h2>
      <div id="shots" class="shots muted">none saved yet</div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>history</h2>
      <table>
        <thead><tr><th>time</th><th>type</th><th>detections</th><th>duration</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
  </section>
</main>
<footer id="src" class="muted"></footer>
<script>
const TOKEN = __TOKEN_JS__;
const q = () => TOKEN ? "?t=" + encodeURIComponent(TOKEN) : "";
const esc = s => { const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML.replace(/"/g, "&quot;").replace(/'/g, "&#39;"); };
const fmt = ts => ts ? new Date(ts * 1000).toLocaleString() : "—";
async function tick() {
  try {
    const r = await fetch("/api/events" + q());
    const d = await r.json();
    const pill = document.getElementById("pill");
    pill.textContent = d.status; pill.className = "pill " + d.status;
    document.getElementById("st").textContent =
      d.status + (d.age != null ? ` (${d.age.toFixed(0)}s old)` : "");
    document.getElementById("lastseenv").textContent =
      d.last_seen ? fmt(d.last_seen) : "—";
    document.getElementById("evcount").textContent = d.events.length;
    document.getElementById("shcount").textContent = d.snapshots.length;
    const stt = d.state;
    document.getElementById("up").textContent =
      stt ? Math.round(stt.uptime) + "s" : "—";
    document.getElementById("sfps").textContent =
      stt ? stt.fps + " fps" : "—";
    const lc = stt && stt.last_command;
    document.getElementById("cmd").textContent =
      lc ? (lc.cmd +
        (lc.ignored ? " (ignored: " + lc.ignored + ")" : "") +
        " · " + new Date(lc.ts * 1000).toLocaleTimeString()) : "—";
    const lsev = document.getElementById("lastsev");
    if (d.last) {
      const e = d.last;
      lsev.innerHTML = (e.type === "start" ? "🚨 " : "✅ ") +
        esc(e.type.toUpperCase() + " at " + fmt(e.ts)) +
        (e.duration != null ? ` — lasted ${esc(e.duration)}s` : "") +
        (e.detections && e.detections.length
          ? `<div class="det">` + e.detections.map(x =>
              esc(`class ${x.class_id} · ${x.score}`)).join("<br>") + "</div>" : "");
    } else { lsev.textContent = "no events yet — watching…"; lsev.className = "muted"; }
    const shots = document.getElementById("shots");
    shots.className = "shots";
    shots.innerHTML = d.snapshots.slice(0, 12).map(s =>
      `<a href="/snapshots/${encodeURIComponent(s.name)}${q()}" target="_blank">` +
      `<img src="/snapshots/${encodeURIComponent(s.name)}${q()}" alt="${esc(s.name)}">` +
      `<div class="muted" style="padding:3px 6px;font-size:11px">${esc(fmt(s.ts))}</div></a>`
    ).join("") || "none saved yet";
    const rows = document.getElementById("rows");
    rows.innerHTML = d.events.slice().reverse().map(e => {
      const hot = e.type === "start" && d.status === "ACTIVE";
      return `<tr class="${hot ? "hot" : ""}"><td>${esc(fmt(e.ts))}</td>` +
        `<td>${esc(e.type)}</td>` +
        `<td>${e.detections ? e.detections.length : 0}</td>` +
        `<td>${esc(e.duration != null ? e.duration + "s" : "—")}</td></tr>`;
    }).join("");
    document.getElementById("src").textContent =
      `log: ${esc(d.src.log)} · snapshots: ${esc(d.src.snaps)}`;
  } catch { /* keep polling */ }
}
setInterval(tick, 2500); tick();
const ck = () => { document.getElementById("clock").textContent =
                     new Date().toLocaleString(); };
setInterval(ck, 1000); ck();
</script>
</body>
</html>
"""

_SNAP_RE = re.compile(r"\d{8}-\d{6}(?:-\d{3})?\.jpg")


def read_events(path, limit=300, tail_bytes=1 << 20):
    """Return the last `limit` events, reading only the file tail.

    Earlier versions parsed the whole log per request (and per poll); the
    tail read keeps both memory and CPU bounded no matter how big the log
    grows. Non-JSON and non-object lines are skipped.
    """
    out = []
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > tail_bytes:
                fh.seek(size - tail_bytes)
                fh.readline()                # drop the partial first line
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict):
                        out.append(obj)
                except ValueError:
                    continue
    except OSError:
        pass
    return out[-limit:]


def derive_status(events, active_timeout):
    """ARMED / ACTIVE / (STALE, age) from the tail of the log."""
    for e in reversed(events):
        t = e.get("type")
        if t == "start":
            try:
                age = time.time() - float(e.get("ts", 0))
            except (TypeError, ValueError):
                age = None
            if age is None or age > active_timeout:
                return "STALE", age
            return "ACTIVE", age
        if t == "end":
            break
    return "ARMED", None


def list_snapshots(snapshot_dir, limit=40):
    if not snapshot_dir:
        return []
    try:
        hits = sorted(glob.glob(os.path.join(snapshot_dir, "*.jpg")),
                      key=os.path.getmtime, reverse=True)
    except OSError:
        return []
    out = []
    for p in hits[:limit]:
        try:
            out.append({"name": os.path.basename(p),
                        "ts": os.path.getmtime(p)})
        except OSError:
            continue
    return out


def read_state(path):
    """Load the sentry state.json written by intrusion.py, if present."""
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
            # a JSON list (e.g. a truncated/partially-flushed write) is not a
            # valid state dict; skip it instead of crashing on .get()
            return obj if isinstance(obj, dict) else None
    except (OSError, ValueError):
        return None


def make_handler(log_path, snapshot_dir, active_timeout, state_file=None,
                 token=None, heartbeat_timeout=20.0):
    # "</" would close the <script> element: escape it so a token value
    # cannot inject markup (page is token-gated, so this is self-XSS only)
    page = PAGE.replace(_TOKEN_JS,
                        json.dumps(token or "").replace("</", "<\\/"))
    class Handler(BaseHTTPRequestHandler):
        timeout = 10.0                      # no per-connection hangs

        def log_message(self, fmt, *args):  # quieter
            pass

        def _authed(self, qs):
            """True if no token configured, or the request carries it."""
            if not token:
                return True
            t = qs.get("t", [""])[0]
            # compare_digest is ASCII-only for str; encode both sides so a
            # non-ASCII query value is rejected, not raised as an exception
            if t and hmac.compare_digest(t.encode("utf-8"),
                                         token.encode("utf-8")):
                return True
            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                return hmac.compare_digest(auth[7:].encode("utf-8"),
                                           token.encode("utf-8"))
            return False

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)
            if not self._authed(qs):
                self._json({"error": "unauthorized"}, 401)
                return
            if path in ("/", "/index.html"):
                body = page.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/events":
                ev = read_events(log_path)
                state = read_state(state_file)
                status, age = derive_status(ev, active_timeout)
                if state is not None:
                    try:
                        stale = time.time() - float(state.get("ts", 0))
                    except (TypeError, ValueError):
                        stale = None
                    # OFFLINE first: a sentry that died mid-watch (stale heartbeats, or a
                    # graceful shutdown that wrote online:false to
                    # state.json) must not keep showing a calm
                    # DISARMED/ARMED state.
                    if state.get("online") is False:
                        status, age = "OFFLINE", 0.0
                    elif stale is not None and stale > heartbeat_timeout:
                        status, age = "OFFLINE", stale
                    elif state.get("armed") is False:
                        status, age = "DISARMED", None
                    elif state.get("stream_ok") is False:
                        status, age = "STREAM-ERR", None
                    elif state.get("active") is True:
                        # A fresh heartbeat that says the sentry is watching
                        # right now means a long unresolved START is a
                        # loitering intruder, not an orphaned event — keep it
                        # ACTIVE (STALE is for a heartbeat that is stale too).
                        status, age = "ACTIVE", 0.0
                last_seen = None
                if ev:
                    last_seen = ev[-1].get("ts", time.time())
                self._json({
                    "status": status, "age": age,
                    "last_seen": last_seen,
                    "last": ev[-1] if ev else None,
                    "events": ev, "snapshots": list_snapshots(snapshot_dir),
                    "state": state,
                    "src": {
                        "log": os.path.basename(log_path) if log_path else "(none)",
                        "snaps": (os.path.basename(snapshot_dir)
                                  if snapshot_dir else "(none)"),
                    },
                })
                return
            if path.startswith("/snapshots/"):
                name = os.path.basename(unquote(path[len("/snapshots/"):]))
                # strict name check: only detector snapshot names, no
                # "../", no dequoting tricks, no symlinked files
                if not _SNAP_RE.fullmatch(name):
                    self._json({"error": "bad name"}, 400)
                    return
                if snapshot_dir:
                    real_dir = os.path.realpath(snapshot_dir)
                    real_path = os.path.realpath(os.path.join(real_dir, name))
                    if not real_path.startswith(real_dir + os.sep):
                        self._json({"error": "bad name"}, 400)
                        return
                    try:
                        with open(real_path, "rb") as fh:
                            body = fh.read()
                    except OSError:
                        self._json({"error": "not found"}, 404)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Content-Disposition",
                                     "inline; filename=" + name)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self._json({"error": "no snapshot dir"}, 404)
                return
            if path == "/healthz":
                self._json({"ok": True})
                return
            self._json({"error": "not found"}, 404)

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description="intrusion sentry dashboard")
    p.add_argument("--log", default="events.jsonl")
    p.add_argument("--snapshot-dir", default=None)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--allow-open", action="store_true",
                   help="serve with no token on a non-loopback bind "
                        "(exposes camera imagery to the LAN)")
    p.add_argument("--token", default=os.environ.get("DASH_TOKEN"),
                   help="optional access token; required as ?t=<token> or "
                        "Authorization: Bearer <token> on every request "
                        "('--token' overrides the DASH_TOKEN env var)")
    p.add_argument("--active-timeout", type=float, default=120.0,
                   help="seconds before an unresolved START is STALE")
    p.add_argument("--heartbeat-timeout", type=float, default=20.0,
                   help="state.json older than this shows OFFLINE "
                        "(detector heartbeat runs every ~10 s)")
    p.add_argument("--state-file", default="state.json",
                   help="sentry state.json written by intrusion.py")
    args = p.parse_args(argv)
    if not args.allow_open and not args.token:
        try:
            loopback = (ipaddress.ip_address(args.bind).version == 4
                        and ipaddress.ip_address(args.bind).is_loopback)
        except ValueError:
            loopback = args.bind == "localhost"
        if not loopback:
            p.error("refusing to serve an open dashboard on a non-loopback "
                    "bind; set --token, bind 127.0.0.1, or pass --allow-open")
    handler = make_handler(args.log, args.snapshot_dir, args.active_timeout,
                           args.state_file, token=args.token,
                           heartbeat_timeout=args.heartbeat_timeout)
    srv = ThreadingHTTPServer((args.bind, args.port), handler)
    auth = f", token={'on' if args.token else 'off'}"
    print(f"dashboard on http://{args.bind}:{args.port}  "
          f"(log={args.log}, snaps={args.snapshot_dir}{auth})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()