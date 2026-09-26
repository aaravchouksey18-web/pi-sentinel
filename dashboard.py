#!/usr/bin/env python3
"""pi-intrusion-detection dashboard — a tiny web UI for the sentry.

Dependency-free (Python stdlib only): it reads the JSONL event log the
detector appends to, serves the annotated snapshots it saves, and exposes
a small JSON API that the single-page UI polls every couple of seconds.

    python3 dashboard.py --log events.jsonl --snapshot-dir snapshots \
        --port 8001

Open http://<pi-ip>:8001 in a browser. Read-only by design — no secrets,
no POST endpoints; safe to expose on your LAN.

Status is derived from the log:
    ARMED  - no unresolved START
    ACTIVE - the last event is a START with no END yet
    STALE  - ACTIVE but that START is older than --active-timeout
             (the detector may have died or the camera dropped)
"""

import argparse
import glob
import html
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

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
const esc = s => { const d = document.createElement("div"); d.textContent = s;
                   return d.innerHTML; };
const fmt = ts => ts ? new Date(ts * 1000).toLocaleString() : "—";
async function tick() {
  try {
    const r = await fetch("/api/events");
    const d = await r.json();
    const pill = document.getElementById("pill");
    pill.textContent = d.status; pill.className = "pill " + d.status;
    document.getElementById("st").textContent =
      d.status + (d.age != null ? ` (${d.age.toFixed(0)}s old)` : "");
    document.getElementById("lastseenv").textContent =
      d.last_seen ? fmt(d.last_seen) : "—";
    document.getElementById("evcount").textContent = d.events.length;
    document.getElementById("shcount").textContent = d.snapshots.length;
    const lsev = document.getElementById("lastsev");
    if (d.last) {
      const e = d.last;
      lsev.innerHTML = (e.type === "start" ? "🚨 " : "✅ ") +
        esc(e.type.toUpperCase() + " at " + fmt(e.ts)) +
        (e.duration ? ` — lasted ${e.duration}s` : "") +
        (e.detections && e.detections.length
          ? `<div class="det">` + e.detections.map(x =>
              esc(`class ${x.class_id} · ${x.score}`)).join("<br>") + "</div>" : "");
    } else { lsev.textContent = "no events yet — watching…"; lsev.className = "muted"; }
    const shots = document.getElementById("shots");
    shots.className = "shots";
    shots.innerHTML = d.snapshots.slice(0, 12).map(s =>
      `<a href="/snapshots/${encodeURIComponent(s.name)}" target="_blank">` +
      `<img src="/snapshots/${encodeURIComponent(s.name)}" alt="${esc(s.name)}">` +
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
      `log: ${d.src.log} · snapshots: ${d.src.snaps}`;
  } catch { /* keep polling */ }
}
setInterval(tick, 2500); tick();
const ck = () => { document.getElementById("clock").textContent =
                     new Date().toLocaleTimeString(); };
setInterval(ck, 1000); ck();
</script>
</body>
</html>
"""


def read_events(path, limit=300):
    out = []
    try:
        with open(path) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    return out[-limit:]


def derive_status(events, active_timeout):
    """ARMED / ACTIVE / (STALE, age) from the tail of the log."""
    for e in reversed(events):
        t = e.get("type")
        if t == "start":
            age = time.time() - float(e.get("ts", 0))
            if age > active_timeout:
                return "STALE", age
            return "ACTIVE", age
        if t == "end":
            break
    return "ARMED", None


def list_snapshots(snapshot_dir, limit=40):
    if not snapshot_dir:
        return []
    hits = sorted(glob.glob(os.path.join(snapshot_dir, "*.jpg")),
                  key=os.path.getmtime, reverse=True)
    return [{"name": os.path.basename(p),
             "ts": os.path.getmtime(p)} for p in hits[:limit]]


def make_handler(log_path, snapshot_dir, active_timeout):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/events":
                ev = read_events(log_path)
                status, age = derive_status(ev, active_timeout)
                last_seen = None
                if ev:
                    last_seen = ev[-1].get("ts", time.time())
                self._json({
                    "status": status, "age": age,
                    "last_seen": last_seen,
                    "last": ev[-1] if ev else None,
                    "events": ev, "snapshots": list_snapshots(snapshot_dir),
                    "src": {"log": log_path, "snaps": snapshot_dir or "(none)"},
                })
                return
            if path.startswith("/snapshots/"):
                name = os.path.basename(unquote(path[len("/snapshots/"):]))
                if not name:
                    self._json({"error": "no file"}, 400)
                    return
                try:
                    with open(os.path.join(snapshot_dir, name), "rb") as fh:
                        body = fh.read()
                except (OSError, TypeError):
                    self._json({"error": "not found"}, 404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
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
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--active-timeout", type=float, default=120.0,
                   help="seconds before an unresolved START is STALE")
    args = p.parse_args(argv)
    handler = make_handler(args.log, args.snapshot_dir, args.active_timeout)
    srv = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"dashboard on http://{args.bind}:{args.port}  "
          f"(log={args.log}, snaps={args.snapshot_dir})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()