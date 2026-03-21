"""
server.py — Central server for distributed CCTV system.

Run:
    cd central
    python server.py
"""

import sys
import time
import logging
import asyncio
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from cctv_agent.config import Config, setup_logging

# ── Config ────────────────────────────────────────────────────────────────────
cfg = Config.load(str(Path(__file__).parent / "config.yaml"))
setup_logging(cfg)
log = logging.getLogger("server")

HEARTBEAT_TIMEOUT  = cfg.heartbeat.timeout_s
RECORDING_ENABLED  = cfg.get("recording.enabled", False)
RECORDING_DIR      = Path(cfg.get("recording.output_dir", "recordings"))

if RECORDING_ENABLED:
    RECORDING_DIR.mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# ── In-memory state ───────────────────────────────────────────────────────────
nodes:             dict = {}   # node_id → node info
stream_pool:       dict = {}   # stream_url → node_id | None
latest_frames:     dict = {}   # "node_id:stream_id" → jpeg bytes
latest_detections: dict = {}   # "node_id:stream_id" → detection list
stream_stats:      dict = {}   # "node_id:stream_id" → stats dict
alerts:            list = []   # smart alerts list (last 100)

# ── Detection summary state ───────────────────────────────────────────────────
# "node_id:stream_id" → {label: count}
detection_counts: dict = defaultdict(lambda: defaultdict(int))
last_summary_time: float = time.time()
SUMMARY_INTERVAL  = 60.0  # summarise every 60 seconds

# ── Alert cooldown state ──────────────────────────────────────────────────────
# key → last_alert_time
alert_cooldowns: dict = {}
CPU_HIGH_THRESHOLD  = cfg.get("alerts.cpu_high_threshold",  80)
RAM_HIGH_THRESHOLD  = cfg.get("alerts.ram_high_threshold",  88)
FPS_DROP_THRESHOLD  = cfg.get("alerts.fps_drop_threshold",  0.5)  # 50% drop
NO_FRAME_TIMEOUT    = cfg.get("alerts.no_frame_timeout_s",  8)
COOLDOWN_NODE_CPU   = cfg.get("alerts.cooldown_node_cpu_s", 30)
COOLDOWN_NODE_RAM   = cfg.get("alerts.cooldown_node_ram_s", 30)
COOLDOWN_STREAM     = cfg.get("alerts.cooldown_stream_s",   15)
COOLDOWN_FPS        = cfg.get("alerts.cooldown_fps_s",      20)

app = FastAPI(title="CCTV Central Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Alert helpers ─────────────────────────────────────────────────────────────

def _can_alert(key: str, cooldown: float) -> bool:
    now = time.time()
    if now - alert_cooldowns.get(key, 0) >= cooldown:
        alert_cooldowns[key] = now
        return True
    return False


def _push_alert(category: str, level: str, title: str,
                message: str, node_id: str = None,
                stream_id=None, suggestion: str = None):
    """
    category : node_health | stream_health | performance | summary
    level    : info | warning | critical
    """
    alert = {
        "category":   category,
        "level":      level,
        "title":      title,
        "message":    message,
        "node_id":    node_id,
        "stream_id":  stream_id,
        "suggestion": suggestion,
        "time":       time.strftime("%H:%M:%S"),
        "timestamp":  time.time(),
    }
    alerts.append(alert)
    if len(alerts) > 100:
        alerts.pop(0)
    log.info(f"[{level.upper()}] {title} — {message}")


def _check_node_health(node_id: str, info: dict):
    """Fire alerts based on node CPU/RAM."""
    cpu = info.get("cpu_pct", 0)
    ram = info.get("ram_pct", 0)

    if cpu > CPU_HIGH_THRESHOLD:
        key = f"cpu_high:{node_id}"
        if _can_alert(key, COOLDOWN_NODE_CPU):
            # Find the least loaded other node for suggestion
            others = [
                (nid, n) for nid, n in nodes.items()
                if nid != node_id and n.get("status") == "online"
                and n.get("cpu_pct", 100) < 60
            ]
            suggestion = None
            if others:
                lightest = min(others, key=lambda x: x[1].get("cpu_pct", 100))
                # Find streams assigned to this node
                my_streams = [u for u, nid in stream_pool.items() if nid == node_id]
                if my_streams:
                    suggestion = f"Consider moving a stream to {lightest[0]} (CPU: {lightest[1].get('cpu_pct',0):.0f}%)"

            _push_alert(
                "node_health", "warning",
                f"High CPU on {node_id}",
                f"CPU at {cpu:.0f}% (threshold: {CPU_HIGH_THRESHOLD}%)",
                node_id=node_id,
                suggestion=suggestion,
            )

    if ram > RAM_HIGH_THRESHOLD:
        key = f"ram_high:{node_id}"
        if _can_alert(key, COOLDOWN_NODE_RAM):
            _push_alert(
                "node_health", "critical",
                f"High RAM on {node_id}",
                f"RAM at {ram:.0f}% — pipeline may degrade soon",
                node_id=node_id,
                suggestion="Reduce number of streams on this node",
            )


def _check_stream_health():
    """Check for missing frames and FPS drops."""
    now = time.time()
    for url, node_id in stream_pool.items():
        if not node_id:
            continue

        # Find stream index (approximate by position)
        assigned = [u for u, nid in stream_pool.items() if nid == node_id]
        for stream_id, u in enumerate(assigned):
            if u != url:
                continue

            key      = f"{node_id}:{stream_id}"
            s        = stream_stats.get(key, {})
            ts       = s.get("timestamp", 0)
            fps      = s.get("fps", 0)

            # No frames received for N seconds
            if ts > 0 and (now - ts) > NO_FRAME_TIMEOUT:
                ckey = f"no_frame:{key}"
                if _can_alert(ckey, COOLDOWN_STREAM):
                    _push_alert(
                        "stream_health", "critical",
                        f"Stream down on {node_id}",
                        f"No frames from stream{stream_id} for {NO_FRAME_TIMEOUT}s",
                        node_id=node_id,
                        stream_id=stream_id,
                        suggestion=f"Check camera/source: {url}",
                    )

            # FPS degradation
            if fps > 0 and fps < 10:
                ckey = f"fps_low:{key}"
                if _can_alert(ckey, COOLDOWN_FPS):
                    _push_alert(
                        "performance", "warning",
                        f"Low FPS on {node_id}/stream{stream_id}",
                        f"FPS dropped to {fps:.1f} (expected ~25)",
                        node_id=node_id,
                        stream_id=stream_id,
                        suggestion="Node may be overloaded — check CPU usage",
                    )


def _check_detection_summary():
    """Every 60s push a detection count summary per stream."""
    global last_summary_time
    now = time.time()
    if now - last_summary_time < SUMMARY_INTERVAL:
        return
    last_summary_time = now

    for key, counts in detection_counts.items():
        if not counts:
            continue
        node_id, stream_id = key.split(":", 1)
        parts = ", ".join(f"{label}: {cnt}" for label, cnt in
                          sorted(counts.items(), key=lambda x: -x[1]))
        _push_alert(
            "summary", "info",
            f"60s Summary — {node_id}/stream{stream_id}",
            parts,
            node_id=node_id,
            stream_id=stream_id,
        )
    # Reset counts
    detection_counts.clear()


# ── Watchdog ──────────────────────────────────────────────────────────────────
async def _watchdog():
    while True:
        now = time.time()
        for nid, info in nodes.items():
            if now - info.get("last_seen", 0) > HEARTBEAT_TIMEOUT:
                if info.get("status") != "offline":
                    info["status"] = "offline"
                    log.warning(f"Node {nid} went offline")
                    _push_alert(
                        "node_health", "critical",
                        f"Node offline: {nid}",
                        f"No heartbeat received for {HEARTBEAT_TIMEOUT}s",
                        node_id=nid,
                        suggestion="Check network connection or restart edge agent",
                    )
            elif info.get("status") == "online":
                _check_node_health(nid, info)

        _check_stream_health()
        _check_detection_summary()
        await asyncio.sleep(3)


@app.on_event("startup")
async def startup():
    asyncio.create_task(_watchdog())
    log.info("Central server started.")
    print(f"[Server] Dashboard → http://localhost:{cfg.server.port}/dashboard")
    print(f"[Server] API docs  → http://localhost:{cfg.server.port}/docs")


# ── Registration + heartbeat ──────────────────────────────────────────────────
@app.post("/register")
async def register(request: Request):
    data    = await request.json()
    node_id = data.get("node_id")
    if not node_id:
        raise HTTPException(400, "node_id required")

    was_offline = node_id in nodes and nodes[node_id].get("status") == "offline"

    nodes[node_id] = {
        "node_id":       node_id,
        "hostname":      data.get("hostname", "unknown"),
        "cpu_cores":     data.get("cpu_cores", 0),
        "ram_gb":        data.get("ram_gb", 0),
        "max_streams":   data.get("max_streams", 4),
        "platform":      data.get("platform", "unknown"),
        "registered_at": time.time(),
        "last_seen":     time.time(),
        "cpu_pct":       0.0,
        "ram_pct":       0.0,
        "stream_count":  0,
        "avg_fps":       0.0,
        "status":        "online",
    }
    log.info(f"Node registered: {node_id}")

    if was_offline:
        _push_alert(
            "node_health", "info",
            f"Node back online: {node_id}",
            f"{data.get('hostname')} reconnected successfully",
            node_id=node_id,
        )
    else:
        _push_alert(
            "node_health", "info",
            f"New node connected: {node_id}",
            f"{data.get('hostname')} · {data.get('cpu_cores')} cores · {data.get('ram_gb')}GB RAM",
            node_id=node_id,
            suggestion="Add streams and assign them to this node from the dashboard",
        )

    assigned = [url for url, nid in stream_pool.items() if nid == node_id]
    return {"status": "ok", "streams": assigned}


@app.post("/heartbeat")
async def heartbeat(request: Request):
    data    = await request.json()
    node_id = data.get("node_id")
    if not node_id:
        raise HTTPException(400, "node_id required")
    if node_id not in nodes:
        nodes[node_id] = {
            "node_id": node_id, "hostname": "unknown",
            "registered_at": time.time(), "status": "online",
            "cpu_cores": 0, "ram_gb": 0, "max_streams": 4, "platform": "unknown",
        }
    nodes[node_id].update({
        "last_seen":    time.time(),
        "cpu_pct":      data.get("cpu_pct",      0.0),
        "ram_pct":      data.get("ram_pct",       0.0),
        "stream_count": data.get("stream_count",  0),
        "avg_fps":      data.get("avg_fps",        0.0),
        "status":       "online",
    })
    return {"status": "ok"}


@app.get("/assignment")
async def get_assignment(node_id: str):
    assigned = [url for url, nid in stream_pool.items() if nid == node_id]
    return {"node_id": node_id, "streams": assigned}


# ── Ingest ────────────────────────────────────────────────────────────────────
@app.post("/ingest/frame")
async def ingest_frame(request: Request, node_id: str, stream_id: int):
    latest_frames[f"{node_id}:{stream_id}"] = await request.body()
    return {"status": "ok"}


@app.post("/ingest/detections")
async def ingest_detections(request: Request):
    data      = await request.json()
    node_id   = data.get("node_id")
    stream_id = data.get("stream_id")
    key       = f"{node_id}:{stream_id}"

    latest_detections[key] = data.get("detections", [])
    stream_stats[key] = {
        "fps":       data.get("fps",       0.0),
        "model_ms":  data.get("model_ms",  0.0),
        "e2e_ms":    data.get("e2e_ms",    0.0),
        "timestamp": data.get("timestamp", time.time()),
        "node_id":   node_id,
        "stream_id": stream_id,
    }

    # Accumulate detection counts for summary
    for det in data.get("detections", []):
        label = det.get("label", "unknown")
        detection_counts[key][label] += 1

    # Save snapshot on detection if recording enabled
    if RECORDING_ENABLED and data.get("detections"):
        if key in latest_frames:
            _save_snapshot(node_id, stream_id, latest_frames[key])

    return {"status": "ok"}


def _save_snapshot(node_id, stream_id, jpeg_bytes):
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = RECORDING_DIR / f"{node_id}_stream{stream_id}_{ts}.jpg"
    try:
        path.write_bytes(jpeg_bytes)
        log.info(f"Snapshot saved: {path}")
    except Exception as e:
        log.warning(f"Snapshot failed: {e}")


# ── MJPEG ─────────────────────────────────────────────────────────────────────
async def _mjpeg_gen(node_id: str, stream_id: int):
    key = f"{node_id}:{stream_id}"
    while True:
        frame = latest_frames.get(key)
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + frame + b"\r\n")
        await asyncio.sleep(0.033)


@app.get("/streams/{node_id}/{stream_id}/mjpeg")
async def mjpeg_feed(node_id: str, stream_id: int):
    return StreamingResponse(
        _mjpeg_gen(node_id, stream_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Stream pool management ────────────────────────────────────────────────────
@app.post("/streams/add")
async def add_stream(request: Request):
    data = await request.json()
    url  = data.get("url")
    if not url:
        raise HTTPException(400, "url required")
    stream_pool.setdefault(url, None)
    log.info(f"Stream added: {url}")
    return {"status": "ok", "stream_pool": stream_pool}


@app.post("/streams/remove")
async def remove_stream(request: Request):
    data = await request.json()
    url  = data.get("url")
    stream_pool.pop(url, None)
    return {"status": "ok"}


@app.post("/streams/assign")
async def assign_stream(request: Request):
    data    = await request.json()
    url     = data.get("url")
    node_id = data.get("node_id")
    if not url or not node_id:
        raise HTTPException(400, "url and node_id required")
    if node_id not in nodes:
        raise HTTPException(404, f"Node '{node_id}' not registered")
    stream_pool[url] = node_id
    log.info(f"Assigned {url} → {node_id}")
    _push_alert(
        "stream_health", "info",
        f"Stream assigned",
        f"Stream assigned to {node_id}",
        node_id=node_id,
        suggestion=None,
    )
    return {"status": "ok", "url": url, "node_id": node_id}


# ── Query endpoints ───────────────────────────────────────────────────────────
@app.get("/nodes")
async def get_nodes():
    return {"nodes": list(nodes.values())}


@app.get("/stats")
async def get_stats():
    return {
        "nodes":        list(nodes.values()),
        "stream_pool":  stream_pool,
        "stream_stats": stream_stats,
        "alert_count":  len(alerts),
    }


@app.get("/alerts")
async def get_alerts():
    return {"alerts": list(reversed(alerts))}


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard not found — add dashboard.html to central/</h1>")


@app.get("/")
async def root():
    return JSONResponse({"status": "ok",
                         "dashboard": "/dashboard", "docs": "/docs"})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[Server] Starting on http://{cfg.server.host}:{cfg.server.port}")
    uvicorn.run("server:app", host=cfg.server.host,
                port=cfg.server.port, reload=False, log_level="warning")