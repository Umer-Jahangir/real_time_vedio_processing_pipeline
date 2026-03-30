import time
import queue
import logging
import socket
import psutil
import cv2
import platform

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    requests = None

from .config import Config, get_config
from .monitor import get_system_usage

log = logging.getLogger("reporter")


def _make_session(retries=2, backoff=0.3):
    if requests is None:
        return None
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    return session


def _base_url(cfg):
    return f"http://{cfg.central.host}:{cfg.central.port}"


def _post(session, url, payload, timeout=3.0):
    try:
        r = session.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug(f"POST {url} failed: {e}")
        return None


def _get(session, url, timeout=3.0):
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug(f"GET {url} failed: {e}")
        return None


def _post_bytes(session, url, data, headers, timeout=3.0):
    try:
        r = session.post(url, data=data, headers=headers, timeout=timeout)
        return r.status_code == 200
    except Exception as e:
        log.debug(f"POST bytes {url} failed: {e}")
        return False


def reporter_worker(stop_event, reporter_q, assign_q, cfg_dict=None):
    """
    Drains reporter_q, sends heartbeats, polls assignment, ingests frames.

    Key decisions:
    - ingest_interval floored at 0.5s regardless of config value.
      The original 0.1s caused JPEG encode + HTTP POST to compete for CPU
      with the inference process, contributing to OS preemption during
      pickle operations and E2E spikes.
    - last_pushed_assignment dedup: only pushes to assign_q when the
      assignment list actually changes, preventing spurious pipeline
      restarts from repeated identical poll responses.
    - 100Hz loop (0.01s sleep) for fast reporter_q draining.
    """
    if cfg_dict is not None:
        cfg = Config.from_dict(cfg_dict)
    else:
        cfg = get_config()

    base    = _base_url(cfg)
    node_id = cfg.node.id
    session = _make_session()

    hb_interval     = cfg.reporter.heartbeat_interval_s
    poll_interval   = cfg.reporter.poll_interval_s
    ingest_interval = max(cfg.reporter.ingest_interval_s, 0.5)   # floor at 0.5s

    log.info(f"Reporter started — central: {base}, node: {node_id}")
    print(f"[Reporter] Starting — central: {base}")

    _register(session, base, node_id, cfg)

    latest_frames          = {}
    latest_detections      = {}
    stats_dict             = {}
    last_hb                = 0.0
    last_poll              = 0.0
    last_ingest            = 0.0
    last_pushed_assignment = None   # dedup: don't push same list twice

    while not stop_event.is_set():

        # Drain up to 128 messages per tick
        drained = 0
        while drained < 128:
            try:
                msg = reporter_q.get_nowait()
            except queue.Empty:
                break
            sid = msg["stream_id"]
            stats_dict[sid] = {
                "live_fps":      msg.get("fps",      0.0),
                "last_model_ms": msg.get("model_ms", 0.0),
                "last_e2e_ms":   msg.get("e2e_ms",   0.0),
            }
            latest_detections[sid] = msg.get("detections", [])
            if "frame" in msg and msg["frame"] is not None:
                latest_frames[sid] = msg["frame"]
            drained += 1

        now_t = time.time()

        if now_t - last_hb >= hb_interval:
            _heartbeat(session, base, node_id, stats_dict)
            last_hb = now_t

        if now_t - last_poll >= poll_interval:
            new = _poll_assignment(session, base, node_id)
            if new is not None and new != last_pushed_assignment:
                last_pushed_assignment = new
                try:
                    assign_q.put_nowait(new)
                except Exception:
                    pass
            last_poll = now_t

        if now_t - last_ingest >= ingest_interval:
            _ingest(session, base, node_id,
                    latest_frames, latest_detections, stats_dict)
            last_ingest = now_t

        time.sleep(0.01)   # 100 Hz loop

    log.info("Reporter stopped.")
    print("[Reporter] Stopped.")


def _register(session, base, node_id, cfg):
    payload = {
        "node_id":     node_id,
        "hostname":    socket.gethostname(),
        "cpu_cores":   psutil.cpu_count(logical=True),
        "ram_gb":      round(psutil.virtual_memory().total / (1024 ** 3), 1),
        "max_streams": cfg.get("node.max_streams", 4),
        "platform":    platform.system(),
    }
    result = _post(session, f"{base}/register", payload)
    if result:
        print("[Reporter] Registered with central server.")
    else:
        print("[Reporter] Central server unreachable — will retry on next heartbeat.")


def _heartbeat(session, base, node_id, stats_dict):
    cpu, ram = get_system_usage(interval=0.1)
    fps_list = [s.get("live_fps", 0.0) for s in stats_dict.values()]
    avg_fps  = round(sum(fps_list) / len(fps_list), 1) if fps_list else 0.0
    payload  = {
        "node_id":      node_id,
        "cpu_pct":      round(cpu, 1),
        "ram_pct":      round(ram, 1),
        "stream_count": len(stats_dict),
        "avg_fps":      avg_fps,
        "timestamp":    time.time(),
    }
    _post(session, f"{base}/heartbeat", payload)


def _poll_assignment(session, base, node_id):
    result = _get(session, f"{base}/assignment?node_id={node_id}")
    if result is None:
        return None
    return result.get("streams", [])


def _ingest(session, base, node_id, latest_frames, latest_detections, stats_dict):
    for stream_id, frame in latest_frames.items():
        if frame is None:
            continue
        try:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue
            jpeg_bytes = buf.tobytes()
        except Exception:
            continue
        _post_bytes(
            session,
            f"{base}/ingest/frame?node_id={node_id}&stream_id={stream_id}",
            jpeg_bytes,
            headers={"Content-Type": "image/jpeg"},
        )

    for stream_id in latest_detections:
        s = stats_dict.get(stream_id, {})
        payload = {
            "node_id":    node_id,
            "stream_id":  stream_id,
            "detections": latest_detections.get(stream_id, []),
            "fps":        round(s.get("live_fps",      0.0), 1),
            "model_ms":   round(s.get("last_model_ms", 0.0), 1),
            "e2e_ms":     round(s.get("last_e2e_ms",   0.0), 1),
            "timestamp":  time.time(),
        }
        _post(session, f"{base}/ingest/detections", payload)