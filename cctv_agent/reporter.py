import time
import logging
import socket
import multiprocessing
import psutil
import cv2
import numpy as np
import platform

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    requests = None

from .config import get_config
from .monitor import get_system_usage

log = logging.getLogger("reporter")


def _make_session(retries=2, backoff=0.3):
    if requests is None:
        return None
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff,
                  status_forcelist=[500, 502, 503, 504],
                  allowed_methods=["GET", "POST"])
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


def reporter_worker(stop_event, stats_dict, latest_frames,
                    latest_detections, assigned_streams):
    cfg = get_config()
    base = _base_url(cfg)
    node_id = cfg.node.id
    session = _make_session()

    hb_interval = cfg.reporter.heartbeat_interval_s
    poll_interval = cfg.reporter.poll_interval_s
    ingest_interval = cfg.reporter.ingest_interval_s

    log.info(f"Reporter started - central: {base}, node: {node_id}")
    print(f"[Reporter] Starting - central: {base}")

    _register(session, base, node_id, cfg)

    last_hb = 0.0
    last_poll = 0.0
    last_ingest = 0.0

    while not stop_event.is_set():
        now = time.time()

        if now - last_hb >= hb_interval:
            _heartbeat(session, base, node_id, stats_dict)
            last_hb = now

        if now - last_poll >= poll_interval:
            new = _poll_assignment(session, base, node_id)
            if new is not None:
                del assigned_streams[:]
                assigned_streams.extend(new)
            last_poll = now

        if now - last_ingest >= ingest_interval:
            _ingest(session, base, node_id,
                    latest_frames, latest_detections, stats_dict)
            last_ingest = now

        time.sleep(0.05)

    log.info("Reporter stopped.")
    print("[Reporter] Stopped.")


def _register(session, base, node_id, cfg):
    payload = {
        "node_id": node_id,
        "hostname": socket.gethostname(),
        "cpu_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
        "max_streams": cfg.get("node.max_streams", 4),
        "platform": platform.system(),
    }
    result = _post(session, f"{base}/register", payload)
    if result:
        print(f"[Reporter] Registered with central server.")
    else:
        print(f"[Reporter] Central server unreachable - retrying...")


def _heartbeat(session, base, node_id, stats_dict):
    cpu, ram = get_system_usage(interval=0.1)
    fps_list = [s.get("live_fps", 0.0) for s in stats_dict.values()]
    avg_fps = round(sum(fps_list) / len(fps_list), 1) if fps_list else 0.0
    payload = {
        "node_id": node_id,
        "cpu_pct": round(cpu, 1),
        "ram_pct": round(ram, 1),
        "stream_count": len(stats_dict),
        "avg_fps": avg_fps,
        "timestamp": time.time(),
    }
    _post(session, f"{base}/heartbeat", payload)


def _poll_assignment(session, base, node_id):
    result = _get(session, f"{base}/assignment?node_id={node_id}")
    if result is None:
        return None
    return result.get("streams", [])


def _ingest(session, base, node_id, latest_frames,
            latest_detections, stats_dict):
    for stream_id, frame in latest_frames.items():
        if frame is None:
            continue
        try:
            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 70])
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

        s = stats_dict.get(stream_id, {})
        payload = {
            "node_id": node_id,
            "stream_id": stream_id,
            "detections": latest_detections.get(stream_id, []),
            "fps": round(s.get("live_fps", 0.0), 1),
            "model_ms": round(s.get("last_model_ms", 0.0), 1),
            "e2e_ms": round(s.get("last_e2e_ms", 0.0), 1),
            "timestamp": time.time(),
        }
        _post(session, f"{base}/ingest/detections", payload)
