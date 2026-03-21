"""
main.py — Edge node pipeline orchestration.

Two modes:
  Standalone : python -m cctv_agent src/videos/cctv1.mp4 rtsp://...
               Streams fixed at startup from command line.

  Agent mode : python -m cctv_agent --server 192.168.1.100
               No streams on command line. Node registers with central,
               waits for stream assignments, then starts the full
               pipeline with whatever streams central assigns.
               Re-checks assignments every poll_interval_s — if central
               changes assignments the pipeline restarts automatically.
"""

import multiprocessing
import multiprocessing.shared_memory
import sys
import time
import cv2
import numpy as np
from queue import Empty
import psutil
import os
import ctypes
import logging
import requests

from .config import Config, setup_logging, get_config
from .worker import decoder_worker, inference_worker
from .monitor import monitor_worker
from .reporter import reporter_worker
from .utils import compute_fps, summarize_latency


log = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_cores_per_decoder(num_streams):
    total = psutil.cpu_count(logical=False) or 4
    inference_reserve = max(2, total // 4)
    decoder_pool = total - inference_reserve
    per_decoder  = max(1, decoder_pool // max(num_streams, 1))
    while (per_decoder * num_streams) > (total - 2) and per_decoder > 1:
        per_decoder -= 1
    return per_decoder


def create_tiled_dashboard(frames_dict, num_streams):
    active, ref_size = [], None
    for i in range(num_streams):
        f = frames_dict.get(i)
        if f is not None:
            if ref_size is None:
                ref_size = (f.shape[1], f.shape[0])
                active.append(f)
            else:
                if (f.shape[1], f.shape[0]) != ref_size:
                    f = cv2.resize(f, ref_size)
                active.append(f)
    return np.hstack(active) if active else None


def collect_video_sources(args):
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.mjpeg')
    sources = []
    for arg in args:
        if os.path.isdir(arg):
            for file in sorted(os.listdir(arg)):
                if file.lower().endswith(video_extensions):
                    sources.append(os.path.join(arg, file))
        elif os.path.isfile(arg):
            sources.append(arg)
        else:
            sources.append(arg)
    return sources


def draw_boxes(frame, boxes, ids, classes, confs):
    if boxes is None:
        return frame
    for box, track_id, cls, conf in zip(boxes, ids, classes, confs):
        x1, y1, x2, y2 = map(int, box)
        track_id = -1 if track_id is None else int(track_id)
        conf = float(conf)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        id_text    = f"ID {track_id}"
        class_text = f"{cls}"
        conf_text  = f"{conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness, padding = 0.5, 2, 4
        (w1, h1), _ = cv2.getTextSize(id_text,    font, scale, thickness)
        (w2, h2), _ = cv2.getTextSize(class_text, font, scale, thickness)
        (w3, h3), _ = cv2.getTextSize(conf_text,  font, scale, thickness)
        total_width  = w1 + w2 + w3 + padding * 4
        label_height = max(h1, h2, h3) + padding * 2
        cv2.rectangle(frame, (x1, y1 - label_height),
                      (x1 + total_width, y1), (255, 255, 255), -1)
        text_y = y1 - padding
        cv2.putText(frame, id_text,
                    (x1 + padding, text_y), font, scale, (0, 0, 0), thickness)
        cv2.putText(frame, class_text,
                    (x1 + w1 + padding * 2, text_y), font, scale, (0, 255, 0), thickness)
        cv2.putText(frame, conf_text,
                    (x1 + w1 + w2 + padding * 3, text_y), font, scale, (255, 0, 0), thickness)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline — runs one fixed set of streams
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(stream_urls: list, cfg, outer_stop: multiprocessing.Event,
                  manager, reporter_stats, reporter_frames,
                  reporter_detections, assigned_streams):
    """
    Start and run the full pipeline for a fixed list of stream_urls.
    Returns when outer_stop is set OR when stream_urls from central change.
    Returns the new stream list so caller can restart with updated streams.
    """
    num_streams = len(stream_urls)
    log.info(f"Pipeline starting with {num_streams} streams: {stream_urls}")
    print(f"[Main] Starting pipeline — {num_streams} stream(s): {stream_urls}")

    stop_event      = multiprocessing.Event()
    inference_alive = multiprocessing.Event()

    fh, fw, fc  = cfg.pipeline.frame_height, cfg.pipeline.frame_width, cfg.pipeline.frame_channels
    frame_shape = (fh, fw, fc)
    frame_size  = fh * fw * fc

    # Shared memory
    shm_objects, shm_names = [], []
    for _ in range(num_streams):
        shm_a = multiprocessing.shared_memory.SharedMemory(create=True, size=frame_size)
        shm_b = multiprocessing.shared_memory.SharedMemory(create=True, size=frame_size)
        shm_objects.append((shm_a, shm_b))
        shm_names.append((shm_a.name, shm_b.name))

    meta_queues  = [multiprocessing.Queue(maxsize=cfg.pipeline.meta_queue_size)
                    for _ in range(num_streams)]
    result_queue = multiprocessing.Queue(maxsize=cfg.pipeline.result_queue_size)

    cores_per_decoder = compute_cores_per_decoder(num_streams)
    total_cores       = psutil.cpu_count(logical=False) or 4
    print(f"[Main] {total_cores} cores | {cores_per_decoder} per decoder | {num_streams} streams")

    # Start processes
    inference_p = multiprocessing.Process(
        target=inference_worker,
        args=(num_streams, frame_shape, shm_names,
              meta_queues, result_queue, stop_event,
              inference_alive, num_streams, cores_per_decoder),
    )
    inference_p.start()

    decoder_processes = []
    for i, url in enumerate(stream_urls):
        p = multiprocessing.Process(
            target=decoder_worker,
            args=(i, url, shm_names[i], frame_shape,
                  meta_queues[i], stop_event, inference_alive,
                  num_streams, cores_per_decoder),
        )
        p.start()
        decoder_processes.append(p)

    shared_cpu = multiprocessing.Value('d', 0.0)
    shared_ram = multiprocessing.Value('d', 0.0)
    monitor_p  = multiprocessing.Process(
        target=monitor_worker, args=(shared_cpu, shared_ram, stop_event),
    )
    monitor_p.start()

    # Stats
    stats = {i: {
        "frames": 0, "model_latencies": [], "pipeline_latencies": [],
        "start_time": None, "end_time": None,
        "last_fps_time": time.time(), "last_fps_frames": 0,
        "live_fps": 0.0, "last_model_ms": 0.0, "last_e2e_ms": 0.0,
    } for i in range(num_streams)}

    system_cpu, system_ram  = [], []
    latest_frames           = {i: None for i in range(num_streams)}
    last_log_time           = time.time()
    last_dashboard_time     = 0.0
    last_assignment_check   = time.time()
    LOG_INTERVAL            = cfg.dashboard.log_interval_s
    DASHBOARD_INTERVAL      = 1.0 / cfg.dashboard.fps_cap
    POLL_INTERVAL           = cfg.reporter.poll_interval_s

    new_streams = None  # will be set if central changes assignment

    print("🚀 Pipeline running. Press 'q' to quit.")

    try:
        while True:
            # Drain result queue
            try:
                for _ in range(64):
                    msg = result_queue.get_nowait()
                    if msg[0] == "frame":
                        (_, idx, frame, boxes, track_ids,
                         classes, confs, labels,
                         capture_time, model_lat, e2e_lat) = msg

                        s = stats[idx]
                        if s["start_time"] is None:
                            s["start_time"] = capture_time
                        s["end_time"]       = capture_time
                        s["frames"]        += 1
                        s["last_model_ms"]  = model_lat
                        s["last_e2e_ms"]    = e2e_lat

                        if len(s["model_latencies"]) >= 1000:
                            s["model_latencies"].pop(0)
                        s["model_latencies"].append(model_lat)
                        if len(s["pipeline_latencies"]) >= 1000:
                            s["pipeline_latencies"].pop(0)
                        s["pipeline_latencies"].append(e2e_lat)

                        frame = draw_boxes(frame.copy(), boxes,
                                           track_ids, labels, confs)
                        latest_frames[idx] = frame

                        reporter_frames[idx] = frame
                        reporter_stats[idx]  = {
                            "live_fps":      s["live_fps"],
                            "last_model_ms": model_lat,
                            "last_e2e_ms":   e2e_lat,
                        }
                        if boxes is not None and labels is not None:
                            det_list = [
                                {"label":    labels[j] if j < len(labels) else "?",
                                 "conf":     float(confs[j]) if confs is not None else 0.0,
                                 "track_id": int(track_ids[j]) if track_ids is not None else -1,
                                 "box":      [float(x) for x in boxes[j]]}
                                for j in range(len(boxes))
                            ]
                            reporter_detections[idx] = det_list
                        else:
                            reporter_detections[idx] = []
            except Empty:
                pass

            now_t = time.time()

            # Live FPS
            for i, s in stats.items():
                elapsed = now_t - s["last_fps_time"]
                if elapsed >= 1.0:
                    s["live_fps"]        = (s["frames"] - s["last_fps_frames"]) / elapsed
                    s["last_fps_time"]   = now_t
                    s["last_fps_frames"] = s["frames"]
                    if i in reporter_stats:
                        d = dict(reporter_stats[i])
                        d["live_fps"] = s["live_fps"]
                        reporter_stats[i] = d

            # Check if central changed assignments
            if now_t - last_assignment_check >= POLL_INTERVAL:
                try:
                    base = f"http://{cfg.central.host}:{cfg.central.port}"
                    r = requests.get(
                        f"{base}/assignment",
                        params={"node_id": cfg.node.id},
                        timeout=3
                    )
                    if r.status_code == 200:
                        current_from_central = r.json().get("streams", [])
                        if current_from_central and set(current_from_central) != set(stream_urls):
                            log.info(f"Assignment changed: {current_from_central}")
                            print(f"[Main] Assignment changed — restarting with {current_from_central}")
                            new_streams = current_from_central
                            break
                except Exception as e:
                    log.debug(f"Assignment check failed: {e}")
                last_assignment_check = now_t

            # Check outer stop
            if outer_stop.is_set():
                break

            # System stats
            cpu = shared_cpu.value
            ram = shared_ram.value
            if len(system_cpu) >= 1000:
                system_cpu.pop(0); system_ram.pop(0)
            system_cpu.append(cpu); system_ram.append(ram)

            # Periodic log
            if now_t - last_log_time >= LOG_INTERVAL:
                for i in range(num_streams):
                    s = stats[i]
                    if s["model_latencies"]:
                        line = (f"Stream {i} | FPS: {s['live_fps']:.1f} | "
                                f"Model: {s['model_latencies'][-1]:.1f}ms | "
                                f"E2E: {s['pipeline_latencies'][-1]:.1f}ms")
                        print(line); log.info(line)
                last_log_time = now_t

            # Dashboard
            if now_t - last_dashboard_time >= DASHBOARD_INTERVAL:
                dashboard = create_tiled_dashboard(latest_frames, num_streams)
                if dashboard is not None:
                    cv2.putText(dashboard, f"CPU: {cpu:.1f}%  RAM: {ram:.1f}%",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    tile_w = dashboard.shape[1] // max(num_streams, 1)
                    for i in range(num_streams):
                        cv2.putText(dashboard,
                                    f"S{i} {stats[i]['live_fps']:.1f}fps",
                                    (i * tile_w + 5, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    cv2.imshow("OpenVINO Dashboard", dashboard)
                last_dashboard_time = now_t

            if cv2.waitKey(1) & 0xFF == ord('q'):
                outer_stop.set()
                break

    finally:
        stop_event.set()
        for p in decoder_processes + [inference_p, monitor_p]:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate(); p.join(timeout=3)
                if p.is_alive(): p.kill(); p.join(timeout=2)

        for shm_a, shm_b in shm_objects:
            for shm in (shm_a, shm_b):
                try: shm.close()
                except Exception: pass
                try: shm.unlink()
                except Exception: pass

        cv2.destroyAllWindows()

        # Performance report
        print("\n=== Performance Report ===")
        for i in range(num_streams):
            s = stats[i]
            if not s["start_time"] or not s["end_time"]:
                continue
            duration = s["end_time"] - s["start_time"]
            fps = compute_fps(s["frames"], duration)
            m_avg, m_p95, m_max = summarize_latency(s["model_latencies"])
            p_avg, p_p95, p_max = summarize_latency(s["pipeline_latencies"])
            print(f"\nPipeline {i+1} ({stream_urls[i]})")
            print(f"  FPS: {fps:.2f} | Model: {m_avg:.1f}ms avg | E2E: {p_avg:.1f}ms avg")

        if system_cpu:
            print(f"\nSystem: CPU {sum(system_cpu)/len(system_cpu):.1f}% | "
                  f"RAM {sum(system_ram)/len(system_ram):.1f}%")

    return new_streams


# ─────────────────────────────────────────────────────────────────────────────
# Run entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(server=None, node_id=None, config_path=None, stream_urls=None):
    multiprocessing.set_start_method("spawn", force=True)

    cfg = Config.load(config_path)
    if server:
        cfg._data.setdefault("central", {})["host"] = server
        cfg._ns.central.host = server
    if node_id:
        cfg._data.setdefault("node", {})["id"] = node_id
        cfg._ns.node.id = node_id

    # Auto-resolve node_id: CLI arg > config > hostname
    import socket as _socket
    hostname = _socket.gethostname()
    resolved_node_id = node_id or cfg.node.id
    if not resolved_node_id or resolved_node_id in ('auto', 'node_01', 'my_laptop'):
        resolved_node_id = hostname

    cfg._data.setdefault('node', {})['id'] = resolved_node_id
    cfg._ns.node.id = resolved_node_id

    # Auto log file
    import os as _os
    log_file = cfg.get('logging.file', 'auto')
    if not log_file or log_file == 'auto':
        _os.makedirs('logs', exist_ok=True)
        cfg._data.setdefault('logging', {})['file'] = f'logs/{resolved_node_id}.log'

    # Write resolved node_id back to config.yaml so all subprocesses use it
    # This is the key fix — subprocesses load config fresh, so we persist it
    try:
        import yaml as _yaml
        from pathlib import Path as _Path
        cfg_path = config_path or 'config.yaml'
        if _Path(cfg_path).exists():
            with open(cfg_path) as f:
                raw = _yaml.safe_load(f)
            if raw.get('node', {}).get('id') in ('auto', 'node_01', 'my_laptop', None):
                raw.setdefault('node', {})['id'] = resolved_node_id
                raw.setdefault('logging', {})['file'] = f'logs/{resolved_node_id}.log'
                with open(cfg_path, 'w') as f:
                    _yaml.dump(raw, f, default_flow_style=False, allow_unicode=True)
                print(f"[Main] Node ID saved to config: {resolved_node_id}")
    except Exception as e:
        pass  # non-fatal — subprocesses will still use the manager-passed value

    setup_logging(cfg)

    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

    if stream_urls is None:
        stream_urls = []   # default to agent mode if called directly

    agent_mode = len(stream_urls) == 0

    outer_stop = multiprocessing.Event()
    manager    = multiprocessing.Manager()

    reporter_stats      = manager.dict()
    reporter_frames     = manager.dict()
    reporter_detections = manager.dict()
    assigned_streams    = manager.list()

    if agent_mode:
        print("[Main] Agent mode — no streams on command line.")
        print(f"[Main] Connected to central at {cfg.central.host}:{cfg.central.port}")
        print("[Main] Assign streams from the dashboard to start processing.")

        # Register with central directly before starting any processes
        try:
            import socket, platform, psutil as _psutil
            base = f"http://{cfg.central.host}:{cfg.central.port}"
            requests.post(f"{base}/register", json={
                "node_id":   cfg.node.id,
                "hostname":  socket.gethostname(),
                "cpu_cores": _psutil.cpu_count(logical=True),
                "ram_gb":    round(_psutil.virtual_memory().total / (1024**3), 1),
                "max_streams": cfg.get("node.max_streams", 4),
                "platform":  platform.system(),
            }, timeout=3)
            print(f"[Main] Registered with central server.")
        except Exception as e:
            print(f"[Main] Could not register: {e}")

        # Wait for stream assignment — simple blocking loop, no processes yet
        while True:
            try:
                base = f"http://{cfg.central.host}:{cfg.central.port}"
                r = requests.get(
                    f"{base}/assignment",
                    params={"node_id": cfg.node.id},
                    timeout=3
                )
                if r.status_code == 200:
                    current_urls = r.json().get("streams", [])
                    if current_urls:
                        print(f"[Main] Assignment received: {current_urls}")
                        stream_urls = current_urls
                        break
            except Exception as e:
                print(f"[Main] Poll error: {e}")
            print("[Main] Waiting for stream assignment from central...")
            time.sleep(cfg.reporter.poll_interval_s)

    # Start reporter (stays alive for the entire session)
    reporter_p = multiprocessing.Process(
        target=reporter_worker,
        args=(outer_stop, reporter_stats, reporter_frames,
              reporter_detections, assigned_streams),
    )
    reporter_p.start()

    try:
        # Main loop — restart pipeline if assignments change
        while not outer_stop.is_set() and stream_urls:
            new_streams = _run_pipeline(
                stream_urls, cfg, outer_stop,
                manager, reporter_stats, reporter_frames,
                reporter_detections, assigned_streams
            )
            if new_streams is not None and not outer_stop.is_set():
                stream_urls = new_streams
                print(f"[Main] Restarting pipeline with new streams: {stream_urls}")
            else:
                break
    finally:
        outer_stop.set()
        reporter_p.join(timeout=5)
        if reporter_p.is_alive():
            reporter_p.terminate()
        manager.shutdown()
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        print("[Main] Session ended.")