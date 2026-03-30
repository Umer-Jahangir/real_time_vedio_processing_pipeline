"""
main.py — Edge node pipeline orchestration.

Key fixes in this version:

  1. cv2 WINDOW BLOCKING FIX
     cv2.waitKey(1) on Windows stalls the main thread when the window is
     minimized or hidden — the Win32 message pump deprioritizes invisible
     windows. When the drain loop is stalled, frames age past 150ms,
     inference drops them, and the reporter sees zero frames → stream
     shows as offline on the central dashboard.
     FIX: cv2.imshow / cv2.waitKey run in a dedicated daemon thread
     (_DisplayThread). The main drain loop never calls cv2. It runs
     uninterrupted regardless of window visibility.

  2. 3-SLOT SHARED MEMORY — ZERO PICKLE PER FRAME
     Each stream has 3 shm slots:
       Slot 0, 1: decoder double-buffer (decoder writes, inference reads)
       Slot 2:    display buffer (inference writes, main reads)
     Result message contains ONLY metadata (~2KB, not ~245KB).
     This eliminates the 50-150ms pickle block that caused high E2E on
     stream 2 and 3.

  3. PER-FRAME SEQUENTIAL INFERENCE
     Each frame is inferred immediately after dequeue. E2E is stamped
     right after that frame's own inference — not after a batch.

  4. _AssignmentPoller REMOVED
     Reporter is the sole assignment source.

  5. WARMUP_SKIP_FRAMES=30
     First 30 frames excluded from latency stats.
"""

import multiprocessing
import multiprocessing.shared_memory
import queue
import time
import threading
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
from .utils import compute_fps, summarize_latency, now


log = logging.getLogger("main")

FRAME_SKIP         = 15   # 1 in 15 frames forwarded to reporter
WARMUP_SKIP_FRAMES = 30   # frames excluded from latency stats after pipeline start


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_cores_per_decoder(num_streams):
    total         = psutil.cpu_count(logical=False) or 4
    inference_rsv = max(2, total // 4)
    decoder_pool  = total - inference_rsv
    per_decoder   = max(1, decoder_pool // max(num_streams, 1))
    while (per_decoder * num_streams) > (total - 2) and per_decoder > 1:
        per_decoder -= 1
    return per_decoder


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
        conf     = float(conf)
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
        cv2.rectangle(frame,
                      (x1, y1 - label_height), (x1 + total_width, y1),
                      (255, 255, 255), -1)
        text_y = y1 - padding
        cv2.putText(frame, id_text,
                    (x1 + padding, text_y), font, scale, (0, 0, 0), thickness)
        cv2.putText(frame, class_text,
                    (x1 + w1 + padding * 2, text_y), font, scale, (0, 255, 0), thickness)
        cv2.putText(frame, conf_text,
                    (x1 + w1 + w2 + padding * 3, text_y), font, scale, (255, 0, 0), thickness)
    return frame


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


# ─────────────────────────────────────────────────────────────────────────────
# Display thread — isolates cv2.waitKey from the main drain loop
# ─────────────────────────────────────────────────────────────────────────────

class _DisplayThread(threading.Thread):
    """
    Runs cv2.imshow and cv2.waitKey in a dedicated daemon thread so that
    Windows message-pump stalls when the window is minimized/hidden do NOT
    block the main result_queue drain loop.

    Communication with main loop via latest_display_state (plain dict, no
    lock needed — Python dict assignment is atomic for simple value types
    and worst-case main sees a slightly stale frame, which is fine for a
    display-only thread).
    """

    def __init__(self, stop_event, quit_event, num_streams,
                 stats_ref, latest_display_state, fps_cap):
        super().__init__(daemon=True, name="display")
        self._stop_event           = stop_event
        self._quit_event           = quit_event
        self._num_streams          = num_streams
        self._stats_ref            = stats_ref            # reference to main stats dict
        self._latest_display_state = latest_display_state
        self._interval             = 1.0 / max(fps_cap, 1)
        self._window_created       = False

    def run(self):
        last_render = 0.0
        while not self._stop_event.is_set():
            now_t = time.time()
            if now_t - last_render < self._interval:
                time.sleep(0.005)
                continue

            state  = self._latest_display_state
            frames = state.get("frames", {})
            cpu    = state.get("cpu", 0.0)
            ram    = state.get("ram", 0.0)

            dashboard = create_tiled_dashboard(frames, self._num_streams)
            if dashboard is not None:
                if not self._window_created:
                    cv2.namedWindow("CCTV Dashboard", cv2.WINDOW_NORMAL)
                    self._window_created = True

                cv2.putText(dashboard, f"CPU: {cpu:.1f}%  RAM: {ram:.1f}%",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                tile_w = dashboard.shape[1] // max(self._num_streams, 1)
                for i in range(self._num_streams):
                    s      = self._stats_ref.get(i, {})
                    fps_val = s.get("live_fps", 0.0)
                    cv2.putText(dashboard, f"S{i} {fps_val:.1f}fps",
                                (i * tile_w + 5, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow("CCTV Dashboard", dashboard)

            # waitKey must be called from the same thread as imshow.
            # If the window is minimized this may stall briefly — that only
            # affects THIS thread, not the main drain loop.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._quit_event.set()

            last_render = now_t

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(stream_urls: list, cfg, outer_stop: multiprocessing.Event,
                  reporter_q: multiprocessing.Queue,
                  assign_q:   multiprocessing.Queue):
    num_streams = len(stream_urls)
    log.info(f"Pipeline starting with {num_streams} streams: {stream_urls}")
    print(f"[Main] Starting pipeline — {num_streams} stream(s): {stream_urls}")

    stop_event      = multiprocessing.Event()
    inference_alive = multiprocessing.Event()

    fh, fw, fc  = cfg.pipeline.frame_height, cfg.pipeline.frame_width, cfg.pipeline.frame_channels
    frame_shape = (fh, fw, fc)
    frame_size  = fh * fw * fc
    cfg_dict    = cfg.to_dict()

    # ── 3-slot shm per stream ─────────────────────────────────────────────────
    shm_objects, shm_names = [], []
    for _ in range(num_streams):
        shm_a = multiprocessing.shared_memory.SharedMemory(create=True, size=frame_size)
        shm_b = multiprocessing.shared_memory.SharedMemory(create=True, size=frame_size)
        shm_c = multiprocessing.shared_memory.SharedMemory(create=True, size=frame_size)
        shm_objects.append((shm_a, shm_b, shm_c))
        shm_names.append((shm_a.name, shm_b.name, shm_c.name))

    # Main-process views onto display slot (slot 2)
    shm_display_views = [
        np.ndarray(frame_shape, dtype=np.uint8, buffer=shm_triple[2].buf)
        for shm_triple in shm_objects
    ]

    meta_queues  = [multiprocessing.Queue(maxsize=cfg.pipeline.meta_queue_size)
                    for _ in range(num_streams)]
    result_queue = multiprocessing.Queue(maxsize=cfg.pipeline.result_queue_size)

    cores_per_decoder = compute_cores_per_decoder(num_streams)
    total_cores       = psutil.cpu_count(logical=False) or 4
    print(f"[Main] {total_cores} cores | {cores_per_decoder} per decoder | {num_streams} streams")

    pipeline_start_time = now()

    inference_p = multiprocessing.Process(
        target=inference_worker,
        args=(num_streams, frame_shape, shm_names,
              meta_queues, result_queue, stop_event,
              inference_alive, num_streams, cores_per_decoder,
              cfg_dict, pipeline_start_time),
        name="inference",
    )
    inference_p.start()

    decoder_processes = []
    for i, url in enumerate(stream_urls):
        p = multiprocessing.Process(
            target=decoder_worker,
            args=(i, url, shm_names[i], frame_shape,
                  meta_queues[i], stop_event, inference_alive,
                  num_streams, cores_per_decoder, cfg_dict),
            name=f"decoder-{i}",
        )
        p.start()
        decoder_processes.append(p)

    shared_cpu = multiprocessing.Value('d', 0.0)
    shared_ram = multiprocessing.Value('d', 0.0)
    monitor_p  = multiprocessing.Process(
        target=monitor_worker,
        args=(shared_cpu, shared_ram, stop_event),
        name="monitor",
    )
    monitor_p.start()

    stats = {i: {
        "frames":             0,
        "model_latencies":    [],
        "pipeline_latencies": [],
        "start_time":         None,
        "end_time":           None,
        "last_fps_time":      time.time(),
        "last_fps_frames":    0,
        "live_fps":           0.0,
        "last_model_ms":      0.0,
        "last_e2e_ms":        0.0,
    } for i in range(num_streams)}

    system_cpu           = []
    system_ram           = []
    latest_frames        = {i: None for i in range(num_streams)}
    last_log_time        = time.time()
    LOG_INTERVAL         = cfg.dashboard.log_interval_s
    new_streams          = None
    _restart_lock        = threading.Lock()
    _restart_issued      = [False]
    latest_display_state = {"frames": {}, "cpu": 0.0, "ram": 0.0}
    quit_event           = threading.Event()

    display_thread = _DisplayThread(
        stop_event           = stop_event,
        quit_event           = quit_event,
        num_streams          = num_streams,
        stats_ref            = stats,
        latest_display_state = latest_display_state,
        fps_cap              = cfg.dashboard.fps_cap,
    )
    display_thread.start()

    print("🚀 Pipeline running. Press 'q' in the dashboard window to quit.")

    try:
        while not stop_event.is_set():

            # ── Drain result queue fully ──────────────────────────────────────
            drained = 0
            try:
                while True:
                    msg = result_queue.get_nowait()
                    if msg[0] != "frame":
                        continue

                    (_, idx,
                     boxes, track_ids, classes, confs, labels,
                     capture_time, model_lat, e2e_lat) = msg

                    s = stats[idx]
                    s["frames"] += 1
                    if s["start_time"] is None:
                        s["start_time"] = capture_time
                    s["end_time"]      = capture_time
                    s["last_model_ms"] = model_lat
                    s["last_e2e_ms"]   = e2e_lat

                    if s["frames"] > WARMUP_SKIP_FRAMES:
                        if len(s["model_latencies"]) >= 1000:
                            s["model_latencies"].pop(0)
                        s["model_latencies"].append(model_lat)
                        if len(s["pipeline_latencies"]) >= 1000:
                            s["pipeline_latencies"].pop(0)
                        s["pipeline_latencies"].append(e2e_lat)

                    # Read from display shm slot 2 — zero pickle cost
                    raw_frame = shm_display_views[idx].copy()
                    annotated = draw_boxes(raw_frame, boxes, track_ids, labels, confs)
                    latest_frames[idx] = annotated

                    det_list = []
                    if boxes is not None and labels is not None:
                        det_list = [
                            {
                                "label":    labels[j] if j < len(labels) else "?",
                                "conf":     float(confs[j]) if confs is not None else 0.0,
                                "track_id": int(track_ids[j]) if track_ids is not None else -1,
                                "box":      [float(x) for x in boxes[j]],
                            }
                            for j in range(len(boxes))
                        ]

                    reporter_msg = {
                        "stream_id":  idx,
                        "fps":        s["live_fps"],
                        "model_ms":   model_lat,
                        "e2e_ms":     e2e_lat,
                        "detections": det_list,
                    }
                    if s["frames"] % FRAME_SKIP == 0:
                        reporter_msg["frame"] = annotated

                    try:
                        reporter_q.put_nowait(reporter_msg)
                    except Exception:
                        pass

                    drained += 1

            except Empty:
                pass

            now_t = time.time()

            # ── Live FPS ──────────────────────────────────────────────────────
            for i, s in stats.items():
                elapsed = now_t - s["last_fps_time"]
                if elapsed >= 1.0:
                    s["live_fps"]        = (s["frames"] - s["last_fps_frames"]) / elapsed
                    s["last_fps_time"]   = now_t
                    s["last_fps_frames"] = s["frames"]

            # ── Update display thread state ───────────────────────────────────
            cpu = shared_cpu.value
            ram = shared_ram.value
            latest_display_state["frames"] = dict(latest_frames)
            latest_display_state["cpu"]    = cpu
            latest_display_state["ram"]    = ram

            # ── Quit via 'q' in display window ───────────────────────────────
            if quit_event.is_set():
                outer_stop.set()
                break

            # ── Assignment change from reporter ───────────────────────────────
            try:
                new_urls = assign_q.get_nowait()
                if new_urls and set(new_urls) != set(stream_urls):
                    with _restart_lock:
                        if not _restart_issued[0]:
                            _restart_issued[0] = True
                            log.info(f"Assignment changed (via reporter): {new_urls}")
                            print(f"[Main] Assignment changed — restarting with {new_urls}")
                            new_streams = new_urls
                            stop_event.set()
                            break
            except Exception:
                pass

            if outer_stop.is_set():
                break

            # ── System stats ──────────────────────────────────────────────────
            if len(system_cpu) >= 1000:
                system_cpu.pop(0)
                system_ram.pop(0)
            system_cpu.append(cpu)
            system_ram.append(ram)

            # ── Periodic log ──────────────────────────────────────────────────
            if now_t - last_log_time >= LOG_INTERVAL:
                for i in range(num_streams):
                    s = stats[i]
                    if s["model_latencies"]:
                        line = (f"Stream {i} | FPS: {s['live_fps']:.1f} | "
                                f"Model: {s['model_latencies'][-1]:.1f}ms | "
                                f"E2E: {s['pipeline_latencies'][-1]:.1f}ms")
                        print(line)
                        log.info(line)
                last_log_time = now_t

            # Yield briefly if queue was empty this tick
            if drained == 0:
                time.sleep(0.001)

    finally:
        stop_event.set()
        display_thread.join(timeout=2)

        all_procs = decoder_processes + [inference_p, monitor_p]
        for p in all_procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=3)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2)

        for shm_triple in shm_objects:
            for shm in shm_triple:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass

        print("\n=== Performance Report ===")
        for i in range(num_streams):
            s = stats[i]
            if not s["start_time"] or not s["end_time"]:
                continue
            duration = s["end_time"] - s["start_time"]
            fps      = compute_fps(s["frames"], duration)
            m_avg, m_p95, m_max = summarize_latency(s["model_latencies"])
            p_avg, p_p95, p_max = summarize_latency(s["pipeline_latencies"])
            print(f"\nPipeline {i+1} ({stream_urls[i]})")
            print(f"  Total Frames : {s['frames']} "
                  f"(stats from frame {WARMUP_SKIP_FRAMES+1} onwards)")
            print(f"  FPS          : {fps:.2f}")
            print(f"  Model Latency:")
            print(f"    Avg: {m_avg:.1f} ms | P95: {m_p95:.1f} ms | Max: {m_max:.1f} ms")
            print(f"  E2E Latency:")
            print(f"    Avg: {p_avg:.1f} ms | P95: {p_p95:.1f} ms | Max: {p_max:.1f} ms")

        if system_cpu:
            print(f"\nSystem: CPU {sum(system_cpu)/len(system_cpu):.1f}% avg | "
                  f"RAM {sum(system_ram)/len(system_ram):.1f}% avg")

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

    import socket as _socket
    resolved_node_id = node_id or cfg.node.id
    if not resolved_node_id or resolved_node_id in ('auto', 'node_01', 'my_laptop'):
        resolved_node_id = _socket.gethostname()

    cfg._data.setdefault('node', {})['id'] = resolved_node_id
    cfg._ns.node.id = resolved_node_id

    log_file = cfg.get('logging.file', 'auto')
    if not log_file or log_file == 'auto':
        os.makedirs('logs', exist_ok=True)
        cfg._data.setdefault('logging', {})['file'] = f'logs/{resolved_node_id}.log'

    try:
        import yaml as _yaml
        from pathlib import Path as _Path
        cfg_path = config_path or 'config.yaml'
        if _Path(cfg_path).exists():
            with open(cfg_path) as f:
                raw = _yaml.safe_load(f)
            if raw.get('node', {}).get('id') in ('auto', 'node_01', 'my_laptop', None):
                raw.setdefault('node', {})['id']      = resolved_node_id
                raw.setdefault('logging', {})['file'] = f'logs/{resolved_node_id}.log'
                with open(cfg_path, 'w') as f:
                    _yaml.dump(raw, f, default_flow_style=False, allow_unicode=True)
                print(f"[Main] Node ID saved to config: {resolved_node_id}")
    except Exception:
        pass

    setup_logging(cfg)

    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

    if stream_urls is None:
        stream_urls = []

    agent_mode = len(stream_urls) == 0
    outer_stop = multiprocessing.Event()

    reporter_q = multiprocessing.Queue(maxsize=256)
    assign_q   = multiprocessing.Queue(maxsize=8)

    if agent_mode:
        print("[Main] Agent mode — no streams on command line.")
        print(f"[Main] Connected to central at {cfg.central.host}:{cfg.central.port}")
        print("[Main] Assign streams from the dashboard to start processing.")

        try:
            import platform, psutil as _psutil
            base = f"http://{cfg.central.host}:{cfg.central.port}"
            requests.post(f"{base}/register", json={
                "node_id":     cfg.node.id,
                "hostname":    _socket.gethostname(),
                "cpu_cores":   _psutil.cpu_count(logical=True),
                "ram_gb":      round(_psutil.virtual_memory().total / (1024 ** 3), 1),
                "max_streams": cfg.get("node.max_streams", 4),
                "platform":    platform.system(),
            }, timeout=3)
            print("[Main] Registered with central server.")
        except Exception as e:
            print(f"[Main] Could not register: {e}")

        while True:
            try:
                base = f"http://{cfg.central.host}:{cfg.central.port}"
                r = requests.get(f"{base}/assignment",
                                 params={"node_id": cfg.node.id}, timeout=3)
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

    reporter_p = multiprocessing.Process(
        target=reporter_worker,
        args=(outer_stop, reporter_q, assign_q, cfg.to_dict()),
        name="reporter",
    )
    reporter_p.start()

    try:
        while not outer_stop.is_set() and stream_urls:
            new_streams = _run_pipeline(
                stream_urls, cfg, outer_stop, reporter_q, assign_q,
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
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        print("[Main] Session ended.")