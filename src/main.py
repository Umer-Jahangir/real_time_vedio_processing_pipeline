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
from worker import decoder_worker, inference_worker
from monitor import monitor_worker
from utils import compute_fps, summarize_latency


def compute_cores_per_decoder(num_streams):
    total = psutil.cpu_count(logical=False) or 4
    inference_reserve = max(2, total // 4)
    decoder_pool = total - inference_reserve
    per_decoder = max(1, decoder_pool // num_streams)
    while (per_decoder * num_streams) > (total - 2) and per_decoder > 1:
        per_decoder -= 1
    return per_decoder


def create_tiled_dashboard(frames_dict, num_streams):
    active = []
    ref_size = None
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
    if not active:
        return None
    return np.hstack(active)


def collect_video_sources(args):
    """Convert command-line arguments into a list of video sources.
       If an argument is a directory, include all video files inside.
       Supported extensions: .mp4, .avi, .mkv, .mov, .wmv, .flv, .mjpeg
    """
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.mjpeg')
    sources = []
    for arg in args:
        if os.path.isdir(arg):
            for file in sorted(os.listdir(arg)):  # sorted for determinism (#7)
                if file.lower().endswith(video_extensions):
                    full_path = os.path.join(arg, file)
                    sources.append(full_path)
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

        id_text = f"ID {track_id}"
        class_text = f"{cls}"
        conf_text = f"{conf:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 2
        padding = 4

        (w1, h1), _ = cv2.getTextSize(id_text, font, scale, thickness)
        (w2, h2), _ = cv2.getTextSize(class_text, font, scale, thickness)
        (w3, h3), _ = cv2.getTextSize(conf_text, font, scale, thickness)

        total_width = w1 + w2 + w3 + padding * 4
        label_height = max(h1, h2, h3) + padding * 2

        cv2.rectangle(
            frame,
            (x1, y1 - label_height),
            (x1 + total_width, y1),
            (255, 255, 255),
            -1
        )

        text_y = y1 - padding

        cv2.putText(frame, id_text, (x1 + padding, text_y),
                    font, scale, (0, 0, 0), thickness)
        cv2.putText(frame, class_text, (x1 + w1 + padding * 2, text_y),
                    font, scale, (0, 255, 0), thickness)
        cv2.putText(frame, conf_text, (x1 + w1 + w2 + padding * 3, text_y),
                    font, scale, (255, 0, 0), thickness)

    return frame


if __name__ == "__main__":

    multiprocessing.set_start_method("spawn", force=True)

    # Set Windows timer resolution to 1ms (default is ~15ms) so that
    # time.sleep() in decoders is accurate and decoder pacing jitter
    # is reduced from ±15ms to ±1ms.
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass  # non-Windows — no-op

    raw_args = sys.argv[1:]
    stream_urls = collect_video_sources(raw_args)

    if not stream_urls:
        print("Usage for test with local files: python main.py D:\Projects\real_time_vedio\src\videos")
        print("Usage for test with rtsp: python main.py rtsp://localhost:8554/Name1 ....")
        print("Sources can be video files, directories, or stream URLs (RTSP, HTTP, etc.)")
        sys.exit(1)

    num_streams = len(stream_urls)
    stop_event = multiprocessing.Event()
    inference_alive = multiprocessing.Event()

    cores_per_decoder = compute_cores_per_decoder(num_streams)
    total_cores = psutil.cpu_count(logical=False) or 4
    print(f"[Main] {total_cores} physical cores | "
          f"{cores_per_decoder} per decoder | "
          f"{num_streams} streams")
    print(f"[Main] Sources: {stream_urls}")

    # Shared memory — 2 slots (double-buffer) per stream (#1)
    frame_height, frame_width, frame_channels = 256, 320, 3
    frame_shape = (frame_height, frame_width, frame_channels)
    frame_size = frame_height * frame_width * frame_channels

    shm_objects = []
    shm_names = []

    for i in range(num_streams):
        shm_a = multiprocessing.shared_memory.SharedMemory(create=True, size=frame_size)
        shm_b = multiprocessing.shared_memory.SharedMemory(create=True, size=frame_size)
        shm_objects.append((shm_a, shm_b))
        shm_names.append((shm_a.name, shm_b.name))

    meta_queues = [multiprocessing.Queue(maxsize=8) for _ in range(num_streams)]
    result_queue = multiprocessing.Queue(maxsize=512)

    # Start inference
    inference_p = multiprocessing.Process(
        target=inference_worker,
        args=(num_streams, frame_shape, shm_names,
              meta_queues, result_queue, stop_event,
              inference_alive,
              num_streams, cores_per_decoder),
    )
    inference_p.start()

    # Start decoders
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

    # Start monitor
    shared_cpu = multiprocessing.Value('d', 0.0)
    shared_ram = multiprocessing.Value('d', 0.0)
    monitor_p = multiprocessing.Process(
        target=monitor_worker, args=(shared_cpu, shared_ram, stop_event)
    )
    monitor_p.start()

    # Stats
    stats = {i: {
        "frames": 0,
        "model_latencies": [],
        "pipeline_latencies": [],
        "start_time": None,
        "end_time": None,
        "last_fps_time": time.time(),
        "last_fps_frames": 0,
        "live_fps": 0.0,
    } for i in range(num_streams)}

    system_cpu = []
    system_ram = []
    latest_frames = {i: None for i in range(num_streams)}
    last_log_time = time.time()
    LOG_INTERVAL = 5.0

    # Dashboard rate limiter (#5)
    last_dashboard_time = 0.0
    DASHBOARD_INTERVAL = 1.0 / 30.0  # cap at 30 fps

    print("🚀 All pipelines started. Press 'q' to quit.")

    try:
        last_frame_time = time.time()

        while True:
            # Drain result queue — capped at 64 per iteration (#4)
            try:
                for _ in range(64):
                    msg = result_queue.get_nowait()
                    if msg[0] == "frame":
                        (
                            _,
                            idx,
                            frame,
                            boxes,
                            track_ids,
                            classes,
                            confs,
                            labels,
                            capture_time,
                            model_lat,
                            e2e_lat
                        ) = msg

                        s = stats[idx]
                        if s["start_time"] is None:
                            s["start_time"] = capture_time
                        s["end_time"] = capture_time
                        s["frames"] += 1
                        # Rolling window of 1000 samples (#10)
                        if len(s["model_latencies"]) >= 1000:
                            s["model_latencies"].pop(0)
                        s["model_latencies"].append(model_lat)
                        if len(s["pipeline_latencies"]) >= 1000:
                            s["pipeline_latencies"].pop(0)
                        s["pipeline_latencies"].append(e2e_lat)
                        # Draw on a copy to avoid mutating shared frame (#6)
                        frame = draw_boxes(frame.copy(), boxes, track_ids, labels, confs)
                        latest_frames[idx] = frame
                        last_frame_time = time.time()
            except Empty:
                pass

            # Live FPS
            now_t = time.time()
            for i, s in stats.items():
                elapsed = now_t - s["last_fps_time"]
                if elapsed >= 1.0:
                    s["live_fps"] = (s["frames"] - s["last_fps_frames"]) / elapsed
                    s["last_fps_time"] = now_t
                    s["last_fps_frames"] = s["frames"]

            # Exit when all decoders done and queue has been quiet for 2s
            # DEBUG print removed (#8)
            all_decoders_dead = all(not p.is_alive() for p in decoder_processes)
            if all_decoders_dead and (now_t - last_frame_time > 2.0):
                print("[Main] All streams finished. Exiting.")
                break

            # System stats — rolling window of 1000 samples (#9)
            cpu = shared_cpu.value
            ram = shared_ram.value
            if len(system_cpu) >= 1000:
                system_cpu.pop(0)
                system_ram.pop(0)
            system_cpu.append(cpu)
            system_ram.append(ram)

            # Periodic console log
            if now_t - last_log_time >= LOG_INTERVAL:
                for i in range(num_streams):
                    s = stats[i]
                    if s["model_latencies"]:
                        print(f"Stream {i} | FPS: {s['live_fps']:.1f} | "
                              f"Model: {s['model_latencies'][-1]:.1f}ms | "
                              f"E2E: {s['pipeline_latencies'][-1]:.1f}ms")
                last_log_time = now_t

            # Dashboard — rate-limited to 30fps (#5)
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
                break

    finally:
        print("[Main] Shutting down...")
        stop_event.set()

        all_procs = decoder_processes + [inference_p, monitor_p]
        for p in all_procs:
            p.join(timeout=5)
            if p.is_alive():
                print(f"[Main] PID {p.pid} still alive — terminating...")
                p.terminate()
                p.join(timeout=3)
                if p.is_alive():
                    print(f"[Main] PID {p.pid} not terminating — killing...")
                    p.kill()
                    p.join(timeout=2)

        for shm_a, shm_b in shm_objects:
            for shm in (shm_a, shm_b):
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass

        cv2.destroyAllWindows()
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        print("[Main] Cleanup complete.")

        print("\n=== Performance Report ===")
        for i in range(num_streams):
            s = stats[i]
            if not s["start_time"] or not s["end_time"]:
                print(f"\nPipeline {i+1}: No data collected.")
                continue

            duration = s["end_time"] - s["start_time"]
            fps = compute_fps(s["frames"], duration)
            model_avg, model_p95, model_max = summarize_latency(s["model_latencies"])
            pipe_avg,  pipe_p95,  pipe_max  = summarize_latency(s["pipeline_latencies"])

            print(f"\nPipeline {i+1}")
            print(f"  Frames : {s['frames']}")
            print(f"  FPS    : {fps:.2f}")
            print(f"  Model   -> Avg: {model_avg:.2f} ms | P95: {model_p95:.2f} | Max: {model_max:.2f}")
            print(f"  E2E     -> Avg: {pipe_avg:.2f} ms | P95: {pipe_p95:.2f} | Max: {pipe_max:.2f}")

        if system_cpu:
            print("\nSystem Usage:")
            print(f"  Avg CPU : {sum(system_cpu)/len(system_cpu):.2f}%")
            print(f"  Avg RAM : {sum(system_ram)/len(system_ram):.2f}%")