import multiprocessing
import os
import sys
import time
import cv2
from queue import Empty
import psutil

from worker import process_worker
from monitor import monitor_worker
from display import show_frame, show_finished
from utils import compute_fps, summarize_latency

REAL_TIME = True
SPIKE_MODEL_MS = 100
SPIKE_PIPELINE_MS = 150

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # ---------- Get stream URLs from command line, or fallback to videos folder ----------
    if len(sys.argv) < 2:
        print("No stream URLs provided. Using videos from 'videos/' folder.")
        base_dir = os.path.dirname(__file__)
        videos_dir = os.path.join(base_dir, "videos")
        if not os.path.exists(videos_dir):
            raise RuntimeError("'videos/' folder not found. Please create it and add video files.")
        stream_urls = sorted([os.path.join(videos_dir, f) for f in os.listdir(videos_dir) if os.path.isfile(os.path.join(videos_dir, f))])
        if not stream_urls:
            raise RuntimeError("No video files found in 'videos/' folder.")
    else:
        stream_urls = sys.argv[1:]   # treat all arguments as stream URLs

    num_streams = len(stream_urls)
    print(f"Starting {num_streams} pipelines.")

    # Shared memory for system monitoring
    shared_cpu = multiprocessing.Value('d', 0.0)
    shared_ram = multiprocessing.Value('d', 0.0)
    stop_event = multiprocessing.Event()

    monitor_p = multiprocessing.Process(target=monitor_worker, args=(shared_cpu, shared_ram, stop_event))
    monitor_p.start()

    physical_cores = psutil.cpu_count(logical=False)
    num_pipelines = min(physical_cores, num_streams)
    print(f"Using {num_pipelines} pipelines (physical cores: {physical_cores})")

    output_q = multiprocessing.Queue(maxsize=300)
    processes = []

    # Launch workers with per‑pipeline divisors
    for i in range(num_pipelines):
        divisor = 5 if i == 0 else 3   # first pipeline gets larger divisor
        p = multiprocessing.Process(target=process_worker,
                                     args=(i, stream_urls[i], output_q, stop_event, REAL_TIME, divisor))
        p.start()
        processes.append(p)

    # Stats initialisation
    stats = {i: {
        "frames": 0,
        "preprocess_latencies": [],
        "model_latencies": [],
        "plot_latencies": [],
        "read_latencies": [],
        "pipeline_latencies": [],
        "start_time": None,
        "end_time": None
    } for i in range(num_pipelines)}

    latest_frames = {i: None for i in range(num_pipelines)}
    finished = {i: False for i in range(num_pipelines)}
    system_cpu = []
    system_ram = []

    try:
        while True:
            # Drain queue
            try:
                while True:
                    msg = output_q.get_nowait()
                    tag = msg[0]
                    idx = msg[1]

                    if tag == "frame":
                        (_, idx, frame, capture_time,
                         preprocess_latency, model_latency,
                         plot_latency, read_latency,
                         pipeline_latency) = msg

                        s = stats[idx]
                        if s["start_time"] is None:
                            s["start_time"] = capture_time
                        s["end_time"] = capture_time
                        s["frames"] += 1
                        s["preprocess_latencies"].append(preprocess_latency)
                        s["model_latencies"].append(model_latency)
                        s["plot_latencies"].append(plot_latency)
                        s["read_latencies"].append(read_latency)
                        s["pipeline_latencies"].append(pipeline_latency)

                        latest_frames[idx] = (frame, model_latency, pipeline_latency)

                        # Spike detection
                        curr_cpu = shared_cpu.value
                        if model_latency > SPIKE_MODEL_MS:
                            print(f"[MODEL SPIKE] Pipeline {idx+1} | Latency {model_latency:.2f} ms | CPU {curr_cpu:.1f}%")
                        if pipeline_latency > SPIKE_PIPELINE_MS:
                            print(f"[E2E SPIKE] Pipeline {idx+1} | Latency {pipeline_latency:.2f} ms | CPU {curr_cpu:.1f}%")

                    elif tag == "done":
                        print(f"Pipeline {idx+1} received 'done' signal.")
                        finished[idx] = True
                        latest_frames[idx] = None
            except Empty:
                pass

            # System metrics
            cpu = shared_cpu.value
            ram = shared_ram.value
            system_cpu.append(cpu)
            system_ram.append(ram)

            # Display windows
            for i in range(num_pipelines):
                window_name = f"Pipeline {i+1}"
                if latest_frames[i] is not None:
                    frame, model_lat, pipe_lat = latest_frames[i]
                    duration = (stats[i]["end_time"] - stats[i]["start_time"]) if stats[i]["start_time"] else 1e-6
                    fps = compute_fps(stats[i]["frames"], duration)
                    show_frame(window_name, frame, model_lat, pipe_lat, cpu, ram, fps)
                elif finished[i]:
                    show_finished(window_name)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if all(finished.values()):
                print("All pipelines finished. Closing...")
                break

    finally:
        stop_event.set()
        monitor_p.join(timeout=2)
        for p in processes:
            p.join(timeout=2)
        cv2.destroyAllWindows()

        # Final report (unchanged)
        print("\n=== Performance Report ===")
        for i in range(num_pipelines):
            s = stats[i]
            duration = (s["end_time"] - s["start_time"]) if s["start_time"] and s["end_time"] else 0
            fps = compute_fps(s["frames"], duration)

            model_avg, model_p95, model_max = summarize_latency(s["model_latencies"])
            plot_avg, plot_p95, plot_max = summarize_latency(s["plot_latencies"])
            pre_avg, pre_p95, pre_max = summarize_latency(s["preprocess_latencies"])
            read_avg, read_p95, read_max = summarize_latency(s["read_latencies"])
            pipe_avg, pipe_p95, pipe_max = summarize_latency(s["pipeline_latencies"])

            print(f"\nPipeline {i+1}")
            print(f"  Frames: {s['frames']}")
            print(f"  FPS: {fps:.2f}")
            print(f"  Preproc -> Avg: {pre_avg:.2f} ms | P95: {pre_p95:.2f} | Max: {pre_max:.2f}")
            print(f"  Model   -> Avg: {model_avg:.2f} ms | P95: {model_p95:.2f} | Max: {model_max:.2f}")
            print(f"  Plot    -> Avg: {plot_avg:.2f} ms | P95: {plot_p95:.2f} | Max: {plot_max:.2f}")
            print(f"  Read    -> Avg: {read_avg:.2f} ms | P95: {read_p95:.2f} | Max: {read_max:.2f}")
            print(f"  E2E     -> Avg: {pipe_avg:.2f} ms | P95: {pipe_p95:.2f} | Max: {pipe_max:.2f}")

        if system_cpu:
            avg_cpu = sum(system_cpu) / len(system_cpu)
            avg_ram = sum(system_ram) / len(system_ram)
            print("\nSystem Usage:")
            print(f"  Avg CPU: {avg_cpu:.2f}%")
            print(f"  Avg RAM: {avg_ram:.2f}%")