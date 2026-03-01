import multiprocessing
import os
import glob
import time
import cv2
from queue import Empty
import psutil

from worker import process_worker
from monitor import get_system_usage
from display import show_frame, show_finished
from utils import compute_fps, summarize_latency

REAL_TIME = True
SPIKE_MODEL_MS = 100       # threshold for model latency spike
SPIKE_PIPELINE_MS = 150    # threshold for E2E latency spike

def get_cpu_stats(interval=0.2):
    cores = psutil.cpu_percent(interval=interval, percpu=True)
    return sum(cores) / len(cores), max(cores)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    base_dir = os.path.dirname(__file__)
    videos_dir = os.path.join(base_dir, "videos")

    video_files = sorted([v for v in glob.glob(os.path.join(videos_dir, "*")) if os.path.isfile(v)])
    if not video_files:
        raise RuntimeError("No video files found.")

    # Get number of physical cores (avoid hyperthreading contention)
    physical_cores = psutil.cpu_count(logical=False)
    # Use at most physical cores, but also respect video count
    num_pipelines = min(physical_cores, len(video_files))
    print(f"Using {num_pipelines} pipelines (physical cores: {physical_cores})")

    output_q = multiprocessing.Queue(maxsize=300)
    stop_event = multiprocessing.Event()

    # start processes
    processes = []
    for i in range(num_pipelines):
        p = multiprocessing.Process(target=process_worker,
                                    args=(i, video_files[i], output_q, stop_event, REAL_TIME))
        p.start()
        processes.append(p)

    # Metrics initialization
    stats = {i: {"frames": 0, "model_latencies": [], "pipeline_latencies": [],
                 "start_time": None, "end_time": None} for i in range(num_pipelines)}
    latest_frames = {i: None for i in range(num_pipelines)}
    finished = {i: False for i in range(num_pipelines)}
    system_cpu = []
    system_ram = []

    try:
        while True:
            # drain queue
            try:
                while True:
                    msg = output_q.get_nowait()
                    tag = msg[0]

                    if tag == "frame":
                        _, idx, frame, capture_time, preprocess_latency, model_latency, pipeline_latency = msg
                        s = stats[idx]
                        if s["start_time"] is None:
                            s["start_time"] = capture_time
                        s["end_time"] = capture_time
                        s["frames"] += 1
                        s["model_latencies"].append(model_latency)
                        s["pipeline_latencies"].append(pipeline_latency)
                        latest_frames[idx] = (frame, model_latency, pipeline_latency)

                        # detect spikes
                        if model_latency > SPIKE_MODEL_MS:
                            cpu_avg, cpu_max = get_cpu_stats()
                            print(f"[MODEL SPIKE] Pipeline {idx+1} | Latency {model_latency:.2f} ms | CPU {cpu_max:.1f}%")
                        if pipeline_latency > SPIKE_PIPELINE_MS:
                            cpu_avg, cpu_max = get_cpu_stats()
                            print(f"[E2E SPIKE] Pipeline {idx+1} | Latency {pipeline_latency:.2f} ms | CPU {cpu_max:.1f}%")

                    elif tag == "done":
                        # Unpack depending on message length (for backward compatibility)
                        if len(msg) == 5:
                            _, idx, frame_count, dropped_frames, skipped_frames = msg
                        else:
                            _, idx, frame_count, dropped_frames = msg
                            skipped_frames = 0
                        finished[idx] = True
                        latest_frames[idx] = None
                        # You could store skipped_frames in stats if you want to track it
            except Empty:
                pass

            cpu, ram = get_system_usage(0.2)
            system_cpu.append(cpu)
            system_ram.append(ram)

            # display
            for i in range(num_pipelines):
                if latest_frames[i] is not None:
                    frame, model_lat, pipe_lat = latest_frames[i]
                    duration = (stats[i]["end_time"] - stats[i]["start_time"]) if stats[i]["start_time"] and stats[i]["end_time"] else 1e-6
                    fps = compute_fps(stats[i]["frames"], duration)
                    show_frame(f"Pipeline {i+1}", frame, model_lat, pipe_lat, cpu, ram, fps)
                elif finished[i]:
                    show_finished(f"Pipeline {i+1}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

            if all(finished.values()):
                break

    finally:
        stop_event.set()
        time.sleep(0.5)
        for p in processes:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        cv2.destroyAllWindows()

        # Final report
        print("\n=== Performance Report ===")
        for i in range(num_pipelines):
            s = stats[i]
            duration = (s["end_time"] - s["start_time"]) if s["start_time"] and s["end_time"] else 0
            fps = compute_fps(s["frames"], duration)
            model_avg, model_p95, model_max = summarize_latency(s["model_latencies"])
            pipe_avg, pipe_p95, pipe_max = summarize_latency(s["pipeline_latencies"])
            print(f"\nPipeline {i+1}")
            print(f"  Frames: {s['frames']}")
            print(f"  FPS: {fps:.2f}")
            print(f"  Model  -> Avg: {model_avg:.2f} ms | P95: {model_p95:.2f} | Max: {model_max:.2f}")
            print(f"  E2E    -> Avg: {pipe_avg:.2f} ms | P95: {pipe_p95:.2f} | Max: {pipe_max:.2f}")
        if system_cpu:
            avg_cpu = sum(system_cpu) / len(system_cpu)
            avg_ram = sum(system_ram) / len(system_ram)
            print("\nSystem Usage:")
            print(f"  Avg CPU: {avg_cpu:.2f}%")
            print(f"  Avg RAM: {avg_ram:.2f}%")