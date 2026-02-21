import cv2
import time
import multiprocessing
import os
import glob
import numpy as np
import statistics

from detector import Detector
from monitor import get_system_usage
from utils import calculate_latency, calculate_fps


def process_worker(index, video_path, output_q, stop_event):
    cap = cv2.VideoCapture(video_path)
    
    # ... info code ...
    
    detector = Detector()
    prev_time = 0
    frame_count = 0
    dropped_frames = 0  # Track dropped frames
    
    PROCESS_WIDTH = 320
    PROCESS_HEIGHT = 256
    loop_count = 0
    max_loops = 2
    
    last_log_time = time.time()
    
    while not stop_event.is_set():
        iteration_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            loop_count += 1
            if loop_count >= max_loops:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_resized = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        
        start = time.time()
        processed_frame = detector.detect(frame_resized)
        end = time.time()
        
        latency = calculate_latency(start, end)
        fps = calculate_fps(prev_time, end)
        prev_time = end
        
        frame_count += 1
        
        # NON-BLOCKING PUT
        try:
            output_q.put_nowait(("frame", index, processed_frame, latency, fps, time.time()))
        except:
            dropped_frames += 1  # Count drops
        
        # Log every 5 seconds
        if time.time() - last_log_time > 5.0:
            print(f"Worker {index}: Processed {frame_count} frames, dropped {dropped_frames}")
            last_log_time = time.time()
        
        # CRITICAL: Check iteration time
        iteration_time = (time.time() - iteration_start) * 1000
        if iteration_time > 100:
            print(f"Worker {index} SLOW ITERATION: {iteration_time:.1f}ms (latency={latency:.1f}ms)")
    
    cap.release()
    print(f"Worker {index} exiting: Processed {frame_count}, dropped {dropped_frames}")
    
    # Send done
    try:
        output_q.put(("done", index, frame_count), timeout=1.0)
    except:
        pass


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    videos_dir = os.path.join(base_dir, "vedios")
    
    all_videos = sorted(glob.glob(os.path.join(videos_dir, "*")))
    video_files = [p for p in all_videos if os.path.isfile(p)]
    
    if not video_files:
        raise RuntimeError(f"No videos found in {videos_dir}")
    
    num_pipelines = min(4, len(video_files))
    selected_videos = video_files[:num_pipelines]
    
    # BIGGER QUEUE
    output_queue = multiprocessing.Queue(maxsize=100)
    stop_event = multiprocessing.Event()
    
    # Start processes
    processes = []
    for i, video_path in enumerate(selected_videos):
        p = multiprocessing.Process(
            target=process_worker,
            args=(i, video_path, output_queue, stop_event)
        )
        p.start()
        processes.append(p)
    
    # Stats
    stats = {}
    latest_data = {i: None for i in range(num_pipelines)}
    finished = {i: False for i in range(num_pipelines)}
    
    for i in range(num_pipelines):
        stats[i] = {
            "frames": 0,
            "total_latency": 0.0,
            "min_latency": float("inf"),
            "max_latency": 0.0,
            "cpu_samples": [],
            "mem_samples": [],
            "first_ts": None,
            "last_ts": None,
            "width": None,
            "height": None,
            "video_fps": None,
            "reported_frame_count": 0,
        }
    
    try:
        while True:
            # PRIORITY: Empty queue FAST
            msgs_processed = 0
            while not output_queue.empty() and msgs_processed < 100:
                try:
                    msg = output_queue.get_nowait()
                    tag = msg[0]
                    
                    if tag == "info":
                        _, idx, w, h, vfps = msg
                        stats[idx]["width"] = w
                        stats[idx]["height"] = h
                        stats[idx]["video_fps"] = vfps
                    
                    elif tag == "frame":
                        _, idx, frame, lat, fpst, ts = msg
                        
                        s = stats[idx]
                        s["frames"] += 1
                        s["total_latency"] += lat
                        s["min_latency"] = min(s["min_latency"], lat)
                        s["max_latency"] = max(s["max_latency"], lat)
                        if s["first_ts"] is None:
                            s["first_ts"] = ts
                        s["last_ts"] = ts
                        
                        cpu, memory = get_system_usage()
                        s["cpu_samples"].append(cpu)
                        s["mem_samples"].append(memory)
                        
                        latest_data[idx] = (frame, lat, fpst)
                    
                    elif tag == "done":
                        _, idx, reported_count = msg
                        finished[idx] = True
                        stats[idx]["reported_frame_count"] = reported_count
                        latest_data[idx] = None
                    
                    msgs_processed += 1
                except:
                    break
            
            # Display AFTER emptying queue
            for i in range(num_pipelines):
                if latest_data[i] is not None:
                    display, lat, fps = latest_data[i]
                    if display is not None and display.size > 0:
                        cpu, memory = get_system_usage()
                        cv2.putText(display, f"Pipeline {i+1} | Lat: {lat:.1f}ms | FPS: {fps:.1f}",
                                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(display, f"CPU: {cpu}% | RAM: {memory}%",
                                   (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.imshow(f"Pipeline {i+1}", display)
                
                elif finished[i]:
                    placeholder = np.zeros((256, 320, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"Pipeline {i+1} finished", (10, 130),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow(f"Pipeline {i+1}", placeholder)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
            
            if all(finished.values()):
                break
    
    finally:
        stop_event.set()
        time.sleep(0.5)
        
        for p in processes:
            p.join(timeout=2.0)
            if p.is_alive():
                print(f"Force terminating worker {p.pid}")
                p.terminate()
        
        cv2.destroyAllWindows()
        
        # Report
        print("\n=== Per-pipeline performance report ===")
        for i in range(num_pipelines):
            s = stats[i]
            frames = s["frames"]
            duration = 0.0
            if s["first_ts"] and s["last_ts"] and s["last_ts"] > s["first_ts"]:
                duration = s["last_ts"] - s["first_ts"]
            
            approx_fps = (frames / duration) if duration > 0 else 0.0
            avg_latency = (s["total_latency"] / frames) if frames > 0 else 0.0
            min_latency = s["min_latency"] if s["min_latency"] != float("inf") else 0.0
            max_latency = s["max_latency"]
            avg_cpu = statistics.mean(s["cpu_samples"]) if s["cpu_samples"] else 0.0
            avg_mem = statistics.mean(s["mem_samples"]) if s["mem_samples"] else 0.0
            
            print(f"Pipeline {i+1} ({selected_videos[i]}):")
            print(f"  resolution: {s['width']}x{s['height']}")
            print(f"  source_fps: {s['video_fps']}")
            print(f"  frames_processed: {frames}")
            print(f"  reported_frame_count (worker): {s['reported_frame_count']}")
            print(f"  duration(s): {duration:.2f}")
            print(f"  approx_fps: {approx_fps:.2f}")
            print(f"  avg_latency(ms): {avg_latency:.2f}, min: {min_latency:.2f}, max: {max_latency:.2f}")
            print(f"  avg_cpu_percent: {avg_cpu:.2f}, avg_ram_percent: {avg_mem:.2f}\n")