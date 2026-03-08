import cv2
import time
import numpy as np
import queue
import os
from detector import Detector
from utils import now, latency_ms

# Time to wait before reconnecting after a stream failure (seconds)
RECONNECT_DELAY = 2.0

def process_worker(index, stream_url, output_q, stop_event, real_time=True, send_divisor=3):
    # Keep trying to open the stream until stop_event is set
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    while not stop_event.is_set():
        cap = cv2.VideoCapture(stream_url)
        if cap.isOpened():
            break
        print(f"[Worker {index}] Failed to open stream {stream_url}. Retrying in {RECONNECT_DELAY}s...")
        time.sleep(RECONNECT_DELAY)

    if not cap.isOpened():
        output_q.put(("error", index, f"Cannot open stream after multiple attempts: {stream_url}"))
        return

    # Read stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if video_fps == 0:
        video_fps = 30.0   # fallback for streams that don't report FPS

    frame_interval = 1.0 / video_fps
    output_q.put(("info", index, width, height, video_fps))

    detector = Detector()

    PROCESS_WIDTH = 320
    PROCESS_HEIGHT = 256

    # Warm‑up inference
    dummy_frame = np.zeros((PROCESS_HEIGHT, PROCESS_WIDTH, 3), dtype=np.uint8)
    _ = detector.detect(dummy_frame)
    print(f"[Worker {index}] Warm-up complete")

    frame_count = 0
    dropped_frames = 0
    skipped_frames = 0
    SEND_DIVISOR = send_divisor
    send_counter = 0

    while not stop_event.is_set():
        loop_start = now()

        # ---------------- Capture ----------------
        capture_time = now()
        ret, frame = cap.read()
        if not ret:
            print(f"[Worker {index}] Stream read failed. Attempting to reconnect...")
            cap.release()
            # Reconnection loop
            reconnected = False
            for attempt in range(5):   # try up to 5 times
                time.sleep(RECONNECT_DELAY)
                cap = cv2.VideoCapture(stream_url)
                if cap.isOpened():
                    reconnected = True
                    break
            if not reconnected:
                print(f"[Worker {index}] Could not reconnect to stream. Exiting.")
                break
            # Reconnected successfully – continue to next loop iteration
            continue

        read_latency = latency_ms(loop_start, capture_time)

        # ---------------- Preprocess ----------------
        preprocess_start = now()
        frame_resized = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        preprocess_end = now()
        preprocess_latency = latency_ms(preprocess_start, preprocess_end)

        # ---------------- Inference ----------------
        model_start = now()
        results = detector.detect_raw(frame_resized)
        model_end = now()
        model_latency = latency_ms(model_start, model_end)

        frame_count += 1
        send_counter += 1
        should_send = (send_counter % SEND_DIVISOR == 0)

        if should_send:
            plot_start = now()
            annotated_frame = results.plot()
            plot_end = now()
            plot_latency = latency_ms(plot_start, plot_end)

            send_time = now()
            total_pipeline_latency = latency_ms(capture_time, send_time)

            try:
                output_q.put_nowait((
                    "frame",
                    index,
                    annotated_frame,
                    capture_time,
                    preprocess_latency,
                    model_latency,
                    plot_latency,
                    read_latency,
                    total_pipeline_latency
                ))
            except queue.Full:
                dropped_frames += 1

        # ---------------- Real‑time pacing ----------------
        if real_time:
            elapsed = now() - loop_start
            if elapsed > 2 * frame_interval:
                # Skip one frame to catch up (but with a live stream, skipping is fine)
                skipped_frames += 1
                continue
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    cap.release()
    output_q.put(("done", index, frame_count, dropped_frames, skipped_frames))