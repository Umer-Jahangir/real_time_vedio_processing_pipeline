import cv2
import time
from detector import Detector
from utils import now, latency_ms


def process_worker(index, video_path, output_q, stop_event, real_time=True):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        output_q.put(("error", index, "Cannot open video"))
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if video_fps == 0:
        video_fps = 30.0

    frame_interval = 1.0 / video_fps

    output_q.put(("info", index, width, height, video_fps))

    detector = Detector()

    PROCESS_WIDTH = 320
    PROCESS_HEIGHT = 256

    frame_count = 0
    dropped_frames = 0

    while not stop_event.is_set():
        loop_start = now()

        # ---------------- Capture ----------------
        capture_time = now()
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------- Preprocess ----------------
        preprocess_start = now()
        frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        preprocess_end = now()

        # ---------------- Model ----------------
        model_start = now()
        frame = detector.detect(frame)
        model_end = now()

        # ---------------- Send ----------------
        send_time = now()

        preprocess_latency = latency_ms(preprocess_start, preprocess_end)
        model_latency = latency_ms(model_start, model_end)
        total_pipeline_latency = latency_ms(capture_time, send_time)

        frame_count += 1

        try:
            output_q.put_nowait((
                "frame",
                index,
                frame,
                capture_time,
                preprocess_latency,
                model_latency,
                total_pipeline_latency
            ))
        except:
            dropped_frames += 1

        # ---------------- Real-Time Pacing ----------------
        if real_time:
            elapsed = now() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    cap.release()

    output_q.put((
        "done",
        index,
        frame_count,
        dropped_frames
    ))