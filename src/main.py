import cv2
import threading
import time
from detector import Detector
from monitor import get_system_usage
from utils import calculate_latency, calculate_fps

# -------------------------------
# Initialize 3 independent detectors
# -------------------------------
detectors = [Detector() for _ in range(3)]

# -------------------------------
# Initialize webcam
# -------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

shared_frame = None
lock = threading.Lock()

# Stream info
streams = [
    {"latency": 0, "fps": 0, "prev_time": 0, "frame": None}
    for _ in range(3)
]

# -------------------------------
# Camera reader thread
# -------------------------------
def camera_reader():
    global shared_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        with lock:
            shared_frame = frame.copy()

# -------------------------------
# Processing threads
# -------------------------------
def process_stream(index):
    global shared_frame
    detector = detectors[index]
    while True:
        frame = None
        while frame is None:
            with lock:
                if shared_frame is not None:
                    frame = shared_frame.copy()

        start = time.time()
        results = detector.detect(frame)
        end = time.time()

        streams[index]["latency"] = calculate_latency(start, end)
        streams[index]["fps"] = calculate_fps(streams[index]["prev_time"], end)
        streams[index]["prev_time"] = end
        streams[index]["frame"] = results

# -------------------------------
# Start threads
# -------------------------------
threading.Thread(target=camera_reader, daemon=True).start()

for i in range(3):
    threading.Thread(target=process_stream, args=(i,), daemon=True).start()

# -------------------------------
# Display loop (main thread)
# -------------------------------
while True:
    cpu, memory = get_system_usage()

    for i in range(3):
        frame = streams[i]["frame"]
        if frame is None:
            continue

        with lock:
            display = frame.copy()

        cv2.putText(display,
                    f"Pipeline {i+1} | Lat: {streams[i]['latency']:.1f} ms | FPS: {streams[i]['fps']:.1f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

        cv2.putText(display,
                    f"CPU: {cpu}% | RAM: {memory}%",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,255),
                    2)

        # Show each pipeline in its own window
        cv2.imshow(f"Pipeline {i+1}", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
