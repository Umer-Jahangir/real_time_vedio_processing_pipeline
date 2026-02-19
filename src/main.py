import cv2
import time
import multiprocessing
from detector import Detector
from monitor import get_system_usage
from utils import calculate_latency, calculate_fps

# -------------------------------
# Worker function for each process
# -------------------------------
def process_worker(index, input_q, output_q):
    # Initialize detector inside the child process
    detector = Detector()
    prev_time = 0
    
    while True:
        frame = input_q.get() # Waits efficiently for a frame
        if frame is None: break

        start = time.time()
        processed_frame = detector.detect(frame)
        end = time.time()

        latency = calculate_latency(start, end)
        fps = calculate_fps(prev_time, end)
        prev_time = end

        # Send back the frame and metrics
        output_q.put((index, processed_frame, latency, fps))

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

    # Queues for inter-process communication
    input_queues = [multiprocessing.Queue(maxsize=1) for _ in range(3)]
    output_queue = multiprocessing.Queue()

    # Start 3 independent processes
    processes = []
    for i in range(3):
        p = multiprocessing.Process(target=process_worker, args=(i, input_queues[i], output_queue))
        p.daemon = True
        p.start()
        processes.append(p)

    latest_data = {i: None for i in range(3)}

    try:
        while True:
            ret, frame = cap.read()
            if not ret: continue

            # Distribute the same frame to all 3 pipelines
            for q in input_queues:
                if q.empty():
                    q.put(frame.copy())

            # Collect results from the output queue
            while not output_queue.empty():
                idx, out_frame, lat, fps = output_queue.get()
                latest_data[idx] = (out_frame, lat, fps)

            cpu, memory = get_system_usage()

            # Display windows
            for i in range(3):
                if latest_data[i] is not None:
                    display, lat, fps = latest_data[i]
                    
                    cv2.putText(display, f"Pipeline {i+1} | Lat: {lat:.1f}ms | FPS: {fps:.1f}", 
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(display, f"CPU: {cpu}% | RAM: {memory}%", 
                                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    cv2.imshow(f"Pipeline {i+1}", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
