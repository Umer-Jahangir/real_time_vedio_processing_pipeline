from ultralytics import YOLO
import os

class Detector:
    def __init__(self, use_openvino=True):
        if use_openvino and os.path.exists('models/yolov8n_openvino_model'):
            # Use OpenVINO optimized model
            self.model = YOLO('models/yolov8n_openvino_model', task='detect')
            print(f"Loaded OpenVINO model")
        else:
            # Fallback to regular PyTorch model
            self.model = YOLO('models/yolov8n.pt')
            print(f"Loaded PyTorch model")

    def detect(self, frame):
        results = self.model(frame, imgsz=320, verbose=False)
        return results[0].plot()

    def detect_raw(self, frame):
        """
        Runs inference and returns the raw Results object (no drawing).
        """
        results = self.model(frame, imgsz=320, verbose=False)
        return results[0]   # return the first (and only) Results object