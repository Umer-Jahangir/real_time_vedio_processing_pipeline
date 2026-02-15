from ultralytics import YOLO

class Detector:
    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, imgsz=320)
        return results[0].plot()
