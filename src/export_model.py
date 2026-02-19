from ultralytics import YOLO

model = YOLO('models/yolov8n.pt')
model.export(format='openvino', imgsz=320)
print("Model exported to OpenVINO format")