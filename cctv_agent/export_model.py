from ultralytics import YOLO
import shutil, os

model = YOLO('models/yolov8n.pt')

# FP16 export — no nncf, no calibration data needed, works with any openvino version
model.export(format='openvino', half=True, imgsz=320)

# Move to correct location
src = 'yolov8n_openvino_model'
dst = 'models/yolov8n_openvino_model'
if os.path.exists(src):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)

print("✅ Export complete -> models/yolov8n_openvino_model")