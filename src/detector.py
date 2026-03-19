import os

os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENVINO_NUM_THREADS"] = "4"

import shutil
import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path='models/yolov8n.pt'):
        # NO batch_size parameter — removed completely

        if model_path.endswith('.pt'):
            ov_path = model_path.replace('.pt', '_openvino_model')
        else:
            ov_path = model_path

        if not os.path.exists(ov_path):
            print(f"[Detector] Exporting {model_path} to OpenVINO FP16...")
            base_model = YOLO(model_path)
            base_model.export(format='openvino', half=True, imgsz=320)
            # Move exported folder to correct location if needed
            raw = model_path.replace('.pt', '_openvino_model')
            if os.path.exists(raw) and raw != ov_path:
                if os.path.exists(ov_path):
                    shutil.rmtree(ov_path)
                shutil.move(raw, ov_path)
            print(f"[Detector] Export complete -> {ov_path}")
        else:
            print(f"[Detector] Using existing model at {ov_path}")

        print(f"[Detector] Loading {ov_path}...")
        self.model = YOLO(ov_path, task='detect')

    def warmup(self, num_iters=3):
        """
        Warm up OpenVINO kernels before real inference.
        Wrapped in try/except — failure must NOT crash inference worker.
        """
        dummy = np.zeros((256, 320, 3), dtype=np.uint8)
        print(f"[Detector] Warming up ({num_iters} passes)...")
        for i in range(num_iters):
            try:
                _ = self.model.predict(dummy, imgsz=320, verbose=False, device='cpu')
            except Exception as e:
                print(f"[Detector] Warmup pass {i+1} failed (non-fatal): {e}")
        print("[Detector] Warm-up complete.")

    def detect_raw(self, frames):
        """
        Accepts a single frame (np.ndarray) or list of frames.
        Returns single result or list of results.

        Per-frame loop — ultralytics OpenVINO backend does NOT support
        list input on a batch=1 model. Predictor is reused across calls
        so there is no re-init overhead per frame.
        """
        single = not isinstance(frames, list)
        if single:
            frames = [frames]

        results = []
        for frame in frames:
            res = self.model.predict(frame, imgsz=320, verbose=False, device='cpu')
            results.append(res[0])

        return results[0] if single else results