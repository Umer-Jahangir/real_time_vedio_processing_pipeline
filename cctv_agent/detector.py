import os
import shutil
import logging
import numpy as np

# Limit thread pools before importing OpenVINO/ultralytics
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENVINO_NUM_THREADS", "4")

from ultralytics import YOLO
from .config import get_config


class Detector:
    """YOLOv8 + OpenVINO inference wrapper. Reads all settings from config."""

    def __init__(self, model_path: str = None):
        log = logging.getLogger("detector")
        cfg = get_config()

        if model_path is None:
            model_path = cfg.get("model.path", "models/yolov8n.pt") if cfg else "models/yolov8n.pt"

        self._imgsz        = cfg.get("model.imgsz",        320)  if cfg else 320
        self._confidence   = cfg.get("model.confidence",   0.25) if cfg else 0.25
        self._warmup_iters = cfg.get("model.warmup_iters", 3)    if cfg else 3

        if model_path.endswith(".pt"):
            ov_path = model_path.replace(".pt", "_openvino_model")
        else:
            ov_path = model_path

        if not os.path.exists(ov_path):
            log.info(f"Exporting {model_path} to OpenVINO FP16...")
            print(f"[Detector] Exporting {model_path} to OpenVINO FP16...")
            base_model = YOLO(model_path)
            base_model.export(format="openvino", half=True, imgsz=self._imgsz)
            raw = model_path.replace(".pt", "_openvino_model")
            if os.path.exists(raw) and raw != ov_path:
                if os.path.exists(ov_path):
                    shutil.rmtree(ov_path)
                shutil.move(raw, ov_path)
            log.info(f"Export complete → {ov_path}")
            print(f"[Detector] Export complete -> {ov_path}")
        else:
            log.info(f"Using existing model at {ov_path}")
            print(f"[Detector] Using existing model at {ov_path}")

        log.info(f"Loading {ov_path} (imgsz={self._imgsz}, conf={self._confidence})")
        print(f"[Detector] Loading {ov_path}...")
        self.model = YOLO(ov_path, task="detect")

    def warmup(self, num_iters: int = None):
        """Warm up OpenVINO kernels. Non-fatal on failure."""
        log = logging.getLogger("detector")
        if num_iters is None:
            num_iters = self._warmup_iters

        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        log.info(f"Warming up ({num_iters} passes)...")
        print(f"[Detector] Warming up ({num_iters} passes)...")

        for i in range(num_iters):
            try:
                _ = self.model.predict(dummy, imgsz=self._imgsz, verbose=False, device="cpu")
            except Exception as e:
                log.warning(f"Warmup pass {i+1} failed (non-fatal): {e}")
                print(f"[Detector] Warmup pass {i+1} failed (non-fatal): {e}")

        log.info("Warm-up complete.")
        print("[Detector] Warm-up complete.")

    def detect_raw(self, frames):
        """
        Run inference on a single frame or list of frames.
        Per-frame loop — OpenVINO batch=1 model does not support list input.
        """
        single = not isinstance(frames, list)
        if single:
            frames = [frames]

        results = []
        for frame in frames:
            res = self.model.predict(
                frame, imgsz=self._imgsz,
                conf=self._confidence,
                verbose=False, device="cpu"
            )
            results.append(res[0])

        return results[0] if single else results