import os
# Must be set BEFORE ultralytics/OpenVINO are imported.
# Direct assignment (not setdefault) so inherited env vars are overridden.
os.environ["OMP_NUM_THREADS"]      = "4"
os.environ["OPENVINO_NUM_THREADS"] = "4"

import cv2
import time
import ctypes
import numpy as np
import psutil
import queue
import logging
import supervision as sv
from multiprocessing import shared_memory

from .detector import Detector
from .utils import now, latency_ms
from .config import Config, get_config


# ============================================================
# WINDOWS TIMER RESOLUTION
# ============================================================

def _set_timer_resolution():
    """
    Set Windows timer resolution to 1ms in THIS process.
    Must be called independently in every spawned subprocess —
    the setting is per-process and is NOT inherited from the parent.
    """
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass


# ============================================================
# CPU CORE ASSIGNMENT
# ============================================================

def get_pcore_logical_ids():
    freqs = psutil.cpu_freq(percpu=True)
    if not freqs or len(freqs) < 2:
        total_logical = psutil.cpu_count(logical=True)
        return list(range(4)) if total_logical > 8 else list(range(total_logical))
    max_freq  = max(f.max for f in freqs if f is not None)
    threshold = max_freq * 0.9
    return [i for i, f in enumerate(freqs) if f is not None and f.max >= threshold]


def assign_cores_hybrid(worker_index, num_decoders, cores_per_decoder, is_inference=False):
    total_logical = psutil.cpu_count(logical=True)
    pcore_ids     = get_pcore_logical_ids()
    ecore_ids     = [i for i in range(total_logical) if i not in pcore_ids]
    if is_inference:
        return pcore_ids
    start, end = worker_index * cores_per_decoder, (worker_index + 1) * cores_per_decoder
    if end > len(ecore_ids):
        print(f"Warning: insufficient E-cores for decoder {worker_index}, using any cores.")
        return list(range(total_logical))[start:end]
    return ecore_ids[start:end]


def _resolve_config(cfg_dict):
    """Reconstruct config from dict (zero disk I/O) or fall back to singleton."""
    if cfg_dict is not None:
        return Config.from_dict(cfg_dict)
    return get_config()


# ============================================================
# DECODER WORKER
# ============================================================

def decoder_worker(index, stream_url, shm_names, frame_shape,
                   meta_queue, stop_event, inference_alive,
                   num_decoders, cores_per_decoder,
                   cfg_dict=None):
    """
    Reads frames, writes into double-buffered shared memory slots 0 and 1.
    Slot 2 is the display buffer owned by inference_worker. Decoder NEVER
    touches slot 2.

    shm_names: (name_a, name_b, name_c) — 3 slots per stream.
    """
    _set_timer_resolution()

    log = logging.getLogger(f"decoder.{index}")
    cfg = _resolve_config(cfg_dict)

    p = psutil.Process()
    try:
        core_ids = assign_cores_hybrid(index, num_decoders, cores_per_decoder, is_inference=False)
        p.cpu_affinity(core_ids)
        print(f"[Decoder {index}] Pinned to cores {core_ids}")
        log.info(f"Pinned to cores {core_ids}")
    except Exception as e:
        log.warning(f"Affinity error: {e}")

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

    def open_stream(url):
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    cap = open_stream(stream_url)
    if not cap.isOpened():
        log.error(f"Failed to open stream: {stream_url}")
        print(f"[Decoder {index}] Failed to open stream")
        return

    try:
        # Open all 3 shm slots — decoder only writes slots 0 and 1
        shm_slots     = [shared_memory.SharedMemory(name=n) for n in shm_names]
        frame_buffers = [np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
                         for shm in shm_slots]
    except Exception as e:
        log.error(f"Shared memory error: {e}")
        cap.release()
        return

    print(f"[Decoder {index}] Waiting for inference...")
    while not stop_event.is_set() and not inference_alive.is_set():
        time.sleep(0.1)

    if stop_event.is_set():
        cap.release()
        for shm in shm_slots:
            shm.close()
        return

    log.info(f"Started — {stream_url}")
    print(f"[Decoder {index}] Started for {stream_url}")

    pace_sleep = 0.033
    PACE_MIN   = cfg.decoder.pace_min
    PACE_MAX   = cfg.get("decoder.pace_max_rtsp", cfg.decoder.pace_max)
    ALPHA_UP   = cfg.decoder.alpha_up
    ALPHA_DOWN = cfg.decoder.alpha_down
    READ_STALL_THRESHOLD = cfg.pipeline.stall_threshold_ms

    if os.path.isfile(stream_url):
        source_fps          = cap.get(cv2.CAP_PROP_FPS)
        file_frame_interval = (1.0 / source_fps) if source_fps > 0 else 0.033
        log.info(f"Local file — pacing to {source_fps:.1f} FPS")
        print(f"[Decoder {index}] Local file — pacing to {source_fps:.1f} FPS "
              f"({file_frame_interval*1000:.1f}ms/frame)")
    else:
        file_frame_interval = 0

    drop_count      = 0
    stall_count     = 0
    reconnect_delay = 2
    buf_idx         = 0   # alternates between 0 and 1 only — slot 2 is inference's

    while not stop_event.is_set():

        if not inference_alive.is_set():
            time.sleep(0.5)
            continue

        read_start = time.perf_counter()
        ret, frame = cap.read()
        read_ms    = (time.perf_counter() - read_start) * 1000

        if not ret:
            if os.path.isfile(stream_url):
                log.info("End of file.")
                print(f"[Decoder {index}] End of file")
                break
            log.warning(f"Stream lost — reconnecting in {reconnect_delay}s")
            print(f"[Decoder {index}] Reconnecting in {reconnect_delay}s")
            cap.release()
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 10)
            cap = open_stream(stream_url)
            if cap.isOpened():
                reconnect_delay = 2
            continue

        if read_ms > READ_STALL_THRESHOLD:
            stall_count += 1
            if stall_count % 20 == 0:
                log.warning(f"Stalled reads={stall_count} last={read_ms:.0f}ms")
                print(f"[Decoder {index}] stalled reads={stall_count} last={read_ms:.0f}ms")
            continue

        frame_resized = cv2.resize(
            frame, (frame_shape[1], frame_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        # Write only to slot 0 or 1 — slot 2 is owned by inference_worker
        np.copyto(frame_buffers[buf_idx], frame_resized)

        sent = False
        try:
            meta_queue.put_nowait({
                "stream_id": index,
                "buf_idx":   buf_idx,
                "timestamp": now(),
            })
            sent = True
        except queue.Full:
            drop_count += 1
            if drop_count % 100 == 0:
                log.warning(f"Dropped {drop_count} frames (queue full)")
                print(f"[Decoder {index}] dropped {drop_count}")

        buf_idx ^= 1   # toggle between 0 and 1 only

        try:
            q_size = meta_queue.qsize()
        except Exception:
            q_size = 0

        if file_frame_interval > 0:
            time.sleep(file_frame_interval)
        else:
            if not sent or q_size > 2:
                pace_sleep = min(pace_sleep * (1 + ALPHA_DOWN), PACE_MAX)
            elif q_size == 0:
                pace_sleep = max(pace_sleep * (1 - ALPHA_UP), PACE_MIN)
            time.sleep(pace_sleep)

    cap.release()
    for shm in shm_slots:
        shm.close()

    log.info(f"Stopped | Dropped={drop_count} | Stalled={stall_count}")
    print(f"[Decoder {index}] Stopped | Dropped={drop_count} | Stalled={stall_count}")


# ============================================================
# INFERENCE WORKER
# ============================================================

def inference_worker(num_streams, frame_shape, shm_names,
                     meta_queues, result_queue, stop_event,
                     inference_alive, num_decoders, cores_per_decoder,
                     cfg_dict=None, pipeline_start_time=None):
    """
    Per-frame sequential inference with metadata-only result messages.

    shm_names: list of (name_a, name_b, name_c) tuples, one per stream.
      Slot 0, 1: decoder double-buffer (reads buf_idx from meta)
      Slot 2:    display buffer — inference writes here after inference;
                 main reads here; no pickle, no copy through IPC.

    WHY PER-FRAME SEQUENTIAL (not batched):
    In the previous batched approach, the inference loop collected one
    frame from EVERY stream before running inference. With 3 streams and
    12ms per stream, stream 2 waited up to 24ms after its frame was
    captured before inference even started. E2E for stream 2 was stamped
    after the batch, inflating it by the cumulative wait.

    Per-frame: each frame is inferred immediately. E2E is stamped right
    after that frame's own inference. Streams don't wait for each other.
    The 1ms inter-frame gap (time.sleep(0.001) on empty queue) is
    negligible vs the 12ms inference time.

    WHY METADATA-ONLY RESULT MESSAGES:
    The previous version put the 245KB frame numpy array in every result
    message. result_queue.put_nowait() pickles synchronously. At 3 streams
    × 24 FPS = 72 pickle ops/sec of 245KB each, the OS scheduler regularly
    preempts the inference process mid-pickle for 50-150ms. During that
    block, decoders keep pushing frames. When inference resumes, those
    frames report 50-150ms E2E despite 12ms model time.

    FIX: frame goes into shm slot 2 (np.copyto, zero copy). Result message
    is only boxes + latency numbers (~2KB). Pickle time: <0.1ms.
    """
    _set_timer_resolution()

    log = logging.getLogger("inference")
    cfg = _resolve_config(cfg_dict)

    p = psutil.Process()
    try:
        core_ids = assign_cores_hybrid(0, num_decoders, cores_per_decoder, is_inference=True)
        if not core_ids:
            core_ids = [psutil.cpu_count(logical=False) - 1]
        p.cpu_affinity(core_ids)
        log.info(f"Pinned to cores {core_ids}")
        print(f"[Inference] Pinned to cores {core_ids}")
    except Exception as e:
        log.warning(f"Affinity error: {e}")

    detector    = Detector()
    class_names = detector.model.names

    # 3 slots per stream: [stream_idx][slot_idx]
    shm_handles, frame_buffers = [], []
    for name_triple in shm_names:
        slots = [shared_memory.SharedMemory(name=n) for n in name_triple]
        shm_handles.append(slots)
        frame_buffers.append([
            np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
            for shm in slots
        ])

    detector.warmup()
    time.sleep(0.05)   # let warmup settle before flushing

    for mq in meta_queues:
        while True:
            try:
                mq.get_nowait()
            except queue.Empty:
                break

    trackers        = [sv.ByteTrack() for _ in range(num_streams)]
    FRAME_AGE_LIMIT = 150   # ms — tight limit; stale frames inflate E2E

    if pipeline_start_time is None:
        pipeline_start_time = now() - 10000

    inference_alive.set()
    log.info("Ready — decoders can start")
    print("[Inference] Ready — decoders can start")

    result_drop_count = 0

    while not stop_event.is_set():

        processed_any = False

        for i, mq in enumerate(meta_queues):

            try:
                meta = mq.get_nowait()
            except queue.Empty:
                continue

            # Reject frames from before this pipeline incarnation
            if meta["timestamp"] < pipeline_start_time:
                continue

            # Reject stale frames
            if latency_ms(meta["timestamp"], now()) > FRAME_AGE_LIMIT:
                continue

            # Read from decoder slot (0 or 1)
            frame = frame_buffers[i][meta["buf_idx"]].copy()

            # Infer this frame immediately — no waiting for other streams
            infer_start = now()
            res         = detector.detect_raw(frame)
            model_ms    = latency_ms(infer_start, now())

            # Stamp E2E for THIS frame right after its own inference
            e2e_lat = latency_ms(meta["timestamp"], now())

            # Write raw frame to display slot 2 so main can read it without IPC.
            # Ownership: decoder writes 0/1, inference writes 2, main reads 2.
            # No two processes share any slot as writers.
            np.copyto(frame_buffers[i][2], frame)

            # Tracking
            boxes = classes = confs = labels = track_ids = None

            if res.boxes is not None:
                boxes   = res.boxes.xyxy.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy().astype(int)
                confs   = res.boxes.conf.cpu().numpy()
                labels  = [class_names[int(c)] for c in classes]

                detections = sv.Detections(xyxy=boxes, confidence=confs, class_id=classes)
                tracked    = trackers[meta["stream_id"]].update_with_detections(detections)

                boxes     = tracked.xyxy
                classes   = tracked.class_id
                confs     = tracked.confidence
                track_ids = tracked.tracker_id

            # Metadata-only result message — NO frame array.
            # Pickle: ~2KB instead of ~245KB.
            msg = (
                "frame",
                meta["stream_id"],
                # ← frame intentionally omitted; main reads from shm slot 2
                boxes,
                track_ids,
                classes,
                confs,
                labels,
                meta["timestamp"],
                model_ms,
                e2e_lat,
            )

            try:
                result_queue.put_nowait(msg)
            except queue.Full:
                result_drop_count += 1
                if result_drop_count % 50 == 0:
                    log.warning(f"Dropped {result_drop_count} results (queue full)")
                    print(f"[Inference] dropped results={result_drop_count}")

            processed_any = True

        if not processed_any:
            time.sleep(0.001)

    for slot_list in shm_handles:
        for shm in slot_list:
            shm.close()

    log.info(f"Stopped | Drops={result_drop_count}")
    print(f"[Inference] Stopped | Drops={result_drop_count}")