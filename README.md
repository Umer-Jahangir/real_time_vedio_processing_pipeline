# Real-Time Multi-Stream Object Detection on CPU  
**(YOLOv8n + OpenVINO + Multiprocessing)**

Optimized real-time object detection for **multiple video streams** running entirely on **Intel CPU** — **no GPU required**.

## Project Goal
Process **4+ simultaneous video streams** with YOLOv8n using OpenVINO inference + multiprocessing, targeting stable real-time performance (~20–30 FPS per stream) on commodity hardware.

## Latest Performance (4 parallel pipelines)

Run on: Intel CPU (exact model not specified), Windows, venv  
Model: YOLOv8n exported to OpenVINO IR (likely INT8 quantized)

### Per-pipeline summary (after initialization / warm-up phase)

| Pipeline | Frames Processed | Avg FPS | Model Latency Avg (ms) | Model P95 (ms) | Model Max (ms) | E2E Latency Avg (ms) | E2E P95 (ms) | E2E Max (ms) |
|----------|------------------|---------|------------------------|----------------|----------------|-----------------------|--------------|--------------|
| 1        | 590              | 14.8    | 27.0                   | 33.3           | 1816           | 65.1                  | 176.7        | 1927         |
| 2        | 651              | 24.4    | 24.9                   | 28.9           | 2262           | 39.5                  | 47.8         | 2371         |
| 3        | 1266             | 18.8    | 19.8                   | 28.0           | 1847           | 36.0                  | 52.2         | 1984         |
| 4        | 635              | 20.2    | 26.4                   | 30.9           | 1826           | 47.6                  | 55.9         | 1954         |

**System-wide averages**  
- CPU utilization: ~77%  
- RAM usage: ~83%  
- Very large cold-start / initialization spikes (1.8–2.3 seconds) observed once per pipeline  
- Occasional runtime spikes (150–500 ms) still present — mostly on pipeline 1

## Progress Highlights

**Starting point (single-threaded baseline)**  
~65 ms model latency → 12–15 FPS per stream

**After multiprocessing (4 workers)**  
Significant variability; best pipelines reach **~24 FPS**, worst ~15 FPS

**After OpenVINO optimizations** (IR export + LATENCY mode + likely INT8)  
- Typical model inference: **20–27 ms** (≈37–50 FPS theoretical single-stream)  
- End-to-end (capture → inference → post-processing → queue): **35–65 ms**  
- Stable region performance: **15–24 FPS per stream** (4 streams concurrently)  
- ~2–4× overall throughput improvement vs naive single-thread baseline

**Key remaining challenges**  
- Severe cold-start latency spikes (1.8–2.4 seconds) when pipelines initialize  
- Periodic large runtime spikes (150–600 ms) → frame drops / stuttering  
- Uneven load distribution across pipelines  
- Pipeline 1 consistently worse than others (possible affinity / scheduling issue)

## Current Architecture

- **Main process**: reads frames from 4 video sources, distributes to queues  
- **4 worker processes**: each has its own OpenVINO YOLOv8n inference engine  
- **Communication**: `multiprocessing.Queue` (frame buffers + results)  
- **Inference mode**: OpenVINO **LATENCY** hint, batch=1, CPU device  
- **Logging**: per-inference model latency + end-to-end pipeline latency + CPU/RAM usage

## Key Lessons Learned

- Multiprocessing >> threading for CPU-bound inference (bypasses GIL)  
- OpenVINO model compilation & first inference is **very expensive** — preload & warm-up all pipelines  
- Queue contention + OS scheduling can create large jitter/spikes even with low average latency  
- Per-process CPU affinity pinning may help reduce variability  
- Need better monitoring: per-stream frame drop count, lag detection, restart logic

## Next Steps / Roadmap

- **Preload & warm-up all models** before starting capture loop  
- Implement **pipeline health monitoring** → restart frozen/slow workers  
- Add **CPU affinity pinning** (one pipeline per physical core?)  
- Experiment with **OpenVINO THROUGHPUT** hint + larger batch size (if memory allows)  
- Support **true heterogeneous streams** (RTSP, webcam, file, different resolutions)  
- Add **adaptive quality** (skip frames / lower resolution on overload)  
- Create architecture diagram & detailed profiling breakdown  
- Target stable **≥20 FPS on all 4 streams** with <100 ms P99 latency

A practical deep-dive into Python concurrency trade-offs, OpenVINO optimization patterns, and the real-world challenges of achieving stable multi-stream inference on CPU.
