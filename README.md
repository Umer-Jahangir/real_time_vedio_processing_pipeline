# Real-Time Multi-Stream Object Detection on CPU  
**(YOLOv8n + OpenVINO + Multiprocessing)**

Optimized a real-time object detection system using **YOLOv8n** for multiple video streams on Intel CPU — **no GPU required**.

## Progress so far (Week 2)

- **Single-threaded baseline**  
  ~65 ms latency, 12–15 FPS

- **Multiprocessing** (bypassed Python GIL)  
  ~93–125 ms latency, 7.5–8 FPS  
  → **4× speedup**

- **OpenVINO optimizations** (export to IR, graph fusion, quantization)  
  **24–27 ms** latency, **28–32 FPS** stable across 3 parallel pipelines  
  → **~16× overall improvement** from original baseline

- Detection confidence maintained at **~0.91**

## Current Architecture

- Main process: frame capture
- Multiple worker processes: each runs an independent OpenVINO-optimized YOLOv8n instance
- Queue-based communication for low overhead

## Key Achievements

- Achieved true real-time performance (**~30 FPS**) on CPU
- Demonstrated massive gains from hardware-aware optimization and proper concurrency
- Learned critical lessons: multiprocessing > threading, profile before optimizing, avoid blind pattern application

## Next Steps

- Support true independent multi-stream (different video sources)
- Scale to 5–7+ concurrent streams
- Add profiling, adaptive quality control, monitoring & error handling
- Improve documentation & architecture diagrams

A practical showcase of iterative optimization, Python concurrency tradeoffs, and Intel OpenVINO acceleration for real-world computer vision.
