# Real-Time Multi-Stream Object Detection on CPU

**(YOLOv8n + OpenVINO + Multiprocessing + RTSP)**

Real-time object detection across **multiple RTSP streams** using **YOLOv8n optimized with OpenVINO** running entirely on **CPU**.

No GPU required.

The project focuses on **scaling inference pipelines**, **measuring latency**, and **understanding CPU bottlenecks** in real-world multi-stream deployments.

---

# Project Goal

Run **multiple RTSP video streams concurrently** with stable inference using:

* **YOLOv8n OpenVINO IR model**
* **Python multiprocessing**
* **Per-stage latency profiling**
* **System resource monitoring**

Primary objective:

Achieve **efficient multi-stream inference** while analyzing **scalability limits on CPU hardware**.

---

# Current Architecture

The system uses a **multi-process pipeline architecture**.

### Main Process

Responsible for:

* RTSP stream input handling
* Pipeline orchestration
* Performance monitoring
* Aggregating results

### Worker Processes

Each pipeline runs in an **independent process**.

Each worker performs:

1. Frame decoding
2. Preprocessing
3. YOLOv8 OpenVINO inference
4. Post-processing (drawing boxes)
5. Latency measurement

This avoids Python **GIL limitations** and enables **true CPU parallelism**.

---

# Pipeline Stages Measured

Each frame records latency for:

| Stage   | Description                 |
| ------- | --------------------------- |
| Read    | Frame capture from RTSP     |
| Preproc | Resize, normalization       |
| Model   | OpenVINO YOLOv8 inference   |
| Plot    | Bounding box rendering      |
| E2E     | End-to-end pipeline latency |

For each stage the system reports:

* Average latency
* P95 latency
* Maximum latency

---

# Example Usage

Run with one stream:

```bash
python main.py rtsp://localhost:8554/snap1
```

Run with multiple streams:

```bash
python main.py \
rtsp://localhost:8554/snap1 \
rtsp://localhost:8554/snap2 \
rtsp://localhost:8554/stream1 \
rtsp://localhost:8554/fb
```

Each stream launches **one independent inference pipeline**.

---

# Latest Performance Results

## 1 Pipeline

| Metric    | Value    |
| --------- | -------- |
| Frames    | 125      |
| FPS       | 4.91     |
| Model Avg | 14.79 ms |
| Model P95 | 23.23 ms |
| E2E Avg   | 20.62 ms |
| E2E Max   | 66.89 ms |

System Usage:

CPU: **60.9%**
RAM: **77.8%**

Observation:

Stable inference with low latency.

---

# 2 Pipelines

| Metric    | Pipeline 1 | Pipeline 2 |
| --------- | ---------- | ---------- |
| FPS       | 4.94       | 5.47       |
| Model Avg | 14.10 ms   | 15.38 ms   |
| Model Max | 40.00 ms   | 134.47 ms  |
| E2E Avg   | 19.28 ms   | 20.66 ms   |

System Usage:

CPU: **81.8%**
RAM: **78.2%**

Observation:

Near-linear scaling.

Some latency spikes begin appearing.

---

# 3 Pipelines

| Metric    | Pipeline 1 | Pipeline 2 | Pipeline 3 |
| --------- | ---------- | ---------- | ---------- |
| FPS       | 4.61       | 5.35       | 7.83       |
| Model Avg | 25.70 ms   | 28.72 ms   | 25.09 ms   |
| Model Max | 102.57 ms  | 162.89 ms  | 147.96 ms  |
| E2E Avg   | 33.93 ms   | 37.76 ms   | 37.36 ms   |

System Usage:

CPU: **91.7%**
RAM: **87.1%**

Observation:

CPU approaching saturation.

Model latency roughly doubled.

Spike frequency increased.

---

# 4 Pipelines

| Metric    | Pipeline 1 | Pipeline 2 | Pipeline 3 | Pipeline 4 |
| --------- | ---------- | ---------- | ---------- | ---------- |
| FPS       | 3.89       | 5.22       | 6.49       | 6.17       |
| Model Avg | 41.54 ms   | 39.35 ms   | 37.10 ms   | 42.25 ms   |
| Model Max | 148.98 ms  | 254.22 ms  | 115.02 ms  | 150.27 ms  |
| E2E Avg   | 51.53 ms   | 48.67 ms   | 51.88 ms   | 55.60 ms   |

System Usage:

CPU: **92%**
RAM: **85%**

Observation:

CPU fully saturated.

Large latency spikes occur frequently.

RTSP decoding errors appear under heavy load.

---

# Spike Detection

The system detects abnormal latency events.

Example logs:

```
[MODEL SPIKE] Pipeline 2 | Latency 254.22 ms | CPU 99.7%
[E2E SPIKE] Pipeline 2 | Latency 263.68 ms | CPU 99.7%
```

Spikes usually correlate with:

* CPU contention
* OS scheduling delays
* frame decode backlog

---

# Key Findings

1. **CPU becomes the primary bottleneck beyond two pipelines**

2. **Model latency increases sharply once CPU >90%**

3. **RTSP decoding errors appear under heavy load**

4. **Throughput scaling is limited by CPU scheduling**

5. **Each pipeline loading its own model increases CPU pressure**

---

# Current Limitations

* CPU-only inference
* Each worker loads its own OpenVINO model
* No cross-stream batching
* RTSP decoding handled inside worker
* No adaptive frame skipping

---

# Future Improvements

### Shared Model Execution

Avoid loading a model per worker.

### Asynchronous Pipeline

Separate threads for:

* capture
* inference
* rendering

### Frame Batching

Batch frames across streams for improved CPU efficiency.

### Adaptive Load Control

Implement:

* frame skipping
* dynamic FPS throttling

---

# Research Value

This project demonstrates real-world challenges in:

* **CPU-only multi-stream inference**
* **OpenVINO performance tuning**
* **Python multiprocessing scalability**
* **RTSP decoding under heavy load**

The goal is to evolve this into a **production-grade real-time video analytics pipeline**.

---

# Tech Stack

* Python
* OpenCV
* OpenVINO
* YOLOv8
* Multiprocessing
* RTSP streaming
* Performance profiling tools
