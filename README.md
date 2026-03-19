# Real-Time Distributed CCTV Monitoring System

A high-performance, scalable CCTV monitoring system built for real-time multi-stream object detection using YOLOv8 + OpenVINO on CPU. Designed to run on edge nodes that auto-register with a central server and appear on a live dashboard — no manual configuration required for end users.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Current Implementation](#current-implementation)
- [Benchmark Results](#benchmark-results)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [How to Run](#how-to-run)
- [Node Registration Flow](#node-registration-flow)
- [Dashboard Interface](#dashboard-interface)
- [Roadmap — What's Next](#roadmap--whats-next)
- [Advantages](#advantages)

---

## Project Overview

This system consists of two major components:

### 1. Central Server + Dashboard
- Runs on a single machine (laptop or dedicated server).
- Hosts the web-based monitoring dashboard.
- Assigns streams, settings, and models to edge nodes.
- Tracks node health, CPU usage, and online/offline status.

### 2. Edge Agent (`cctv_agent`)
- Lightweight Python package installed on CCTV nodes.
- Auto-connects to the central server on startup.
- Downloads configuration and model files from the server.
- Starts video decoding, YOLOv8 inference, and reporting pipelines.
- No manual configuration required on the node side.

---

## Current Implementation

The current version (`src/main.py`) is a standalone multi-stream inference engine that serves as the core pipeline for the edge agent. It handles:

- **CPU affinity pinning** — Each decoder is pinned to 2 physical cores; inference uses 4 dedicated cores.
- **Multi-stream decoding** — Each stream runs in its own thread with frame-paced playback (25 FPS for local files).
- **YOLOv8n OpenVINO inference** — Single shared inference worker in LATENCY mode (batch=1) on CPU.
- **System monitoring** — Tracks CPU and RAM usage throughout the session.
- **Performance reporting** — Per-pipeline FPS, model latency (avg/P95/max), and end-to-end latency on shutdown.
- **Zero frame drops** — All sessions ended with `Dropped=0` across all decoders.

### Supported Input Sources
- Local video files (`.mp4`, `.avi`, etc.)
- RTSP streams (`rtsp://...`)

---

## Benchmark Results

Tests run on a **10-core CPU** with **3 concurrent streams**, using **YOLOv8n OpenVINO (CPU, LATENCY mode, batch=1)**.

### Local File Playback (`python main.py D:\Projects\real_time_vedio\src\videos`)

| Pipeline | Frames | FPS   | Model Avg | Model P95 | Model Max | E2E Avg | E2E P95 | E2E Max |
|----------|--------|-------|-----------|-----------|-----------|---------|---------|---------|
| 1        | 2157   | 23.88 | 8.66 ms   | 9.32 ms   | 10.94 ms  | 22.94 ms| 37.11 ms| 43.34 ms|
| 2        | 2157   | 23.89 | 8.61 ms   | 9.23 ms   | 12.68 ms  | 23.48 ms| 38.11 ms| 45.21 ms|
| 3        | 2137   | 23.67 | 8.58 ms   | 9.22 ms   | 10.84 ms  | 22.95 ms| 37.70 ms| 48.42 ms|

**System Usage:** Avg CPU `43.03%` | Avg RAM `71.30%`

---

### RTSP Stream Playback (`python main.py rtsp://localhost:8554/cctv1 rtsp://localhost:8554/cctv3 rtsp://localhost:8554/cctv4`)

| Pipeline | Frames | FPS   | Model Avg | Model P95 | Model Max | E2E Avg | E2E P95 | E2E Max |
|----------|--------|-------|-----------|-----------|-----------|---------|---------|---------|
| 1        | 2509   | 25.17 | 9.02 ms   | 9.86 ms   | 11.67 ms  | 26.77 ms| 35.45 ms| 47.52 ms|
| 2        | 2522   | 25.17 | 8.90 ms   | 9.59 ms   | 11.92 ms  | 29.66 ms| 36.93 ms| 48.26 ms|
| 3        | 2513   | 25.18 | 8.90 ms   | 9.86 ms   | 14.51 ms  | 30.63 ms| 51.66 ms| 58.73 ms|

**System Usage:** Avg CPU `46.59%` | Avg RAM `78.28%`

---

### Key Observations

- Model inference is rock-stable: **8.6–9.0 ms average** across all pipelines, both local and RTSP.
- FPS holds at **~24–25 FPS** for all 3 streams simultaneously with zero drops.
- E2E latency is higher on RTSP due to network jitter and buffer management — expected behavior.
- CPU headroom (~53–57% idle) leaves room to scale to 4–5 streams or add post-processing logic.
- RTSP produced minor H.264 decode warnings (`Missing reference picture`) on stream start — harmless, clears within the first few frames.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Central Server                       │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │  REST API   │    │   Dashboard  │    │  Model    │  │
│  │  (FastAPI)  │    │  (Web UI)    │    │  Store    │  │
│  └──────┬──────┘    └──────────────┘    └───────────┘  │
└─────────┼───────────────────────────────────────────────┘
          │ HTTP / WebSocket
┌─────────┼───────────────────────────────────────────────┐
│         ▼         Edge Node (cctv_agent)                │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │  Controller │───▶│  Decoder(s)  │───▶│ Inference │  │
│  │  (config)   │    │  (per stream)│    │ (YOLOv8n) │  │
│  └─────────────┘    └──────────────┘    └─────┬─────┘  │
│                                               │         │
│  ┌─────────────┐    ┌──────────────┐          │         │
│  │  Reporter   │◀───│   Monitor    │◀─────────┘         │
│  │  (to server)│    │  (CPU/RAM)   │                    │
│  └─────────────┘    └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
your_project/
│
├── central/                        # Central server + dashboard
│   ├── server.py
│   ├── dashboard.html
│   ├── config.yaml
│   └── requirements.txt
│
├── agent/                          # Python package for edge nodes
│   ├── cctv_agent/
│   │   ├── __init__.py
│   │   ├── main.py                 # Entry point: python -m cctv_agent
│   │   ├── worker.py               # Multi-stream pipeline manager
│   │   ├── detector.py             # YOLOv8 OpenVINO inference
│   │   ├── monitor.py              # CPU/RAM monitoring
│   │   ├── reporter.py             # Reports stats back to server
│   │   ├── controller.py           # Handles server config and registration
│   │   ├── config.py               # Default configuration constants
│   │   └── utils.py                # Shared utilities
│   ├── setup.py                    # Makes it pip installable
│   └── requirements.txt
│
├── src/                            # Current standalone implementation
│   ├── main.py                     # CLI entry point
│   ├── models/
│   │   └── yolov8n_openvino_model/ # Pre-converted OpenVINO model
│   └── videos/                     # Test video files
│
└── README.md
```

---

## How to Run

### Current Standalone Version

```bash
cd src

# Local video files
python main.py path/to/videos/

# RTSP streams
python main.py rtsp://192.168.1.10:8554/cam1 rtsp://192.168.1.10:8554/cam2

# Press 'q' to quit — performance report prints on exit
```

---

### Central Server (Planned)

1. Install dependencies:

```bash
pip install -r central/requirements.txt
```

2. Start the server:

```bash
python central/server.py
```

3. Open `dashboard.html` in your browser.

---

### Edge Node (Planned)

1. Install the agent:

```bash
pip install https://github.com/<your_username>/<repo>/releases/download/v1.0.0/cctv_agent-1.0.0.tar.gz
```

2. Run the agent (single command, nothing else needed):

```bash
python -m cctv_agent --server 192.168.1.100
```

The node auto-registers, downloads its config and model, and streams appear on the dashboard automatically.

---

## Node Registration Flow

```
Edge Node Startup
      │
      ▼
POST /register  ──────────────────────────────────────────▶  Central Server
{                                                              │
  "node_id": "node_02",                                       │ Assigns streams,
  "hardware": {                                               │ model URL, settings
    "cpu_cores": 8,                                           │
    "ram_gb": 16,                                             ▼
    "platform": "Windows"                           Response to Node:
  }                                                 {
}                                                     "streams": ["rtsp://cam4", "rtsp://cam5"],
                                                      "model_url": "http://server/models/yolov8n.pt",
                                                      "settings": { "imgsz": 320, "confidence": 0.25 }
                                                    }
      │
      ▼
Node downloads model (if not cached)
      │
      ▼
Pipelines start → Stats reported back to server every N seconds
```

---

## Dashboard Interface

- **Node management panel** — online/offline status, CPU usage, streams assigned per node.
- **Stream assignment** — drag-and-drop or dropdown to assign RTSP streams to nodes.
- **Auto rebalance** — distribute streams evenly across available nodes.
- **Add new streams** — paste an RTSP URL, assign to a node, done.
- **Live stats** — per-node FPS, model latency, and E2E latency updated in real time.

---

## Roadmap — What's Next

The current engine is solid. Here's what's being built on top of it:

### Phase 1 — Central Server + REST API
- [ ] FastAPI server with `/register`, `/config`, `/stats` endpoints
- [ ] Model file hosting and download endpoint
- [ ] Node registry with heartbeat tracking (mark nodes offline after timeout)
- [ ] Config persistence (YAML or SQLite)

### Phase 2 — Edge Agent Package
- [ ] Wrap current `main.py` pipeline into `cctv_agent` Python package
- [ ] `python -m cctv_agent --server <ip>` entry point
- [ ] Auto-registration on startup with hardware info
- [ ] Periodic stats push to server (`/stats` POST every 5s)
- [ ] Model cache: skip download if model already exists locally

### Phase 3 — Dashboard
- [ ] Node list with status indicators (green/red)
- [ ] Per-node stream assignment UI
- [ ] Live FPS and latency display per pipeline
- [ ] Stream health alerts (dropped frames, stall detection)
- [ ] Add / remove stream URLs from the UI

### Phase 4 — Production Hardening
- [ ] RTSP reconnection on stream drop (exponential backoff)
- [ ] Graceful node restart without losing registration
- [ ] Multi-model support (assign different models to different nodes)
- [ ] GPU / NPU inference support (OpenVINO GPU plugin)
- [ ] Docker packaging for central server
- [ ] pip-installable release of `cctv_agent` via GitHub Releases

---

## Advantages

- **Zero config on edge nodes** — operators install and run one command.
- **Centralized control** — manage all nodes and streams from a single web interface.
- **Proven inference performance** — ~24–25 FPS per stream, ~8.6–9.0 ms model latency, zero drops.
- **CPU-only** — no GPU required; runs on commodity hardware using OpenVINO optimization.
- **Horizontally scalable** — add more nodes to handle more streams.
- **Clean architecture** — packaged edge agent keeps logic modular and maintainable.
- **Cross-platform** — runs on Windows, Linux, and macOS (Python + OpenVINO required).

---

## Requirements

```
Python >= 3.9
ultralytics
openvino
opencv-python
numpy
psutil
```

Model: `yolov8n` exported to OpenVINO IR format (`yolov8n_openvino_model/`).

---

## License

MIT — see `LICENSE` for details.