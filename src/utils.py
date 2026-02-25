# High precision timing utilities

import time
import statistics


def now():
    return time.perf_counter()


def latency_ms(start, end):
    return (end - start) * 1000.0


def compute_fps(frames, duration):
    if duration <= 0:
        return 0.0
    return frames / duration


# ---------- Advanced Metrics ----------

def percentile(data, p):
    """
    Compute percentile (e.g., p=95 for P95 latency)
    """
    if not data:
        return 0.0
    return statistics.quantiles(data, n=100)[p - 1]


def summarize_latency(latencies):
    """
    Returns avg, p95, max latency
    """
    if not latencies:
        return 0.0, 0.0, 0.0

    avg = sum(latencies) / len(latencies)
    p95 = percentile(latencies, 95)
    maximum = max(latencies)

    return avg, p95, maximum