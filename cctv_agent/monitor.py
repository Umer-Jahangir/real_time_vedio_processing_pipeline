import time
import logging
import psutil


def monitor_worker(shared_cpu, shared_ram, stop_event, interval: float = 0.5):
    """
    Background process — samples CPU and RAM every `interval` seconds.
    Uses stop_event.wait(timeout) so it wakes immediately on shutdown.
    """
    log = logging.getLogger("monitor")
    log.info("Starting system monitoring.")
    print("[Monitor] Starting system monitoring...")

    while not stop_event.wait(timeout=interval):
        shared_cpu.value = psutil.cpu_percent(interval=None)
        shared_ram.value = psutil.virtual_memory().percent

    log.info("Stopped.")
    print("[Monitor] Stopping system monitoring.")


def get_system_usage(interval: float = 0.2):
    """Synchronous one-shot reading. Used for reporting."""
    return psutil.cpu_percent(interval=interval), psutil.virtual_memory().percent