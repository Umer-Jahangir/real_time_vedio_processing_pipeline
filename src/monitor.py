import time
import psutil
import multiprocessing

def monitor_worker(shared_cpu, shared_ram, stop_event, interval=0.5):
    """
    Background process to monitor system resources without blocking
    the main application loop.
    """
    print("[Monitor] Starting system monitoring...")
    while not stop_event.is_set():
        # Update shared memory with current usage
        shared_cpu.value = psutil.cpu_percent(interval=None)
        shared_ram.value = psutil.virtual_memory().percent
        
        # Monitor interval: how often to sample (e.g., 0.5 seconds)
        time.sleep(interval)
    print("[Monitor] Stopping system monitoring.")

# Optionally, keep this for compatibility with your existing code
def get_system_usage(interval=0.2):
    # This is still used in the main loop for initialization, 
    # but the monitor_worker does the continuous work.
    return psutil.cpu_percent(interval=interval), psutil.virtual_memory().percent
