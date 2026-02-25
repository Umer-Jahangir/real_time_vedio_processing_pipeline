import psutil

def get_system_usage(interval=0.5):
    """
    Stable CPU and memory sampling.
    interval: seconds to average CPU usage
    """
    cpu = psutil.cpu_percent(interval=interval)
    memory = psutil.virtual_memory().percent
    return cpu, memory


def get_per_core_usage(interval=0.5):
    """
    Returns per-core CPU utilization.
    """
    return psutil.cpu_percent(interval=interval, percpu=True)