import time

def calculate_latency(start, end):
    return (end - start) * 1000

def calculate_fps(prev_time, current_time):
    if prev_time == 0:
        return 0
    return 1 / (current_time - prev_time)
