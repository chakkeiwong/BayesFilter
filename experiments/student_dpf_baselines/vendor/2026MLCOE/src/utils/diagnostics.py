"""
Diagnostics Module
==================
Tools for monitoring system resources during filter execution.
"""

import psutil
import tracemalloc

def get_process_stats():
    """
    Returns the current resource utilization.

    Used in the benchmarking scripts to compare the hardware efficiency
    of PF (high particle count) vs. PFF (low particle count).

    Returns:
        tuple: (peak_memory_mb, cpu_percentage)
    """
    # Get peak memory usage since tracemalloc.start()
    _, peak = tracemalloc.get_traced_memory()
    peak_mb = peak / 10**6

    # Get CPU usage (non-blocking)
    cpu_pct = psutil.cpu_percent(interval=None)
    return peak_mb, cpu_pct
