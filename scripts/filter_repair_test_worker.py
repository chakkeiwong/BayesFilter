"""Configure the campaign device before pytest imports numerical test modules."""

import json
import os
import resource
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.environ.get("FILTER_REPAIR_SOURCE_ROOT", str(Path(__file__).resolve().parents[1])))
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def sample_main_thread(stop, thread_id):
    """GIL-safe diagnostic snapshots; no asynchronous signal stack walking."""
    started = time.monotonic()
    while not stop.wait(45):
        print(json.dumps({"diagnostic_elapsed_seconds": time.monotonic() - started,
                          "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),
              file=sys.stderr, flush=True)
        frame = sys._current_frames().get(thread_id)
        if frame is not None:
            traceback.print_stack(frame, file=sys.stderr)
        del frame


if __name__ == "__main__":
    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(
        tf, require_gpu=os.environ.get("CUDA_VISIBLE_DEVICES") != "-1")
    print(json.dumps({"tensorflow_version": tf.__version__, "gpu_memory_policy": memory_policy,
                      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
                      "trust_basis": "owner_designated_managed_session_visible_gpu_trusted"
                      if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "explicit_cpu_reference"}), flush=True)
    import pytest

    # The registered localization group disables pytest's signal timer. A
    # Python watchdog records later stages while respecting the main GIL.
    repeat_stacks = os.environ.get("FILTER_REPAIR_REPEAT_STACKS") == "1"
    stop = threading.Event()
    if repeat_stacks:
        threading.Thread(target=sample_main_thread,
                         args=(stop, threading.main_thread().ident), daemon=True).start()
    exit_code = None
    try:
        exit_code = pytest.main(sys.argv[1:])
    finally:
        stop.set()
        if repeat_stacks:
            print(json.dumps({
                "diagnostic_final_host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "pytest_exit_code": exit_code,
            }), file=sys.stderr, flush=True)
    raise SystemExit(exit_code)
