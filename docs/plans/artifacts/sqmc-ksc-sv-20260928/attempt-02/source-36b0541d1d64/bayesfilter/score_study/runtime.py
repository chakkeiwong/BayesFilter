"""Explicit backend initialization and provenance, outside numerical kernels."""
from __future__ import annotations

import os
import sys

_configuration = None


def configure_runtime(*, device: str, tf32: bool, jit_compile: bool) -> dict:
    global _configuration
    requested = (device, tf32, jit_compile)
    if _configuration is not None:
        if _configuration[0] != requested:
            raise ValueError("one study process must use one device/TF32/JIT execution scope")
        return dict(_configuration[1])
    if device == "CPU":
        if "tensorflow" in sys.modules and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise RuntimeError("CPU reference must hide GPU before TensorFlow import")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device != "GPU":
        raise ValueError("device must be explicitly CPU or GPU")
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    if device == "GPU":
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory = dict(configure_tensorflow_gpu_memory_growth(tf, require_gpu=True))
    else:
        memory = {"mode": "cpu_reference_gpu_intentionally_hidden", "physical_devices": []}
    tf.config.experimental.enable_tensor_float_32_execution(tf32)
    metadata = {"backend": "tensorflow", "tensorflow_version": tf.__version__,
                "device": device, "gpu_intentionally_hidden": device == "CPU",
                "jit_compile": jit_compile, "tf32": tf32, "memory_policy": memory,
                "cpu_threads": 4, "reference_exception": device == "CPU" or not jit_compile}
    _configuration = (requested, metadata)
    return dict(metadata)


def memory_usage(device: str) -> dict:
    if device == "CPU":
        return {"allocator_current_bytes": None, "allocator_peak_bytes": None,
                "reason": "CPU reference; GPU intentionally hidden"}
    import tensorflow as tf
    info = tf.config.experimental.get_memory_info("GPU:0")
    return {"allocator_current_bytes": info["current"], "allocator_peak_bytes": info["peak"]}
