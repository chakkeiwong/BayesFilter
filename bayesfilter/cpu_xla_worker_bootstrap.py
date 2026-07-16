"""Spawn-safe CPU/XLA worker bootstrap with no framework imports at module load."""

from __future__ import annotations

import importlib
import multiprocessing
import os
import sys
from typing import Any, Mapping


_FRAMEWORK_MODULES = ("tensorflow", "tensorflow_probability", "jax", "torch")
_WORKER_FUNCTION: Any | None = None
_WORKER_DIMENSION: int | None = None
_WORKER_BOOTSTRAP: Mapping[str, Any] | None = None


def initialize_cpu_xla_worker(
    factory_path: str,
    dimension: int,
    factory_config: Mapping[str, Any],
    set_affinity: bool,
) -> None:
    """Hide accelerators before importing TensorFlow, then compile static ``B=1``."""

    cuda_at_entry = os.environ.get("CUDA_VISIBLE_DEVICES")
    growth_at_entry = os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")
    if cuda_at_entry != "-1" or str(growth_at_entry).lower() != "true":
        raise RuntimeError(
            "CPU/XLA worker must inherit CUDA_VISIBLE_DEVICES=-1 and "
            "TF_FORCE_GPU_ALLOW_GROWTH=true before process bootstrap"
        )
    frameworks_at_entry = tuple(
        name for name in _FRAMEWORK_MODULES if name in sys.modules
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    if set_affinity and hasattr(os, "sched_setaffinity"):
        available = sorted(os.sched_getaffinity(0))
        if available:
            process_slot = multiprocessing.current_process()._identity  # noqa: SLF001
            slot = 0 if not process_slot else (int(process_slot[0]) - 1) % len(available)
            os.sched_setaffinity(0, {available[slot]})

    import tensorflow as tf

    factory = _import_symbol(factory_path)
    function = factory(dict(factory_config))

    @tf.function(
        input_signature=(tf.TensorSpec([1, dimension], tf.float64),),
        jit_compile=True,
    )
    def compiled(rows: Any) -> tuple[Any, Any]:
        values, scores = function(rows)
        return (
            tf.ensure_shape(tf.convert_to_tensor(values, tf.float64), [1]),
            tf.ensure_shape(tf.convert_to_tensor(scores, tf.float64), [1, dimension]),
        )

    compiled(tf.zeros([1, dimension], tf.float64))
    global _WORKER_FUNCTION, _WORKER_DIMENSION, _WORKER_BOOTSTRAP
    _WORKER_FUNCTION = compiled
    _WORKER_DIMENSION = dimension
    _WORKER_BOOTSTRAP = {
        "cpu_env_inherited_before_initializer": True,
        "framework_modules_at_initializer_entry": list(frameworks_at_entry),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
        "jit_compile": True,
        "batch_size": 1,
    }


def evaluate_cpu_xla_row(
    task: tuple[int, list[float]],
) -> tuple[int, float, list[float], int, Mapping[str, Any]]:
    """Evaluate one row and return bootstrap provenance with the ordered result."""

    if (
        _WORKER_FUNCTION is None
        or _WORKER_DIMENSION is None
        or _WORKER_BOOTSTRAP is None
    ):
        raise RuntimeError("CPU/XLA worker was not initialized")
    import tensorflow as tf

    index, row = task
    values, scores = _WORKER_FUNCTION(
        tf.reshape(tf.convert_to_tensor(row, tf.float64), [1, _WORKER_DIMENSION])
    )
    return (
        index,
        float(values[0].numpy()),
        scores[0].numpy().tolist(),
        os.getpid(),
        dict(_WORKER_BOOTSTRAP),
    )


def _import_symbol(path: str) -> Any:
    module_name, symbol_name = str(path).split(":", 1)
    module = importlib.import_module(module_name)
    symbol = getattr(module, symbol_name)
    if not callable(symbol):
        raise TypeError("worker factory must be callable")
    return symbol
