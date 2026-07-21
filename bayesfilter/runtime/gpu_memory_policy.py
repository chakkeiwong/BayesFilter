"""TensorFlow GPU memory-allocation policy helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


TF_GPU_MEMORY_POLICY_SCHEMA = "bayesfilter.tensorflow.gpu_memory_policy.v1"


class TensorFlowGPUMemoryPolicyError(RuntimeError):
    """Raised when the required TensorFlow GPU memory policy cannot be applied."""


def configure_tensorflow_gpu_memory_growth(
    tf: Any,
    *,
    require_gpu: bool = True,
) -> Mapping[str, Any]:
    """Enable and verify memory growth before TensorFlow initializes GPU runtime."""

    physical_devices = tuple(tf.config.list_physical_devices("GPU"))
    if require_gpu and not physical_devices:
        raise TensorFlowGPUMemoryPolicyError(
            "TensorFlow GPU memory-growth policy requires a visible physical GPU"
        )

    rows = []
    for device in physical_devices:
        name = str(getattr(device, "name", device))
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError as exc:
            raise TensorFlowGPUMemoryPolicyError(
                "TensorFlow GPU memory growth must be configured before logical "
                f"device or tensor initialization: {name}"
            ) from exc
        try:
            enabled = bool(tf.config.experimental.get_memory_growth(device))
        except (RuntimeError, ValueError) as exc:
            raise TensorFlowGPUMemoryPolicyError(
                f"cannot verify TensorFlow GPU memory growth: {name}"
            ) from exc
        if not enabled:
            raise TensorFlowGPUMemoryPolicyError(
                f"TensorFlow GPU memory growth was not enabled: {name}"
            )
        rows.append({"device": name, "memory_growth": True})

    return {
        "schema": TF_GPU_MEMORY_POLICY_SCHEMA,
        "mode": "memory_growth",
        "physical_devices": tuple(rows),
        "all_physical_devices_memory_growth": True,
        "full_device_preallocation_disabled": True,
        "memory_growth_is_hard_memory_cap": False,
        "logical_device_memory_limit_required_for_hard_cap": True,
        "configured_before_logical_device_initialization": True,
        "tf_force_gpu_allow_growth": os.environ.get(
            "TF_FORCE_GPU_ALLOW_GROWTH", "unset"
        ),
    }


def configure_tensorflow_gpu_memory_limit(
    tf: Any,
    *,
    memory_limit_mib: int,
    require_gpu: bool = True,
) -> Mapping[str, Any]:
    """Apply and verify a fixed TensorFlow allocator limit per visible GPU."""

    if isinstance(memory_limit_mib, bool) or not isinstance(memory_limit_mib, int):
        raise TensorFlowGPUMemoryPolicyError("GPU memory limit must be an integer MiB value")
    if memory_limit_mib <= 0:
        raise TensorFlowGPUMemoryPolicyError("GPU memory limit must be positive")

    growth_env = os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "unset")
    if growth_env.strip().lower() in {"1", "true", "yes", "on"}:
        raise TensorFlowGPUMemoryPolicyError(
            "fixed GPU memory limit cannot be combined with TF_FORCE_GPU_ALLOW_GROWTH=true"
        )

    physical_devices = tuple(tf.config.list_physical_devices("GPU"))
    if require_gpu and not physical_devices:
        raise TensorFlowGPUMemoryPolicyError(
            "TensorFlow fixed-memory policy requires a visible physical GPU"
        )

    rows = []
    for device in physical_devices:
        name = str(getattr(device, "name", device))
        configuration = tf.config.LogicalDeviceConfiguration(
            memory_limit=memory_limit_mib
        )
        try:
            tf.config.set_logical_device_configuration(device, [configuration])
        except RuntimeError as exc:
            raise TensorFlowGPUMemoryPolicyError(
                "TensorFlow GPU memory limit must be configured before logical "
                f"device or tensor initialization: {name}"
            ) from exc
        try:
            configured = tf.config.get_logical_device_configuration(device)
        except (RuntimeError, ValueError) as exc:
            raise TensorFlowGPUMemoryPolicyError(
                f"cannot verify TensorFlow GPU memory limit: {name}"
            ) from exc
        if configured is None or len(configured) != 1:
            raise TensorFlowGPUMemoryPolicyError(
                f"TensorFlow GPU memory limit was not installed: {name}"
            )
        realized_limit = getattr(configured[0], "memory_limit", None)
        if realized_limit is None or float(realized_limit) != float(memory_limit_mib):
            raise TensorFlowGPUMemoryPolicyError(
                f"TensorFlow GPU memory limit verification failed: {name}"
            )
        rows.append({"device": name, "memory_limit_mib": memory_limit_mib})

    return {
        "schema": TF_GPU_MEMORY_POLICY_SCHEMA,
        "mode": "fixed_logical_device_limit",
        "physical_devices": tuple(rows),
        "memory_limit_mib_per_physical_device": memory_limit_mib,
        "aggregate_visible_gpu_limit_mib": memory_limit_mib * len(rows),
        "memory_growth": False,
        "hard_allocator_cap": True,
        "configured_before_logical_device_initialization": True,
        "tf_force_gpu_allow_growth": growth_env,
    }
