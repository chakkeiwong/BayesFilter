from __future__ import annotations

import pytest

from bayesfilter.runtime.gpu_memory_policy import (
    TF_GPU_MEMORY_POLICY_SCHEMA,
    TensorFlowGPUMemoryPolicyError,
    configure_tensorflow_gpu_memory_growth,
    configure_tensorflow_gpu_memory_limit,
)


class _Device:
    def __init__(self, name: str) -> None:
        self.name = name


class _ExperimentalConfig:
    def __init__(self, *, initialized: bool = False, ignore_set: bool = False) -> None:
        self.initialized = initialized
        self.ignore_set = ignore_set
        self.growth = {}

    def set_memory_growth(self, device, enabled: bool) -> None:
        if self.initialized:
            raise RuntimeError("runtime already initialized")
        if not self.ignore_set:
            self.growth[device.name] = bool(enabled)

    def get_memory_growth(self, device) -> bool:
        return bool(self.growth.get(device.name, False))

    def get_device_details(self, device):
        return {"device_name": f"Test {device.name}", "compute_capability": (8, 9)}


class _Config:
    def __init__(self, devices, experimental) -> None:
        self.devices = tuple(devices)
        self.experimental = experimental
        self.logical_configurations = {}

    class LogicalDeviceConfiguration:
        def __init__(self, *, memory_limit: int) -> None:
            self.memory_limit = memory_limit

    def list_physical_devices(self, kind: str):
        assert kind == "GPU"
        return self.devices

    def set_logical_device_configuration(self, device, configurations) -> None:
        if self.experimental.initialized:
            raise RuntimeError("runtime already initialized")
        self.logical_configurations[device.name] = tuple(configurations)

    def get_logical_device_configuration(self, device):
        return self.logical_configurations.get(device.name)


class _TensorFlow:
    def __init__(self, devices, experimental) -> None:
        self.config = _Config(devices, experimental)


def test_memory_growth_is_enabled_verified_and_described(monkeypatch) -> None:
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    devices = (_Device("/physical_device:GPU:0"), _Device("/physical_device:GPU:1"))
    policy = configure_tensorflow_gpu_memory_growth(
        _TensorFlow(devices, _ExperimentalConfig())
    )

    assert policy["schema"] == TF_GPU_MEMORY_POLICY_SCHEMA
    assert policy["all_physical_devices_memory_growth"] is True
    assert policy["full_device_preallocation_disabled"] is True
    assert policy["memory_growth_is_hard_memory_cap"] is False
    assert policy["tf_force_gpu_allow_growth"] == "true"
    assert tuple(row["device"] for row in policy["physical_devices"]) == (
        "/physical_device:GPU:0",
        "/physical_device:GPU:1",
    )
    assert policy["physical_devices"][0]["device_details"] == {
        "device_name": "Test /physical_device:GPU:0",
        "compute_capability": [8, 9],
    }


def test_memory_growth_fails_closed_after_runtime_initialization() -> None:
    tf = _TensorFlow((_Device("/physical_device:GPU:0"),), _ExperimentalConfig(initialized=True))

    with pytest.raises(TensorFlowGPUMemoryPolicyError, match="before logical device"):
        configure_tensorflow_gpu_memory_growth(tf)


def test_memory_growth_fails_closed_when_verification_is_false() -> None:
    tf = _TensorFlow((_Device("/physical_device:GPU:0"),), _ExperimentalConfig(ignore_set=True))

    with pytest.raises(TensorFlowGPUMemoryPolicyError, match="was not enabled"):
        configure_tensorflow_gpu_memory_growth(tf)


def test_memory_growth_can_describe_an_explicit_cpu_only_process() -> None:
    policy = configure_tensorflow_gpu_memory_growth(
        _TensorFlow((), _ExperimentalConfig()),
        require_gpu=False,
    )

    assert policy["physical_devices"] == ()
    assert policy["all_physical_devices_memory_growth"] is True


def test_fixed_memory_limit_is_installed_verified_and_described(monkeypatch) -> None:
    monkeypatch.delenv("TF_FORCE_GPU_ALLOW_GROWTH", raising=False)
    devices = (_Device("/physical_device:GPU:0"),)
    policy = configure_tensorflow_gpu_memory_limit(
        _TensorFlow(devices, _ExperimentalConfig()), memory_limit_mib=8192
    )

    assert policy["schema"] == TF_GPU_MEMORY_POLICY_SCHEMA
    assert policy["mode"] == "fixed_logical_device_limit"
    assert policy["memory_limit_mib_per_physical_device"] == 8192
    assert policy["aggregate_visible_gpu_limit_mib"] == 8192
    assert policy["hard_allocator_cap"] is True
    assert policy["memory_growth"] is False


def test_fixed_memory_limit_rejects_memory_growth(monkeypatch) -> None:
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    with pytest.raises(TensorFlowGPUMemoryPolicyError, match="cannot be combined"):
        configure_tensorflow_gpu_memory_limit(
            _TensorFlow((_Device("/physical_device:GPU:0"),), _ExperimentalConfig()),
            memory_limit_mib=8192,
        )


def test_fixed_memory_limit_fails_closed_after_runtime_initialization(
    monkeypatch,
) -> None:
    monkeypatch.delenv("TF_FORCE_GPU_ALLOW_GROWTH", raising=False)

    with pytest.raises(TensorFlowGPUMemoryPolicyError, match="before logical device"):
        configure_tensorflow_gpu_memory_limit(
            _TensorFlow(
                (_Device("/physical_device:GPU:0"),),
                _ExperimentalConfig(initialized=True),
            ),
            memory_limit_mib=8192,
        )
