from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
from bayesfilter.inference.neutra_artifacts import (
    InvalidNeuTraArtifact,
    load_frozen_neutra_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "docs/benchmarks/run_ssl_lstm_neutra_phase5_exact_preflight_2026_07_16.py"
)


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    name = "ssl_lstm_neutra_phase5_exact_preflight_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class QuadraticTarget:
    parameter_dim = 4

    @staticmethod
    def log_prob_and_grad(values: Any) -> tuple[tf.Tensor, tf.Tensor]:
        tensor = tf.convert_to_tensor(values, tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(tensor), axis=-1), -tensor

    @staticmethod
    def adapter_signature() -> str:
        return "quadratic-phase5-fixture"

    @staticmethod
    def value_score_capability() -> dict[str, Any]:
        return {
            "value_score_authority": "graph_native",
            "xla_hmc_ready": False,
            "runtime_backend": "phase5_test_fixture",
            "target_scope": "phase5_test_fixture",
        }


class MutableTransport:
    def __init__(self) -> None:
        self.optimizer = object()
        self.weight = tf.Variable(1.0, dtype=tf.float64)


def real_transport(runner: ModuleType, label: str = "fresh-g") -> Any:
    path, _expected_hash = runner.PAYLOADS[label]
    payload = json.loads((runner.ROOT / path).read_text(encoding="utf-8"))
    return load_frozen_neutra_artifact(
        payload,
        expected_target_signature=runner.TARGET_SIGNATURE,
    ).transport


def test_plan_and_runner_preserve_no_hmc_boundary(runner: ModuleType) -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8").lower()
    assert "tensorflow_probability" not in source
    assert "hamiltonianmontecarlo" not in source
    assert "sample_chain" not in source
    assert runner.PAYLOADS["fresh-g"][1] == (
        "6e147d5b33d003e0c895f294fc6b33523dcf97dc24af794d26a677886dedc354"
    )
    assert runner.PAYLOADS["fresh-h"][1] == (
        "ed0e42602aa39788ca1ea8d3c881d8bf85e15b91a687ef9adbe00a7b2c9120fb"
    )
    assert runner.ORIGINAL_STARTS == (
        (0.0, 0.0, 0.0, 0.0),
        (0.5, -0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5, 0.5),
        (0.5, 0.5, -0.5, -0.5),
    )


def test_probe_bank_is_prospectively_fixed(runner: ModuleType) -> None:
    points, labels, metadata = runner.probe_bank()
    assert tuple(points.shape) == (21, 4)
    assert labels[0] == "prior_center"
    assert labels[-4:] == [f"original_start_{index}" for index in range(4)]
    assert metadata["shell_radii"] == [2.0, 4.0]
    assert metadata["point_count"] == 21
    assert runner.PARITY_ATOL == 1.0e-10


def test_fixed_binding_uses_declared_forward_direction(runner: ModuleType) -> None:
    transport = real_transport(runner)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=QuadraticTarget(),
        transport=transport,
        target_scope="phase5_direction_fixture",
    )
    z = tf.constant(
        [[0.2, -0.4, 0.6, -0.8], [-0.1, 0.3, -0.5, 0.7]],
        tf.float64,
    )
    declared = transport.forward_z_to_theta_batch(z)
    tf.debugging.assert_near(
        adapter.latent_to_position(z), declared, atol=1.0e-15, rtol=0.0
    )
    wrong_direction = transport.inverse_theta_to_z_batch(z)
    assert bool(tf.reduce_any(tf.abs(declared - wrong_direction) > 1.0e-8).numpy())


def test_wrong_sign_and_omitted_logdet_score_fail_finite_differences(
    runner: ModuleType,
) -> None:
    transport = real_transport(runner)
    z = tf.constant(
        [[0.2, -0.4, 0.6, -0.8], [-0.1, 0.3, -0.5, 0.7]],
        tf.float64,
    )
    target = QuadraticTarget()
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=target,
        transport=transport,
        target_scope="phase5_sign_fixture",
    )
    value, correct_score = adapter.log_prob_and_grad_batch(z)
    theta = transport.forward_z_to_theta_batch(z)
    target_value, target_score = target.log_prob_and_grad(theta)
    logdet = transport.log_abs_det_jacobian_batch(z)
    wrong_sign_value = target_value - logdet
    assert float(tf.reduce_max(tf.abs(value - wrong_sign_value)).numpy()) > 1.0e-8

    finite_difference = []
    for coordinate in range(4):
        direction = tf.one_hot(coordinate, 4, dtype=tf.float64)[tf.newaxis, :]
        plus = adapter.log_prob_and_grad_batch(z + runner.FD_STEP * direction)[0]
        minus = adapter.log_prob_and_grad_batch(z - runner.FD_STEP * direction)[0]
        finite_difference.append((plus - minus) / (2.0 * runner.FD_STEP))
    fd = tf.stack(finite_difference, axis=1)
    omitted = transport.pullback_score_batch(z, target_score)
    wrong_score = omitted - transport.log_abs_det_jacobian_score_batch(z)
    tolerance = runner.SCORE_ATOL + runner.SCORE_RTOL * tf.abs(fd)
    assert bool(tf.reduce_all(tf.abs(correct_score - fd) <= tolerance).numpy())
    assert not bool(tf.reduce_all(tf.abs(omitted - fd) <= tolerance).numpy())
    assert not bool(tf.reduce_all(tf.abs(wrong_score - fd) <= tolerance).numpy())


def test_loader_rejects_corrupted_tensor_and_target_mismatch(
    runner: ModuleType,
) -> None:
    path, _expected_hash = runner.PAYLOADS["fresh-g"]
    payload = json.loads((runner.ROOT / path).read_text(encoding="utf-8"))
    corrupted = copy.deepcopy(payload)
    dense = next(
        component
        for component in corrupted["components"]
        if component["kind"] == "dense_autoregressive_iaf"
    )
    dense["weights"][0][0][0] += 1.0e-6
    with pytest.raises(InvalidNeuTraArtifact, match="tensor_hash mismatch"):
        load_frozen_neutra_artifact(
            corrupted,
            expected_target_signature=runner.TARGET_SIGNATURE,
        )
    with pytest.raises(InvalidNeuTraArtifact, match="target_signature mismatch"):
        load_frozen_neutra_artifact(
            payload,
            expected_target_signature="0" * 64,
        )


def test_mutable_state_reachability_fails_closed(runner: ModuleType) -> None:
    with pytest.raises(runner.PreflightError, match="mutable training state reachable"):
        runner.assert_no_mutable_training_state(MutableTransport())
    report = runner.assert_no_mutable_training_state(real_transport(runner))
    assert report == {
        "mutable_tf_variables": [],
        "optimizer_trainer_surfaces": [],
        "passed": True,
    }


def test_target_bridge_dispatches_scalar_and_batch_without_row_loop() -> None:
    class Target:
        parameter_dim = 4
        parameter_names = (
            "latent_mean_weight.0.0",
            "latent_mean_bias.0",
            "observation_weight.0.0",
            "observation_bias.0",
        )
        scalar_calls = 0
        batch_calls = 0
        target_scope = "phase5_bridge_fixture"

        def value_and_score(self, values: Any) -> tuple[tf.Tensor, tf.Tensor]:
            self.scalar_calls += 1
            return QuadraticTarget.log_prob_and_grad(values)

        def batch_value_and_score(self, values: Any) -> tuple[tf.Tensor, tf.Tensor]:
            self.batch_calls += 1
            return QuadraticTarget.log_prob_and_grad(values)

        @staticmethod
        def adapter_signature() -> str:
            return "phase5-bridge-fixture"

        @staticmethod
        def target_signature() -> str:
            return "1" * 64

        @staticmethod
        def value_score_capability() -> dict[str, Any]:
            return QuadraticTarget.value_score_capability()

    target = Target()
    runner_name = "ssl_lstm_neutra_phase5_bridge_runner"
    spec = importlib.util.spec_from_file_location(runner_name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[runner_name] = module
    spec.loader.exec_module(module)
    bridge = module.TargetBatchBridge(target)
    bridge.log_prob_and_grad(tf.zeros((4,), tf.float64))
    bridge.log_prob_and_grad(tf.zeros((3, 4), tf.float64))
    assert target.scalar_calls == 1
    assert target.batch_calls == 1
