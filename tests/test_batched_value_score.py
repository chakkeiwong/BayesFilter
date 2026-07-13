from __future__ import annotations

import ast
import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import (
    FixedTransportValueScoreAdapter,
    LatentAffineBatchValueScoreAdapter,
    LatentAffineHMCTransform,
    ValueScoreCapability,
    evaluate_batch_native_value_score,
    reviewed_value_score_target_fn,
)
from bayesfilter.inference import batched_value_score as batched_value_score_module


class BatchedQuadraticAdapter:
    parameter_dim = 3

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            runtime_backend="tensorflow_batched_quadratic_fixture",
            evidence_path="tests/test_batched_value_score.py",
            target_scope="batched_quadratic_fixture",
            nonclaims=("target value/score fixture only",),
        )

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values


class TelemetryBatchedQuadraticAdapter(BatchedQuadraticAdapter):
    def target_status_telemetry(self, theta: tf.Tensor) -> dict[str, tf.Tensor]:
        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        status_shape = tf.shape(values)[:-1]
        return {
            "status_code": tf.zeros(status_shape, dtype=tf.int32),
            "valid_pre_regularized_score": tf.ones(status_shape, dtype=tf.bool),
            "floor_count_value": tf.zeros(status_shape, dtype=tf.int32),
            "min_innovation_eigenvalue": tf.fill(status_shape, tf.constant(0.5, dtype=tf.float64)),
            "innovation_condition_estimate": tf.fill(status_shape, tf.constant(2.0, dtype=tf.float64)),
            "regularization_derivative_target": tf.zeros(status_shape, dtype=tf.int32),
        }


class ScalarOnlyQuadraticAdapter(BatchedQuadraticAdapter):
    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(values)), -values


class CountingBatchedQuadraticAdapter(BatchedQuadraticAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        self.calls += 1
        return super().log_prob_and_grad(theta)


class Float32OutputBatchedQuadraticAdapter(BatchedQuadraticAdapter):
    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.cast(tf.convert_to_tensor(theta), tf.float32)
        return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values


class AffineFixedTransport:
    parameter_dim = 3

    def __init__(self) -> None:
        self.shift = tf.constant([0.25, -0.5, 0.75], dtype=tf.float64)
        self.factor = tf.constant(
            [
                [2.0, 0.1, 0.0],
                [0.0, 1.5, -0.2],
                [0.3, 0.0, 1.25],
            ],
            dtype=tf.float64,
        )
        self.batch_calls = 0
        self.scalar_calls = 0

    def manifest_payload(self) -> dict[str, object]:
        return {
            "schema": "affine_fixed_transport_fixture.v1",
            "parameter_dim": self.parameter_dim,
            "shift": self.shift.numpy().tolist(),
            "factor": self.factor.numpy().tolist(),
        }

    def forward(self, z: tf.Tensor) -> tf.Tensor:
        self.scalar_calls += 1
        values = tf.convert_to_tensor(z, dtype=tf.float64)
        return self.shift + tf.linalg.matvec(self.factor, values)

    def forward_batch(self, z_batch: tf.Tensor) -> tf.Tensor:
        self.batch_calls += 1
        values = tf.convert_to_tensor(z_batch, dtype=tf.float64)
        return self.shift + tf.linalg.matmul(values, self.factor, transpose_b=True)

    def log_abs_det_jacobian(self, z: tf.Tensor) -> tf.Tensor:
        del z
        _sign, log_abs_det = tf.linalg.slogdet(self.factor)
        return log_abs_det

    def log_abs_det_jacobian_batch(self, z_batch: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(z_batch, dtype=tf.float64)
        _sign, log_abs_det = tf.linalg.slogdet(self.factor)
        return tf.fill(tf.shape(values)[:1], log_abs_det)

    def pullback_score(self, z: tf.Tensor, theta_score: tf.Tensor) -> tf.Tensor:
        del z
        return tf.linalg.matvec(
            self.factor,
            tf.convert_to_tensor(theta_score, dtype=tf.float64),
            transpose_a=True,
        )

    def pullback_score_batch(
        self,
        z_batch: tf.Tensor,
        theta_score_batch: tf.Tensor,
    ) -> tf.Tensor:
        del z_batch
        return tf.linalg.matmul(
            tf.convert_to_tensor(theta_score_batch, dtype=tf.float64),
            self.factor,
        )

    def log_abs_det_jacobian_score(self, z: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(z, dtype=tf.float64)
        return tf.zeros_like(values)

    def log_abs_det_jacobian_score_batch(self, z_batch: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(z_batch, dtype=tf.float64)
        return tf.zeros_like(values)


class NoManifestTransport(AffineFixedTransport):
    manifest_payload = None


class NoBatchTransport(AffineFixedTransport):
    forward_batch = None

    def log_abs_det_jacobian_batch(self, z_batch: tf.Tensor) -> tf.Tensor:
        return super().log_abs_det_jacobian_batch(z_batch)


class NoScalarLogdetTransport(AffineFixedTransport):
    log_abs_det_jacobian = None


class NoBatchLogdetTransport(AffineFixedTransport):
    log_abs_det_jacobian_batch = None


def _theta_batch() -> tf.Tensor:
    return tf.constant(
        [[0.1, -0.2, 0.3], [0.4, 0.0, -0.1], [-0.25, 0.15, 0.05]],
        dtype=tf.float64,
    )


def test_evaluate_batch_native_value_score_returns_shape_metadata_and_nonclaims() -> None:
    adapter = BatchedQuadraticAdapter()
    theta = _theta_batch()

    result = evaluate_batch_native_value_score(adapter, theta)

    expected_value = -0.5 * np.sum(np.square(theta.numpy()), axis=-1)
    np.testing.assert_allclose(result.value.numpy(), expected_value)
    np.testing.assert_allclose(result.score.numpy(), -theta.numpy())
    assert result.metadata.rank == "batch"
    assert result.metadata.batch_size == 3
    assert result.metadata.parameter_dim == 3
    assert result.metadata.value_score_authority == "graph_native"
    assert result.metadata.target_scope == "batched_quadratic_fixture"
    assert "no HMC tuning or sampling claim" in result.metadata.nonclaims
    assert result.diagnostics["path"] == "batch_native_adapter_log_prob_and_grad"
    assert result.diagnostics["accepted_target_xla_authority"] is True
    assert result.diagnostics["full_chain_xla_diagnostic_ready"] is False


def test_reviewed_value_score_target_custom_gradient_matches_adapter_score() -> None:
    theta = _theta_batch()
    target = reviewed_value_score_target_fn(BatchedQuadraticAdapter(), require_batched=True)

    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = target(theta)
    gradient = tape.gradient(value, theta)

    np.testing.assert_allclose(value.numpy(), -0.5 * np.sum(np.square(theta.numpy()), axis=-1))
    np.testing.assert_allclose(gradient.numpy(), -theta.numpy())


def test_reviewed_value_score_target_casts_adapter_outputs_to_hmc_dtype() -> None:
    theta = _theta_batch()
    target = reviewed_value_score_target_fn(
        Float32OutputBatchedQuadraticAdapter(),
        dtype=tf.float64,
        require_batched=True,
    )

    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = target(theta)
    gradient = tape.gradient(value, theta)

    assert value.dtype == tf.float64
    assert gradient.dtype == tf.float64
    expected_value = -0.5 * np.sum(np.square(theta.numpy().astype(np.float32)), axis=-1)
    np.testing.assert_allclose(value.numpy(), expected_value, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(gradient.numpy(), -theta.numpy(), rtol=1.0e-6, atol=1.0e-6)


def test_batch_native_rowwise_parity_against_scalar_fixture() -> None:
    adapter = BatchedQuadraticAdapter()
    theta = _theta_batch()
    batched = evaluate_batch_native_value_score(adapter, theta)

    scalar_values = []
    scalar_scores = []
    for row in tf.unstack(theta, axis=0):
        value, score = adapter.log_prob_and_grad(row)
        scalar_values.append(value)
        scalar_scores.append(score)

    np.testing.assert_allclose(batched.value.numpy(), tf.stack(scalar_values).numpy())
    np.testing.assert_allclose(batched.score.numpy(), tf.stack(scalar_scores).numpy())


def test_latent_affine_batch_value_score_adapter_calls_base_once_and_applies_chain_rule() -> None:
    base = CountingBatchedQuadraticAdapter()
    center = np.array([0.25, -0.5, 0.75])
    factor = np.array(
        [
            [2.0, 0.1, 0.0],
            [0.0, 1.5, -0.2],
            [0.3, 0.0, 1.25],
        ]
    )
    transform = LatentAffineHMCTransform(
        center=center,
        factor=factor,
        covariance_provenance="unit_test_covariance",
    )
    adapter = LatentAffineBatchValueScoreAdapter(
        base_adapter=base,
        transform=transform,
        target_scope="latent_affine_batch_fixture",
    )
    z = _theta_batch()

    value, score = adapter.log_prob_and_grad(z)
    theta = center + z.numpy() @ factor.T

    assert base.calls == 1
    np.testing.assert_allclose(value.numpy(), -0.5 * np.sum(theta * theta, axis=-1))
    np.testing.assert_allclose(score.numpy(), (-theta) @ factor)
    assert value.shape == (3,)
    assert score.shape == (3, 3)


def test_latent_affine_transform_helpers_preserve_sample_leading_axes() -> None:
    base = BatchedQuadraticAdapter()
    center = np.array([0.25, -0.5, 0.75])
    factor = np.array(
        [
            [2.0, 0.1, 0.0],
            [0.0, 1.5, -0.2],
            [0.3, 0.0, 1.25],
        ]
    )
    transform = LatentAffineHMCTransform(
        center=center,
        factor=factor,
        covariance_provenance="unit_test_covariance",
    )
    adapter = LatentAffineBatchValueScoreAdapter(
        base_adapter=base,
        transform=transform,
        target_scope="latent_affine_batch_fixture",
    )
    z = tf.reshape(tf.range(24, dtype=tf.float64), (2, 4, 3)) / 10.0
    theta_score = -adapter.latent_to_position(z)

    theta = adapter.latent_to_position(z)
    latent_score = adapter.theta_score_to_latent_score(theta_score)

    np.testing.assert_allclose(theta.numpy(), center + z.numpy() @ factor.T)
    np.testing.assert_allclose(latent_score.numpy(), theta_score.numpy() @ factor)
    assert theta.shape == (2, 4, 3)
    assert latent_score.shape == (2, 4, 3)


def test_latent_affine_target_status_telemetry_maps_to_position_coordinates() -> None:
    base = TelemetryBatchedQuadraticAdapter()
    center = np.array([0.25, -0.5, 0.75])
    factor = np.array(
        [
            [2.0, 0.1, 0.0],
            [0.0, 1.5, -0.2],
            [0.3, 0.0, 1.25],
        ]
    )
    transform = LatentAffineHMCTransform(
        center=center,
        factor=factor,
        covariance_provenance="unit_test_covariance",
    )
    adapter = LatentAffineBatchValueScoreAdapter(
        base_adapter=base,
        transform=transform,
        target_scope="latent_affine_batch_fixture",
    )

    telemetry = adapter.target_status_telemetry(_theta_batch())

    assert set(telemetry) == {
        "status_code",
        "valid_pre_regularized_score",
        "floor_count_value",
        "min_innovation_eigenvalue",
        "innovation_condition_estimate",
        "regularization_derivative_target",
    }
    assert telemetry["status_code"].shape == (3,)
    assert bool(tf.reduce_all(telemetry["valid_pre_regularized_score"]).numpy()) is True


def test_latent_affine_target_status_telemetry_fails_closed_without_base_method() -> None:
    transform = LatentAffineHMCTransform(
        center=np.zeros(3),
        factor=np.eye(3),
        covariance_provenance="unit_test_covariance",
    )
    adapter = LatentAffineBatchValueScoreAdapter(
        base_adapter=BatchedQuadraticAdapter(),
        transform=transform,
        target_scope="latent_affine_batch_fixture",
    )

    with pytest.raises(TypeError, match="target_status_telemetry"):
        adapter.target_status_telemetry(_theta_batch())


def test_fixed_transport_value_score_adapter_batch_chain_rule_and_signature() -> None:
    base = CountingBatchedQuadraticAdapter()
    transport = AffineFixedTransport()
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=transport,
        target_scope="fixed_transport_fixture",
        xla_hmc_ready=True,
    )
    z = _theta_batch()

    value, score = adapter.log_prob_and_grad_batch(z)
    u = transport.shift.numpy() + z.numpy() @ transport.factor.numpy().T
    logdet = np.linalg.slogdet(transport.factor.numpy())[1]

    assert base.calls == 1
    assert transport.batch_calls == 1
    assert transport.scalar_calls == 0
    np.testing.assert_allclose(value.numpy(), -0.5 * np.sum(u * u, axis=-1) + logdet)
    np.testing.assert_allclose(score.numpy(), (-u) @ transport.factor.numpy())
    assert value.shape == (3,)
    assert score.shape == (3, 3)
    payload = adapter.adapter_signature_payload()
    assert payload["schema"] == "bayesfilter.fixed_transport_value_score_adapter.v1"
    assert payload["fixed_transport_manifest_hash"] == adapter.transport_manifest_hash
    assert payload["target_scope"] == "fixed_transport_fixture"
    assert payload["value_score_authority"] == "graph_native"
    assert payload["batch_native"] is True
    assert len(adapter.adapter_signature()) == 64


def test_fixed_transport_value_score_adapter_uses_no_gradient_tape() -> None:
    source = inspect.getsource(FixedTransportValueScoreAdapter.log_prob_and_grad)
    assert "GradientTape" not in source
    assert "tape." not in source


def test_fixed_transport_value_score_adapter_scalar_chain_rule() -> None:
    base = CountingBatchedQuadraticAdapter()
    transport = AffineFixedTransport()
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=transport,
        target_scope="fixed_transport_fixture",
    )
    z = tf.constant([0.1, -0.2, 0.3], dtype=tf.float64)

    value, score = adapter.log_prob_and_grad(z)
    u = transport.shift.numpy() + transport.factor.numpy() @ z.numpy()
    logdet = np.linalg.slogdet(transport.factor.numpy())[1]

    assert base.calls == 1
    assert transport.scalar_calls == 1
    np.testing.assert_allclose(value.numpy(), -0.5 * np.sum(u * u) + logdet)
    np.testing.assert_allclose(score.numpy(), transport.factor.numpy().T @ (-u))
    assert value.shape == ()
    assert score.shape == (3,)


def test_fixed_transport_target_custom_gradient_matches_adapter_score() -> None:
    z = _theta_batch()
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=BatchedQuadraticAdapter(),
        transport=AffineFixedTransport(),
        target_scope="fixed_transport_fixture",
    )
    target = reviewed_value_score_target_fn(adapter, require_batched=True)

    with tf.GradientTape() as tape:
        tape.watch(z)
        value = target(z)
    gradient = tape.gradient(value, z)
    direct_value, direct_score = adapter.log_prob_and_grad_batch(z)

    np.testing.assert_allclose(value.numpy(), direct_value.numpy())
    np.testing.assert_allclose(gradient.numpy(), direct_score.numpy())


def test_fixed_transport_wrapper_preserves_authority_without_promoting_fallback() -> None:
    class FallbackBase(BatchedQuadraticAdapter):
        def value_score_capability(self) -> ValueScoreCapability:
            return ValueScoreCapability(
                value_score_authority="gradient_tape_fallback",
                xla_hmc_ready=False,
                runtime_backend="fallback_fixture",
                target_scope="fallback_base",
                nonclaims=("fallback fixture only",),
            )

    adapter = FixedTransportValueScoreAdapter(
        base_adapter=FallbackBase(),
        transport=AffineFixedTransport(),
        target_scope="fixed_transport_fixture",
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )

    capability = adapter.value_score_capability()

    assert capability.value_score_authority == "gradient_tape_fallback"
    assert capability.xla_hmc_ready is False
    assert capability.full_chain_xla_diagnostic_ready is False
    assert capability.target_scope == "fixed_transport_fixture"
    assert any(
        "fixed transport wrapper cannot promote fallback base authority" in claim
        for claim in capability.nonclaims
    )


def test_fixed_transport_adapter_fails_closed_for_missing_manifest_or_batch_methods() -> None:
    with pytest.raises(TypeError, match="manifest_payload"):
        FixedTransportValueScoreAdapter(
            base_adapter=BatchedQuadraticAdapter(),
            transport=NoManifestTransport(),
            target_scope="fixed_transport_fixture",
        )

    with pytest.raises(TypeError, match="forward_batch"):
        FixedTransportValueScoreAdapter(
            base_adapter=BatchedQuadraticAdapter(),
            transport=NoBatchTransport(),
            target_scope="fixed_transport_fixture",
            batch_native=True,
        )

    adapter = FixedTransportValueScoreAdapter(
        base_adapter=BatchedQuadraticAdapter(),
        transport=NoBatchTransport(),
        target_scope="fixed_transport_fixture",
        batch_native=False,
    )
    with pytest.raises(ValueError, match="batch-native transport"):
        adapter.log_prob_and_grad_batch(_theta_batch())


def test_fixed_transport_adapter_fails_closed_for_missing_logdet_methods() -> None:
    with pytest.raises(TypeError, match="log_abs_det_jacobian"):
        FixedTransportValueScoreAdapter(
            base_adapter=BatchedQuadraticAdapter(),
            transport=NoScalarLogdetTransport(),
            target_scope="fixed_transport_fixture",
        )

    with pytest.raises(TypeError, match="log_abs_det_jacobian_batch"):
        FixedTransportValueScoreAdapter(
            base_adapter=BatchedQuadraticAdapter(),
            transport=NoBatchLogdetTransport(),
            target_scope="fixed_transport_fixture",
            batch_native=True,
        )

    adapter = FixedTransportValueScoreAdapter(
        base_adapter=BatchedQuadraticAdapter(),
        transport=NoBatchLogdetTransport(),
        target_scope="fixed_transport_fixture",
        batch_native=False,
    )
    value, score = adapter.log_prob_and_grad(tf.zeros((3,), dtype=tf.float64))
    assert value.shape == ()
    assert score.shape == (3,)


def test_fixed_transport_adapter_fails_closed_for_missing_pullback_methods() -> None:
    class NoPullbackTransport(AffineFixedTransport):
        pullback_score = None

    class NoBatchPullbackTransport(AffineFixedTransport):
        pullback_score_batch = None

    with pytest.raises(TypeError, match="pullback_score"):
        FixedTransportValueScoreAdapter(
            base_adapter=BatchedQuadraticAdapter(),
            transport=NoPullbackTransport(),
            target_scope="fixed_transport_fixture",
        )

    with pytest.raises(TypeError, match="pullback_score_batch"):
        FixedTransportValueScoreAdapter(
            base_adapter=BatchedQuadraticAdapter(),
            transport=NoBatchPullbackTransport(),
            target_scope="fixed_transport_fixture",
            batch_native=True,
        )


def test_latent_affine_wrapper_preserves_graph_native_base_authority() -> None:
    base = BatchedQuadraticAdapter()
    transform = LatentAffineHMCTransform(
        center=np.zeros(3),
        factor=np.eye(3),
        covariance_provenance="unit_test_covariance",
    )
    adapter = LatentAffineBatchValueScoreAdapter(
        base_adapter=base,
        transform=transform,
        target_scope="latent_affine_batch_fixture",
        xla_hmc_ready=True,
    )

    capability = adapter.value_score_capability()

    assert capability.value_score_authority == "graph_native"
    assert capability.xla_hmc_ready is True
    assert any("base value/score authority: graph_native" in claim for claim in capability.nonclaims)


def test_latent_affine_wrapper_cannot_promote_gradient_tape_fallback_base() -> None:
    class FallbackBase(CountingBatchedQuadraticAdapter):
        def value_score_capability(self) -> ValueScoreCapability:
            return ValueScoreCapability(
                value_score_authority="gradient_tape_fallback",
                xla_hmc_ready=False,
                runtime_backend="fallback_fixture",
                target_scope="fallback_fixture",
                nonclaims=("fallback fixture only",),
            )

    transform = LatentAffineHMCTransform(
        center=np.zeros(3),
        factor=np.eye(3),
        covariance_provenance="unit_test_covariance",
    )
    adapter = LatentAffineBatchValueScoreAdapter(
        base_adapter=FallbackBase(),
        transform=transform,
        target_scope="latent_affine_batch_fixture",
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
    )

    capability = adapter.value_score_capability()

    assert capability.value_score_authority == "gradient_tape_fallback"
    assert capability.xla_hmc_ready is False
    assert capability.is_accepted_xla_hmc_authority is False
    assert capability.full_chain_xla_diagnostic_ready is False
    assert any(
        "latent wrapper cannot promote fallback base authority" in claim
        for claim in capability.nonclaims
    )


def test_batch_native_target_tf_function_reuses_concrete_function() -> None:
    theta = _theta_batch()
    target = reviewed_value_score_target_fn(BatchedQuadraticAdapter(), require_batched=True)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(3, 3), dtype=tf.float64)],
        reduce_retracing=True,
    )
    def compiled_target(x: tf.Tensor) -> tf.Tensor:
        return target(x)

    first = compiled_target(theta)
    second = compiled_target(tf.identity(theta))

    np.testing.assert_allclose(first.numpy(), second.numpy())
    assert len(compiled_target._list_all_concrete_functions_for_serialization()) == 1


def test_batch_native_target_cpu_xla_value_and_gradient_parity_is_scoped_fixture_only() -> None:
    theta = _theta_batch()
    target = reviewed_value_score_target_fn(BatchedQuadraticAdapter(), require_batched=True)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(3, 3), dtype=tf.float64)],
        jit_compile=True,
        reduce_retracing=True,
    )
    def compiled_value_and_gradient(x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(x)
            value = target(x)
        gradient = tape.gradient(value, x)
        if gradient is None:
            raise ValueError("compiled target gradient is unavailable")
        return value, gradient

    eager_value = target(theta)
    eager_gradient = tf.convert_to_tensor(-theta, dtype=tf.float64)
    compiled_value, compiled_gradient = compiled_value_and_gradient(theta)

    np.testing.assert_allclose(compiled_value.numpy(), eager_value.numpy())
    np.testing.assert_allclose(compiled_gradient.numpy(), eager_gradient.numpy())
    assert len(compiled_value_and_gradient._list_all_concrete_functions_for_serialization()) == 1


def test_batch_native_helpers_fail_closed_for_scalar_only_adapter() -> None:
    theta = _theta_batch()
    adapter = ScalarOnlyQuadraticAdapter()

    with pytest.raises(ValueError, match="batched target value must have rank 1"):
        evaluate_batch_native_value_score(adapter, theta)

    target = reviewed_value_score_target_fn(adapter, require_batched=True)
    with pytest.raises(ValueError, match="batched target value must have rank 1"):
        target(theta)


def test_batch_native_helpers_reject_scalar_input_when_batch_required() -> None:
    adapter = BatchedQuadraticAdapter()
    theta = tf.constant([0.1, -0.2, 0.3], dtype=tf.float64)

    with pytest.raises(ValueError, match="rank 2 theta"):
        evaluate_batch_native_value_score(adapter, theta)

    target = reviewed_value_score_target_fn(adapter, require_batched=True)
    with pytest.raises(ValueError, match="rank 2 theta"):
        target(theta)


def test_batch_native_helper_source_has_no_python_row_loop_or_stack() -> None:
    for fn in (
        batched_value_score_module.evaluate_batch_native_value_score,
        batched_value_score_module.reviewed_value_score_target_fn,
    ):
        source = inspect.getsource(fn)
        tree = ast.parse(source)
        assert not any(isinstance(node, (ast.For, ast.AsyncFor)) for node in ast.walk(tree))
        assert not any(isinstance(node, ast.ListComp) for node in ast.walk(tree))
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"append", "stack", "unstack", "map_fn", "vectorized_map"}
            for node in ast.walk(tree)
        )
        adapter_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "log_prob_and_grad"
        ]
        assert len(adapter_calls) == 1
