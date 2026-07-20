from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter
from experiments.dpf_implementation.tf_tfp.filters import experimental_batched_ledh_pfpf_ot_tf
from experiments.dpf_implementation.tf_tfp.filters.experimental_batched_ledh_pfpf_ot_tf import (
    batched_annealed_transport_core_tf,
    batched_ledh_pfpf_ot_value_and_score_tf,
    batched_ledh_pfpf_ot_value_core_tf,
    batched_ledh_flow_core_tf,
)
from experiments.dpf_implementation.tf_tfp.filters.experimental_batched_ledh_pfpf_ot_tf import (
    BatchedLEDHFlowTensors,
    BatchedLEDHPFPFOTCallbacks,
    BatchedLEDHPFPFOTFixedInputs,
    BatchedLEDHPFPFOTValueTensors,
    MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
    MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
    MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
    SCALAR_PARITY_ATOL,
    SCALAR_PARITY_RTOL,
    TRANSPORT_PARITY_ATOL,
    TRANSPORT_PARITY_RTOL,
    validate_batched_score_tensor,
    validate_batched_value_tensor,
    validate_flow_tensors_against_contract,
)
from experiments.dpf_implementation.tf_tfp.flows.ledh_tf import ledh_flow_batch_tf
from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    annealed_transport_resample_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf


DTYPE = tf.float64
experimental_batched_ledh_pfpf_ot_tf.DTYPE = DTYPE
annealed_transport_tf.DTYPE = DTYPE
MODULE_PATH = Path(
    "experiments/dpf_implementation/tf_tfp/filters/experimental_batched_ledh_pfpf_ot_tf.py"
)


def _fixed_inputs(batch_size: int = 2) -> BatchedLEDHPFPFOTFixedInputs:
    time_steps = 3
    num_particles = 4
    state_dim = 2
    observation_dim = 1
    parameter_dim = 3
    theta = tf.reshape(
        tf.cast(tf.range(batch_size * parameter_dim), DTYPE),
        [batch_size, parameter_dim],
    )
    observations = tf.reshape(
        tf.linspace(tf.constant(-0.2, DTYPE), tf.constant(0.2, DTYPE), time_steps),
        [time_steps, observation_dim],
    )
    initial_particles = tf.reshape(
        tf.cast(tf.range(batch_size * num_particles * state_dim), DTYPE),
        [batch_size, num_particles, state_dim],
    ) / tf.constant(10.0, DTYPE)
    pre_flow_particles = tf.reshape(
        tf.cast(tf.range(batch_size * time_steps * num_particles * state_dim), DTYPE),
        [batch_size, time_steps, num_particles, state_dim],
    ) / tf.constant(20.0, DTYPE)
    fixed_mask = tf.constant(
        [[False, True, False], [True, False, True]][:batch_size],
        dtype=tf.bool,
    )
    return BatchedLEDHPFPFOTFixedInputs(
        theta_batch=theta,
        observations=observations,
        initial_particles=initial_particles,
        pre_flow_particles=pre_flow_particles,
        fixed_resampling_mask=fixed_mask,
    )


def _good_flow(*_args, **_kwargs) -> BatchedLEDHFlowTensors:
    return BatchedLEDHFlowTensors(
        post_flow_particles=tf.zeros([2, 4, 2], dtype=DTYPE),
        pre_flow_log_density=tf.zeros([2, 4], dtype=DTYPE),
        forward_log_det=tf.zeros([2, 4], dtype=DTYPE),
        local_posterior_means=tf.zeros([2, 4, 2], dtype=DTYPE),
        local_posterior_covariances=tf.zeros([2, 4, 2, 2], dtype=DTYPE),
    )


def _good_density(*_args, **_kwargs) -> tf.Tensor:
    return tf.zeros([2, 4], dtype=DTYPE)


def _bad_random_flow(*_args, **_kwargs) -> BatchedLEDHFlowTensors:
    noise = tf.random.normal([2, 4, 2], dtype=DTYPE)
    return BatchedLEDHFlowTensors(
        post_flow_particles=noise,
        pre_flow_log_density=tf.zeros([2, 4], dtype=DTYPE),
        forward_log_det=tf.zeros([2, 4], dtype=DTYPE),
        local_posterior_means=tf.zeros([2, 4, 2], dtype=DTYPE),
        local_posterior_covariances=tf.zeros([2, 4, 2, 2], dtype=DTYPE),
    )


def test_fixed_inputs_lock_shape_contract_and_uniform_weights() -> None:
    fixed = _fixed_inputs()
    contract = fixed.shape_contract

    assert contract.batch_size == 2
    assert contract.time_steps == 3
    assert contract.num_particles == 4
    assert contract.state_dim == 2
    assert contract.observation_dim == 1
    assert contract.parameter_dim == 3
    assert contract.value_shape == (2,)
    assert contract.score_shape == (2, 3)
    assert contract.fixed_pre_flow_shape == (2, 3, 4, 2)
    assert contract.fixed_mask_shape == (2, 3)
    assert contract.scalar_parity_atol == SCALAR_PARITY_ATOL == 1.0e-10
    assert contract.scalar_parity_rtol == SCALAR_PARITY_RTOL == 1.0e-10
    assert contract.transport_parity_atol == TRANSPORT_PARITY_ATOL == 1.0e-8
    assert contract.transport_parity_rtol == TRANSPORT_PARITY_RTOL == 1.0e-8
    tf.debugging.assert_near(
        tf.reduce_logsumexp(fixed.initial_log_weights, axis=1),
        tf.zeros([2], dtype=DTYPE),
        atol=1.0e-12,
    )
    assert "no categorical particle-filter gradient claim" in contract.nonclaims
    assert "production/default target by owner directive" in contract.nonclaims
    assert "no HMC readiness claim" in contract.nonclaims


def test_precision_policy_records_gpu_tf32_production_default() -> None:
    metadata = experimental_batched_ledh_pfpf_ot_tf.precision_policy_metadata()

    assert metadata["precision_default_policy"] == "production_ledh_pfpf_ot_gpu_tf32"
    assert metadata["default_execution_target"] == "gpu"
    assert metadata["default_algorithm_target"] == "ledh_pfpf_ot_tf32"
    assert metadata["default_target_status"] == "production_default_by_owner_directive"
    assert metadata["default_route_acceptance"] == "accepted_default_use_whenever_possible"
    assert "use for BayesFilter DPF LEDH-PFPF-OT work whenever" in metadata[
        "default_route_guidance"
    ]
    assert "avoids dense transport/history storage" in metadata["default_route_rationale"]
    assert metadata["default_dtype"] == "float32"
    assert metadata["default_tf32_mode"] == "enabled"
    assert metadata["fp64_reference_requires_explicit_dtype"] is True
    assert metadata["public_api_exposure"] == "separately_gated"
    assert "production" in metadata["scope"]


def test_fixed_inputs_require_static_compatible_shapes_and_masks() -> None:
    fixed = _fixed_inputs()
    with pytest.raises(ValueError, match="fixed_resampling_mask time steps"):
        BatchedLEDHPFPFOTFixedInputs(
            theta_batch=fixed.theta_batch,
            observations=fixed.observations,
            initial_particles=fixed.initial_particles,
            pre_flow_particles=fixed.pre_flow_particles,
            fixed_resampling_mask=tf.ones([2, 2], dtype=tf.bool),
        )

    with pytest.raises(ValueError, match="pre_flow_particles state dim"):
        BatchedLEDHPFPFOTFixedInputs(
            theta_batch=fixed.theta_batch,
            observations=fixed.observations,
            initial_particles=fixed.initial_particles,
            pre_flow_particles=tf.zeros([2, 3, 4, 3], dtype=DTYPE),
            fixed_resampling_mask=fixed.fixed_resampling_mask,
        )


def test_value_score_and_flow_shape_validation() -> None:
    fixed = _fixed_inputs()
    contract = fixed.shape_contract
    value = validate_batched_value_tensor(tf.zeros([2], dtype=DTYPE), contract)
    score = validate_batched_score_tensor(tf.zeros([2, 3], dtype=DTYPE), contract)
    flow = validate_flow_tensors_against_contract(_good_flow(), contract)

    assert value.shape == (2,)
    assert score.shape == (2, 3)
    assert flow.post_flow_particles.shape == (2, 4, 2)

    with pytest.raises(ValueError, match="score parameter dim"):
        validate_batched_score_tensor(tf.zeros([2, 2], dtype=DTYPE), contract)

    with pytest.raises(ValueError, match="flow particle count"):
        validate_flow_tensors_against_contract(
            BatchedLEDHFlowTensors(
                post_flow_particles=tf.zeros([2, 5, 2], dtype=DTYPE),
                pre_flow_log_density=tf.zeros([2, 5], dtype=DTYPE),
                forward_log_det=tf.zeros([2, 5], dtype=DTYPE),
                local_posterior_means=tf.zeros([2, 5, 2], dtype=DTYPE),
                local_posterior_covariances=tf.zeros([2, 5, 2, 2], dtype=DTYPE),
            ),
            contract,
        )


def test_callbacks_reject_hidden_rng_sources() -> None:
    callbacks = BatchedLEDHPFPFOTCallbacks(
        ledh_flow=_good_flow,
        transition_log_density=_good_density,
        observation_log_density=_good_density,
    )
    assert callbacks.ledh_flow is _good_flow

    with pytest.raises(ValueError, match="forbidden RNG token"):
        BatchedLEDHPFPFOTCallbacks(
            ledh_flow=_bad_random_flow,
            transition_log_density=_good_density,
            observation_log_density=_good_density,
        )


def test_experimental_ledh_pfpf_ot_contract_is_not_reexported_publicly() -> None:
    assert "experimental_batched_ledh_pfpf_ot_tf" not in bayesfilter.__all__
    assert "experimental_batched_ledh_pfpf_ot_tf" not in bayesfilter._EXPORT_MODULES


def test_phase1_module_source_does_not_call_random_generators() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    forbidden_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"normal", "uniform"}:
            owner = func.value
            if isinstance(owner, ast.Attribute) and owner.attr == "random":
                forbidden_calls.append((func.attr, node.lineno))
            if isinstance(owner, ast.Name) and owner.id == "random":
                forbidden_calls.append((func.attr, node.lineno))
    assert forbidden_calls == []


def _flow_fixture() -> dict[str, tf.Tensor]:
    return {
        "pre_flow_particles": tf.constant(
            [[[-0.20], [0.05], [0.30]], [[0.10], [0.35], [0.55]]],
            dtype=DTYPE,
        ),
        "ancestors": tf.constant(
            [[[-0.25], [0.00], [0.20]], [[0.05], [0.30], [0.50]]],
            dtype=DTYPE,
        ),
        "observation": tf.constant([[0.08], [0.42]], dtype=DTYPE),
        "transition_matrix": tf.constant([[[0.90]], [[0.75]]], dtype=DTYPE),
        "transition_covariance": tf.constant([[[0.20]], [[0.35]]], dtype=DTYPE),
        "observation_covariance": tf.constant([[[0.30]], [[0.45]]], dtype=DTYPE),
    }


def _batched_observation(points: tf.Tensor) -> tf.Tensor:
    return points


def _batched_observation_jacobian(points: tf.Tensor) -> tf.Tensor:
    batch_size = points.shape[0]
    num_particles = points.shape[1]
    if batch_size is None or num_particles is None:
        raise ValueError("test fixture requires static batch and particle dimensions")
    return tf.ones([batch_size, num_particles, 1, 1], dtype=DTYPE)


def _batched_observation_residual(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
    return observation[:, None, :] - h_ref


def _scalar_observation(point: tf.Tensor) -> tf.Tensor:
    return tf.reshape(point, [1])


def _scalar_observation_jacobian(_point: tf.Tensor) -> tf.Tensor:
    return tf.constant([[1.0]], dtype=DTYPE)


def _scalar_observation_residual(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
    return tf.reshape(observation, [1]) - tf.reshape(h_ref, [1])


def test_batched_ledh_flow_core_matches_scalar_rows() -> None:
    fixture = _flow_fixture()
    batched = batched_ledh_flow_core_tf(
        **fixture,
        observation_fn=_batched_observation,
        observation_jacobian_fn=_batched_observation_jacobian,
        observation_residual_fn=_batched_observation_residual,
    )

    scalar_results = []
    for row in range(fixture["pre_flow_particles"].shape[0]):
        scalar_results.append(
            ledh_flow_batch_tf(
                pre_flow_particles=fixture["pre_flow_particles"][row],
                ancestors=fixture["ancestors"][row],
                observation=fixture["observation"][row],
                transition_matrix=fixture["transition_matrix"][row],
                transition_covariance=fixture["transition_covariance"][row],
                observation_covariance=fixture["observation_covariance"][row],
                observation_fn=_scalar_observation,
                observation_jacobian_fn=_scalar_observation_jacobian,
                observation_residual_fn=_scalar_observation_residual,
            )
        )

    np.testing.assert_allclose(
        batched.post_flow_particles.numpy(),
        tf.stack([result.post_flow_particles for result in scalar_results], axis=0).numpy(),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batched.pre_flow_log_density.numpy(),
        tf.stack([result.pre_flow_log_density for result in scalar_results], axis=0).numpy(),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batched.forward_log_det.numpy(),
        tf.stack([result.forward_log_det for result in scalar_results], axis=0).numpy(),
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        batched.local_posterior_covariances.numpy(),
        tf.stack(
            [result.local_posterior_covariances for result in scalar_results],
            axis=0,
        ).numpy(),
        atol=1.0e-10,
    )


def test_batched_ledh_flow_core_handles_ill_conditioned_posterior_precision() -> None:
    dtype = DTYPE
    fixture = {
        "pre_flow_particles": tf.constant(
            [[[1.0e-4, 2.0], [2.0e-4, -1.0]]],
            dtype=dtype,
        ),
        "ancestors": tf.constant(
            [[[1.0e-4, 2.0], [2.0e-4, -1.0]]],
            dtype=dtype,
        ),
        "observation": tf.constant([[0.0]], dtype=dtype),
        "transition_matrix": tf.eye(2, batch_shape=[1], dtype=dtype),
        "transition_covariance": tf.constant(
            [[[1.0e12, 0.0], [0.0, 1.0]]],
            dtype=dtype,
        ),
        "observation_covariance": tf.constant([[[1.0]]], dtype=dtype),
    }

    def observation_fn(points: tf.Tensor) -> tf.Tensor:
        return points[..., :1]

    def observation_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
        del points
        return tf.constant([[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=dtype)

    def observation_residual_fn(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        return observation[:, None, :] - h_ref

    result = batched_ledh_flow_core_tf(
        **fixture,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_residual_fn=observation_residual_fn,
    )

    assert bool(tf.reduce_all(tf.math.is_finite(result.post_flow_particles)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(result.local_posterior_covariances)).numpy())
    covariance = result.local_posterior_covariances.numpy()
    np.testing.assert_allclose(
        covariance,
        np.swapaxes(covariance, -1, -2),
        atol=1.0e-10,
    )
    assert np.min(np.linalg.eigvalsh(covariance)) > 0.0


def test_batched_annealed_transport_core_uses_fixed_mask_without_ess_branch() -> None:
    particles = tf.constant(
        [
            [[-0.50], [-0.10], [0.25], [0.60]],
            [[0.10], [0.20], [0.30], [0.40]],
        ],
        dtype=DTYPE,
    )
    log_weights = tf.math.log(
        tf.constant(
            [[0.10, 0.20, 0.30, 0.40], [0.25, 0.25, 0.25, 0.25]],
            dtype=DTYPE,
        )
    )
    mask = tf.constant([True, False], dtype=tf.bool)

    result = batched_annealed_transport_core_tf(
        particles,
        log_weights,
        mask,
        max_iterations=12,
    )
    public_result = annealed_transport_resample_tf(
        particles,
        log_weights,
        ess_mask=mask,
        application_mode="filterflow_all_rows",
        max_iterations=12,
    )

    np.testing.assert_allclose(
        result.particles.numpy()[0],
        public_result.particles.numpy()[0],
        atol=TRANSPORT_PARITY_ATOL,
        rtol=TRANSPORT_PARITY_RTOL,
    )
    np.testing.assert_allclose(
        result.transport_matrix.numpy()[0],
        public_result.transport_matrix.numpy()[0],
        atol=TRANSPORT_PARITY_ATOL,
        rtol=TRANSPORT_PARITY_RTOL,
    )
    np.testing.assert_allclose(result.particles.numpy()[1], particles.numpy()[1], atol=0.0)
    np.testing.assert_allclose(result.log_weights.numpy()[1], log_weights.numpy()[1], atol=0.0)
    np.testing.assert_allclose(
        result.transport_matrix.numpy()[1],
        np.eye(4),
        atol=0.0,
    )


def test_batched_annealed_transport_streaming_matches_dense_without_matrix() -> None:
    particles = tf.constant(
        [
            [[-0.50], [-0.10], [0.25], [0.60]],
            [[0.10], [0.20], [0.30], [0.40]],
        ],
        dtype=DTYPE,
    )
    log_weights = tf.math.log(
        tf.constant(
            [[0.10, 0.20, 0.30, 0.40], [0.40, 0.30, 0.20, 0.10]],
            dtype=DTYPE,
        )
    )
    mask = tf.constant([True, True], dtype=tf.bool)

    dense = batched_annealed_transport_core_tf(
        particles,
        log_weights,
        mask,
        max_iterations=8,
        transport_gradient_mode="raw",
        transport_plan_mode="dense",
    )
    streaming = batched_annealed_transport_core_tf(
        particles,
        log_weights,
        mask,
        max_iterations=8,
        transport_gradient_mode="raw",
        transport_plan_mode="streaming",
        row_chunk_size=4,
        col_chunk_size=4,
    )

    assert streaming.transport_matrix.shape == (2, 0, 0)
    np.testing.assert_allclose(
        streaming.particles.numpy(),
        dense.particles.numpy(),
        atol=TRANSPORT_PARITY_ATOL,
        rtol=TRANSPORT_PARITY_RTOL,
    )
    np.testing.assert_allclose(
        streaming.log_weights.numpy(),
        dense.log_weights.numpy(),
        atol=0.0,
    )
    np.testing.assert_allclose(
        streaming.max_row_residual.numpy(),
        dense.max_row_residual.numpy(),
        atol=TRANSPORT_PARITY_ATOL,
        rtol=TRANSPORT_PARITY_RTOL,
    )


def test_phase2_core_row_independence() -> None:
    fixture = _flow_fixture()
    base_flow = batched_ledh_flow_core_tf(
        **fixture,
        observation_fn=_batched_observation,
        observation_jacobian_fn=_batched_observation_jacobian,
        observation_residual_fn=_batched_observation_residual,
    )
    perturbed_fixture = dict(fixture)
    perturbed_fixture["pre_flow_particles"] = tf.tensor_scatter_nd_add(
        fixture["pre_flow_particles"],
        indices=tf.constant([[1, 0, 0]], dtype=tf.int32),
        updates=tf.constant([0.7], dtype=DTYPE),
    )
    perturbed_flow = batched_ledh_flow_core_tf(
        **perturbed_fixture,
        observation_fn=_batched_observation,
        observation_jacobian_fn=_batched_observation_jacobian,
        observation_residual_fn=_batched_observation_residual,
    )
    np.testing.assert_allclose(
        perturbed_flow.post_flow_particles.numpy()[0],
        base_flow.post_flow_particles.numpy()[0],
        atol=0.0,
    )

    particles = tf.constant(
        [
            [[-0.50], [-0.10], [0.25], [0.60]],
            [[0.10], [0.20], [0.30], [0.40]],
        ],
        dtype=DTYPE,
    )
    log_weights = tf.math.log(
        tf.constant(
            [[0.10, 0.20, 0.30, 0.40], [0.40, 0.30, 0.20, 0.10]],
            dtype=DTYPE,
        )
    )
    mask = tf.constant([True, True], dtype=tf.bool)
    base_transport = batched_annealed_transport_core_tf(
        particles,
        log_weights,
        mask,
        max_iterations=12,
    )
    perturbed_transport = batched_annealed_transport_core_tf(
        tf.tensor_scatter_nd_add(
            particles,
            indices=tf.constant([[1, 0, 0]], dtype=tf.int32),
            updates=tf.constant([1.5], dtype=DTYPE),
        ),
        log_weights,
        mask,
        max_iterations=12,
    )
    np.testing.assert_allclose(
        perturbed_transport.particles.numpy()[0],
        base_transport.particles.numpy()[0],
        atol=TRANSPORT_PARITY_ATOL,
        rtol=TRANSPORT_PARITY_RTOL,
    )


def test_phase2_core_tf_function_smoke() -> None:
    fixture = _flow_fixture()

    @tf.function(reduce_retracing=True)
    def graph_flow(
        pre_flow_particles: tf.Tensor,
        ancestors: tf.Tensor,
        observation: tf.Tensor,
        transition_matrix: tf.Tensor,
        transition_covariance: tf.Tensor,
        observation_covariance: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        result = batched_ledh_flow_core_tf(
            pre_flow_particles=pre_flow_particles,
            ancestors=ancestors,
            observation=observation,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            observation_fn=_batched_observation,
            observation_jacobian_fn=_batched_observation_jacobian,
            observation_residual_fn=_batched_observation_residual,
        )
        return result.post_flow_particles, result.forward_log_det

    @tf.function(reduce_retracing=True)
    def graph_transport(
        particles: tf.Tensor,
        log_weights: tf.Tensor,
        mask: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        result = batched_annealed_transport_core_tf(
            particles,
            log_weights,
            mask,
            max_iterations=8,
        )
        return result.particles, result.log_weights

    flow_particles, flow_logdet = graph_flow(**fixture)
    transported, transported_logw = graph_transport(
        tf.constant(
            [[[-0.50], [-0.10], [0.25], [0.60]], [[0.10], [0.20], [0.30], [0.40]]],
            dtype=DTYPE,
        ),
        tf.math.log(
            tf.constant(
                [[0.10, 0.20, 0.30, 0.40], [0.25, 0.25, 0.25, 0.25]],
                dtype=DTYPE,
            )
        ),
        tf.constant([True, False], dtype=tf.bool),
    )

    assert flow_particles.shape == (2, 3, 1)
    assert flow_logdet.shape == (2, 3)
    assert transported.shape == (2, 4, 1)
    assert transported_logw.shape == (2, 4)
    assert bool(tf.reduce_all(tf.math.is_finite(flow_particles)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(transported)).numpy())
    assert len(graph_flow._list_all_concrete_functions_for_serialization()) == 1
    assert len(graph_transport._list_all_concrete_functions_for_serialization()) == 1


def test_phase2_streaming_transport_tf_function_smoke() -> None:
    @tf.function(reduce_retracing=True)
    def graph_streaming_transport(
        particles: tf.Tensor,
        log_weights: tf.Tensor,
        mask: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        result = batched_annealed_transport_core_tf(
            particles,
            log_weights,
            mask,
            max_iterations=8,
            transport_gradient_mode="raw",
            transport_plan_mode="streaming",
            row_chunk_size=4,
            col_chunk_size=4,
        )
        return result.particles, result.log_weights, result.transport_matrix

    transported, transported_logw, transport_matrix = graph_streaming_transport(
        tf.constant(
            [[[-0.50], [-0.10], [0.25], [0.60]], [[0.10], [0.20], [0.30], [0.40]]],
            dtype=DTYPE,
        ),
        tf.math.log(
            tf.constant(
                [[0.10, 0.20, 0.30, 0.40], [0.25, 0.25, 0.25, 0.25]],
                dtype=DTYPE,
            )
        ),
        tf.constant([True, False], dtype=tf.bool),
    )

    assert transported.shape == (2, 4, 1)
    assert transported_logw.shape == (2, 4)
    assert transport_matrix.shape == (2, 0, 0)
    assert bool(tf.reduce_all(tf.math.is_finite(transported)).numpy())
    assert len(graph_streaming_transport._list_all_concrete_functions_for_serialization()) == 1


def test_phase2_core_sources_do_not_use_numpy_or_python_rng() -> None:
    for function in (batched_ledh_flow_core_tf, batched_annealed_transport_core_tf):
        source = inspect.getsource(function)
        assert ".numpy(" not in source
        assert "tf.random" not in source
        assert "np.random" not in source
        assert "random." not in source


def _value_fixture(batch_size: int) -> dict[str, tf.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    time_steps = 3
    num_particles = 4
    row = tf.cast(tf.range(batch_size), DTYPE)
    base_particles = tf.reshape(
        tf.constant([-0.30, -0.05, 0.20, 0.45], dtype=DTYPE),
        [1, num_particles, 1],
    )
    initial_particles = base_particles + 0.02 * row[:, None, None]
    offsets = tf.reshape(
        tf.constant([-0.04, -0.01, 0.02, 0.05], dtype=DTYPE),
        [1, 1, num_particles, 1],
    )
    observations = tf.reshape(
        tf.constant([0.03, 0.08, 0.12], dtype=DTYPE),
        [time_steps, 1],
    )
    transition_matrix = tf.reshape(0.82 + 0.01 * row, [batch_size, 1, 1])
    pre_flow_particles = (
        transition_matrix[:, None, :, :]
        * initial_particles[:, None, :, :]
        + offsets
        + 0.015 * tf.reshape(tf.cast(tf.range(time_steps), DTYPE), [1, time_steps, 1, 1])
    )
    return {
        "observations": observations,
        "initial_particles": initial_particles,
        "pre_flow_particles": pre_flow_particles,
        "fixed_resampling_mask": tf.tile(
            tf.constant([[False, True, False]], dtype=tf.bool),
            [batch_size, 1],
        ),
        "transition_matrix": transition_matrix,
        "transition_covariance": tf.tile(tf.constant([[[0.16]]], dtype=DTYPE), [batch_size, 1, 1]),
        "observation_covariance": tf.tile(tf.constant([[[0.25]]], dtype=DTYPE), [batch_size, 1, 1]),
    }


def _value_observation(points: tf.Tensor) -> tf.Tensor:
    return points


def _value_observation_jacobian(points: tf.Tensor) -> tf.Tensor:
    batch_size = points.shape[0]
    num_particles = points.shape[1]
    if batch_size is None or num_particles is None:
        raise ValueError("value fixture requires static dimensions")
    return tf.ones([batch_size, num_particles, 1, 1], dtype=DTYPE)


def _value_observation_residual(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
    return observation[None, None, :] - h_ref


def _make_value_transition_log_density(
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
):
    def _transition_log_density(
        x_next: tf.Tensor,
        x_prev: tf.Tensor,
        _time_index: tf.Tensor,
    ) -> tf.Tensor:
        del _time_index
        mean = tf.einsum("bnj,bdj->bnd", x_prev, transition_matrix)
        residual = x_next - mean
        logdet = tf.math.log(transition_covariance[:, 0, 0])
        quad = (
            residual[:, :, 0]
            * residual[:, :, 0]
            / transition_covariance[:, None, 0, 0]
        )
        return -0.5 * (
            tf.math.log(tf.constant(2.0 * np.pi, dtype=DTYPE)) + logdet[:, None] + quad
        )

    return _transition_log_density


def _value_observation_log_density(
    x: tf.Tensor,
    observation: tf.Tensor,
    _time_index: tf.Tensor,
) -> tf.Tensor:
    del _time_index
    variance = tf.constant(0.25, dtype=DTYPE)
    residual = x[:, :, 0] - observation[0]
    return -0.5 * (
        tf.math.log(tf.constant(2.0 * np.pi, dtype=DTYPE))
        + tf.math.log(variance)
        + residual * residual / variance
    )


def _run_value(fixture: dict[str, tf.Tensor]):
    return batched_ledh_pfpf_ot_value_core_tf(
        **fixture,
        observation_fn=_value_observation,
        observation_jacobian_fn=_value_observation_jacobian,
        observation_residual_fn=_value_observation_residual,
        transition_log_density_fn=_make_value_transition_log_density(
            fixture["transition_matrix"],
            fixture["transition_covariance"],
        ),
        observation_log_density_fn=_value_observation_log_density,
        sinkhorn_iterations=10,
    )


def _run_value_with_transport_plan(
    fixture: dict[str, tf.Tensor],
    *,
    transport_plan_mode: str,
):
    return batched_ledh_pfpf_ot_value_core_tf(
        **fixture,
        observation_fn=_value_observation,
        observation_jacobian_fn=_value_observation_jacobian,
        observation_residual_fn=_value_observation_residual,
        transition_log_density_fn=_make_value_transition_log_density(
            fixture["transition_matrix"],
            fixture["transition_covariance"],
        ),
        observation_log_density_fn=_value_observation_log_density,
        sinkhorn_iterations=8,
        transport_gradient_mode="raw",
        transport_plan_mode=transport_plan_mode,
        row_chunk_size=4,
        col_chunk_size=4,
    )


def _select_row(fixture: dict[str, tf.Tensor], row: int) -> dict[str, tf.Tensor]:
    selected: dict[str, tf.Tensor] = {}
    for key, value in fixture.items():
        if key == "observations":
            selected[key] = value
        elif value.shape.rank is not None and value.shape.rank >= 1:
            selected[key] = value[row : row + 1]
        else:
            selected[key] = value
    return selected


def _scalar_stack_value(fixture: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    values = []
    means = []
    variances = []
    esses = []
    batch_size = fixture["initial_particles"].shape[0]
    if batch_size is None:
        raise ValueError("fixture requires static batch size")
    for row in range(batch_size):
        result = _run_value(_select_row(fixture, row))
        values.append(result.log_likelihood[0])
        means.append(result.filtered_means[:, 0, :])
        variances.append(result.filtered_variances[:, 0, :])
        esses.append(result.ess_by_time[:, 0])
    return (
        tf.stack(values, axis=0),
        tf.stack(means, axis=1),
        tf.stack(variances, axis=1),
        tf.stack(esses, axis=1),
    )


def test_batched_value_recursion_b1_matches_scalar_row() -> None:
    fixture = _value_fixture(1)
    batched = _run_value(fixture)
    scalar_value, scalar_means, scalar_variances, scalar_ess = _scalar_stack_value(fixture)

    np.testing.assert_allclose(
        batched.log_likelihood.numpy(),
        scalar_value.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )
    np.testing.assert_allclose(
        batched.filtered_means.numpy(),
        scalar_means.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )
    np.testing.assert_allclose(
        batched.filtered_variances.numpy(),
        scalar_variances.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )
    np.testing.assert_allclose(
        batched.ess_by_time.numpy(),
        scalar_ess.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )


def test_batched_value_recursion_b20_matches_scalar_stack() -> None:
    fixture = _value_fixture(20)
    batched = _run_value(fixture)
    scalar_value, scalar_means, scalar_variances, scalar_ess = _scalar_stack_value(fixture)

    np.testing.assert_allclose(
        batched.log_likelihood.numpy(),
        scalar_value.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )
    np.testing.assert_allclose(
        batched.filtered_means.numpy(),
        scalar_means.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )
    np.testing.assert_allclose(
        batched.filtered_variances.numpy(),
        scalar_variances.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )
    np.testing.assert_allclose(
        batched.ess_by_time.numpy(),
        scalar_ess.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )


def test_batched_value_streaming_transport_matches_dense_value() -> None:
    fixture = _value_fixture(3)
    dense = _run_value_with_transport_plan(fixture, transport_plan_mode="dense")
    streaming = _run_value_with_transport_plan(fixture, transport_plan_mode="streaming")

    np.testing.assert_allclose(
        streaming.log_likelihood.numpy(),
        dense.log_likelihood.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )
    np.testing.assert_allclose(
        streaming.filtered_means.numpy(),
        dense.filtered_means.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )
    np.testing.assert_allclose(
        streaming.ess_by_time.numpy(),
        dense.ess_by_time.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )


def test_batched_value_row_permutation_and_identical_rows() -> None:
    fixture = _value_fixture(5)
    base = _run_value(fixture)
    permutation = tf.constant([4, 2, 0, 3, 1], dtype=tf.int32)
    permuted_fixture = {
        key: value if key == "observations" else tf.gather(value, permutation, axis=0)
        for key, value in fixture.items()
    }
    permuted = _run_value(permuted_fixture)
    np.testing.assert_allclose(
        permuted.log_likelihood.numpy(),
        tf.gather(base.log_likelihood, permutation).numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )

    one = _select_row(fixture, 2)
    repeated_fixture = {
        key: value if key == "observations" else tf.repeat(value, repeats=4, axis=0)
        for key, value in one.items()
    }
    repeated = _run_value(repeated_fixture)
    np.testing.assert_allclose(
        repeated.log_likelihood.numpy(),
        tf.repeat(repeated.log_likelihood[:1], repeats=4, axis=0).numpy(),
        atol=0.0,
    )


def test_phase3_value_core_tf_function_smoke() -> None:
    fixture = _value_fixture(3)

    @tf.function(reduce_retracing=True)
    def graph_value(
        observations: tf.Tensor,
        initial_particles: tf.Tensor,
        pre_flow_particles: tf.Tensor,
        fixed_resampling_mask: tf.Tensor,
        transition_matrix: tf.Tensor,
        transition_covariance: tf.Tensor,
        observation_covariance: tf.Tensor,
    ) -> tf.Tensor:
        return batched_ledh_pfpf_ot_value_core_tf(
            observations=observations,
            initial_particles=initial_particles,
            pre_flow_particles=pre_flow_particles,
            fixed_resampling_mask=fixed_resampling_mask,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            observation_fn=_value_observation,
            observation_jacobian_fn=_value_observation_jacobian,
            observation_residual_fn=_value_observation_residual,
            transition_log_density_fn=_make_value_transition_log_density(
                transition_matrix,
                transition_covariance,
            ),
            observation_log_density_fn=_value_observation_log_density,
            sinkhorn_iterations=10,
        ).log_likelihood

    eager = _run_value(fixture).log_likelihood
    graph = graph_value(**fixture)
    np.testing.assert_allclose(
        graph.numpy(),
        eager.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )
    assert len(graph_value._list_all_concrete_functions_for_serialization()) == 1


def test_phase3_value_core_source_has_no_numpy_rng_or_runtime_ess_branch() -> None:
    source = inspect.getsource(batched_ledh_pfpf_ot_value_core_tf)
    assert ".numpy(" not in source
    assert "tf.random" not in source
    assert "np.random" not in source
    assert "random." not in source
    assert "ess <" not in source
    assert "ess_threshold" not in source


def _score_theta_batch(batch_size: int = 3) -> tf.Tensor:
    rows = tf.cast(tf.range(batch_size), DTYPE)
    return tf.stack(
        [
            0.82 + 0.01 * rows,
            -1.8325814637483102 + 0.02 * rows,
            -1.3862943611198906 + 0.015 * rows,
        ],
        axis=1,
    )


def _score_fixture(
    batch_size: int = 3,
    *,
    active_transport: bool = True,
) -> dict[str, tf.Tensor]:
    fixture = _value_fixture(batch_size)
    if active_transport:
        fixed_resampling_mask = fixture["fixed_resampling_mask"]
    else:
        fixed_resampling_mask = tf.zeros_like(fixture["fixed_resampling_mask"])
    return {
        "observations": fixture["observations"],
        "initial_particles": fixture["initial_particles"],
        "pre_flow_particles": fixture["pre_flow_particles"],
        "fixed_resampling_mask": fixed_resampling_mask,
    }


def _value_from_theta(
    theta_batch: tf.Tensor,
    fixed_fixture: dict[str, tf.Tensor],
    *,
    transport_gradient_mode: str = "raw",
) -> BatchedLEDHPFPFOTValueTensors:
    transition_matrix = theta_batch[:, 0:1, None]
    transition_covariance = tf.exp(theta_batch[:, 1:2])[:, :, None]
    observation_covariance = tf.exp(theta_batch[:, 2:3])[:, :, None]
    return batched_ledh_pfpf_ot_value_core_tf(
        observations=fixed_fixture["observations"],
        initial_particles=fixed_fixture["initial_particles"],
        pre_flow_particles=fixed_fixture["pre_flow_particles"],
        fixed_resampling_mask=fixed_fixture["fixed_resampling_mask"],
        transition_matrix=transition_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        observation_fn=_value_observation,
        observation_jacobian_fn=_value_observation_jacobian,
        observation_residual_fn=_value_observation_residual,
        transition_log_density_fn=_make_value_transition_log_density(
            transition_matrix,
            transition_covariance,
        ),
        observation_log_density_fn=_value_observation_log_density,
        sinkhorn_iterations=10,
        transport_gradient_mode=transport_gradient_mode,
    )


def _value_score_from_theta(
    theta_batch: tf.Tensor,
    fixed_fixture: dict[str, tf.Tensor],
    *,
    transport_gradient_mode: str = "raw",
):
    return batched_ledh_pfpf_ot_value_and_score_tf(
        theta_batch,
        lambda values: _value_from_theta(
            values,
            fixed_fixture,
            transport_gradient_mode=transport_gradient_mode,
        ),
    )


def _central_finite_difference_score(
    theta_batch: tf.Tensor,
    fixed_fixture: dict[str, tf.Tensor],
    *,
    step: float = 1.0e-5,
) -> tf.Tensor:
    theta_np = theta_batch.numpy()
    score = np.zeros_like(theta_np)
    for row in range(theta_np.shape[0]):
        for param in range(theta_np.shape[1]):
            direction = np.zeros_like(theta_np)
            direction[row, param] = step
            plus = _value_from_theta(
                tf.constant(theta_np + direction, dtype=DTYPE),
                fixed_fixture,
                transport_gradient_mode="raw",
            ).log_likelihood[row]
            minus = _value_from_theta(
                tf.constant(theta_np - direction, dtype=DTYPE),
                fixed_fixture,
                transport_gradient_mode="raw",
            ).log_likelihood[row]
            score[row, param] = (float(plus.numpy()) - float(minus.numpy())) / (
                2.0 * step
            )
    return tf.constant(score, dtype=DTYPE)


def test_phase4_value_and_score_shape_finite_and_graph_smoke() -> None:
    theta = _score_theta_batch(3)
    fixture = _score_fixture(3)
    eager = _value_score_from_theta(theta, fixture)

    assert eager.log_likelihood.shape == (3,)
    assert eager.score.shape == (3, 3)
    assert bool(tf.reduce_all(tf.math.is_finite(eager.log_likelihood)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(eager.score)).numpy())

    @tf.function(reduce_retracing=True)
    def graph(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        result = _value_score_from_theta(values, fixture)
        return result.log_likelihood, result.score

    graph_value, graph_score = graph(theta)
    np.testing.assert_allclose(graph_value.numpy(), eager.log_likelihood.numpy(), atol=1.0e-10)
    np.testing.assert_allclose(graph_score.numpy(), eager.score.numpy(), atol=1.0e-10)
    assert len(graph._list_all_concrete_functions_for_serialization()) == 1


def test_phase4_no_resampling_value_and_score_matches_central_finite_difference() -> None:
    theta = _score_theta_batch(2)
    fixture = _score_fixture(2, active_transport=False)
    result = _value_score_from_theta(theta, fixture, transport_gradient_mode="raw")
    finite_difference = _central_finite_difference_score(theta, fixture)

    np.testing.assert_allclose(
        result.score.numpy(),
        finite_difference.numpy(),
        rtol=2.0e-4,
        atol=2.0e-4,
    )


def test_phase4_value_and_score_row_locality() -> None:
    theta = _score_theta_batch(3)
    fixture = _score_fixture(3)
    base = _value_from_theta(theta, fixture, transport_gradient_mode="raw").log_likelihood
    perturbation = tf.constant(
        [[0.0, 0.0, 0.0], [0.05, -0.02, 0.03], [0.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    perturbed = _value_from_theta(
        theta + perturbation,
        fixture,
        transport_gradient_mode="raw",
    ).log_likelihood

    np.testing.assert_allclose(perturbed.numpy()[0], base.numpy()[0], atol=2.0e-4)
    np.testing.assert_allclose(perturbed.numpy()[2], base.numpy()[2], atol=2.0e-4)
    assert abs(float(perturbed.numpy()[1] - base.numpy()[1])) > 1.0e-5

    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = _value_from_theta(theta, fixture, transport_gradient_mode="raw").log_likelihood
    jacobian = tape.jacobian(value, theta)
    row_mask = tf.eye(theta.shape[0], dtype=tf.bool)[:, :, None]
    off_row = tf.where(row_mask, tf.zeros_like(jacobian), jacobian)
    np.testing.assert_allclose(off_row.numpy(), 0.0, atol=1.0e-10)


def test_phase4_value_and_score_source_has_no_numpy_rng_or_runtime_ess_branch() -> None:
    source = inspect.getsource(batched_ledh_pfpf_ot_value_and_score_tf)
    assert ".numpy(" not in source
    assert "tf.random" not in source
    assert "np.random" not in source
    assert "random." not in source
    assert "ess <" not in source
    assert "ess_threshold" not in source
    assert "GradientTape" in source


def test_m5_manual_dense_value_and_score_tiny_opt_in_smoke() -> None:
    theta = _score_theta_batch(2)
    fixture = _score_fixture(2)
    manual = _value_score_from_theta(
        theta,
        fixture,
        transport_gradient_mode=MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
    )

    assert manual.log_likelihood.shape == (2,)
    assert manual.score.shape == (2, 3)
    assert bool(tf.reduce_all(tf.math.is_finite(manual.log_likelihood)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(manual.score)).numpy())

    @tf.function(reduce_retracing=True)
    def graph(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        result = _value_score_from_theta(
            values,
            fixture,
            transport_gradient_mode=MANUAL_DENSE_FINITE_TRANSPORT_GRADIENT_MODE,
        )
        return result.log_likelihood, result.score

    graph_value, graph_score = graph(theta)
    np.testing.assert_allclose(
        graph_value.numpy(),
        manual.log_likelihood.numpy(),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        graph_score.numpy(),
        manual.score.numpy(),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert len(graph._list_all_concrete_functions_for_serialization()) == 1


def test_m6_manual_streaming_value_and_score_tiny_opt_in_smoke() -> None:
    theta = _score_theta_batch(2)
    fixture = _score_fixture(2)

    def value_from_theta(values: tf.Tensor):
        transition_matrix = values[:, 0:1, None]
        transition_covariance = tf.exp(values[:, 1:2])[:, :, None]
        observation_covariance = tf.exp(values[:, 2:3])[:, :, None]
        return batched_ledh_pfpf_ot_value_core_tf(
            observations=fixture["observations"],
            initial_particles=fixture["initial_particles"],
            pre_flow_particles=fixture["pre_flow_particles"],
            fixed_resampling_mask=fixture["fixed_resampling_mask"],
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            observation_fn=_value_observation,
            observation_jacobian_fn=_value_observation_jacobian,
            observation_residual_fn=_value_observation_residual,
            transition_log_density_fn=_make_value_transition_log_density(
                transition_matrix,
                transition_covariance,
            ),
            observation_log_density_fn=_value_observation_log_density,
            sinkhorn_iterations=2,
            transport_gradient_mode=MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
            transport_plan_mode="streaming",
            row_chunk_size=4,
            col_chunk_size=4,
        )

    manual = batched_ledh_pfpf_ot_value_and_score_tf(theta, value_from_theta)

    assert manual.log_likelihood.shape == (2,)
    assert manual.score.shape == (2, 3)
    assert bool(tf.reduce_all(tf.math.is_finite(manual.log_likelihood)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(manual.score)).numpy())

    @tf.function(reduce_retracing=True)
    def graph(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        result = batched_ledh_pfpf_ot_value_and_score_tf(values, value_from_theta)
        return result.log_likelihood, result.score

    graph_value, graph_score = graph(theta)
    np.testing.assert_allclose(
        graph_value.numpy(),
        manual.log_likelihood.numpy(),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        graph_score.numpy(),
        manual.score.numpy(),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert len(graph._list_all_concrete_functions_for_serialization()) == 1


def test_s6_blockwise_manual_streaming_value_and_score_tiny_opt_in_smoke() -> None:
    theta = _score_theta_batch(2)
    fixture = _score_fixture(2)

    def value_from_theta(values: tf.Tensor):
        transition_matrix = values[:, 0:1, None]
        transition_covariance = tf.exp(values[:, 1:2])[:, :, None]
        observation_covariance = tf.exp(values[:, 2:3])[:, :, None]
        return batched_ledh_pfpf_ot_value_core_tf(
            observations=fixture["observations"],
            initial_particles=fixture["initial_particles"],
            pre_flow_particles=fixture["pre_flow_particles"],
            fixed_resampling_mask=fixture["fixed_resampling_mask"],
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            observation_fn=_value_observation,
            observation_jacobian_fn=_value_observation_jacobian,
            observation_residual_fn=_value_observation_residual,
            transition_log_density_fn=_make_value_transition_log_density(
                transition_matrix,
                transition_covariance,
            ),
            observation_log_density_fn=_value_observation_log_density,
            sinkhorn_iterations=2,
            transport_gradient_mode=MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
            transport_plan_mode="streaming",
            transport_ad_mode="stabilized",
            row_chunk_size=4,
            col_chunk_size=4,
        )

    manual = batched_ledh_pfpf_ot_value_and_score_tf(theta, value_from_theta)

    assert manual.log_likelihood.shape == (2,)
    assert manual.score.shape == (2, 3)
    assert bool(tf.reduce_all(tf.math.is_finite(manual.log_likelihood)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(manual.score)).numpy())

    @tf.function(reduce_retracing=True)
    def graph(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        result = batched_ledh_pfpf_ot_value_and_score_tf(values, value_from_theta)
        return result.log_likelihood, result.score

    graph_value, graph_score = graph(theta)
    np.testing.assert_allclose(
        graph_value.numpy(),
        manual.log_likelihood.numpy(),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        graph_score.numpy(),
        manual.score.numpy(),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert len(graph._list_all_concrete_functions_for_serialization()) == 1
