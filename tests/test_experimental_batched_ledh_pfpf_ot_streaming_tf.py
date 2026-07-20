from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.experimental_batched_ledh_pfpf_ot_streaming_tf import (
    batched_ledh_flow_streaming_particles_tf,
    streaming_batched_ledh_pfpf_ot_value_and_score_tf,
    streaming_batched_ledh_pfpf_ot_value_core_tf,
)
from experiments.dpf_implementation.tf_tfp.filters.experimental_batched_ledh_pfpf_ot_tf import (
    DTYPE,
    SCALAR_PARITY_ATOL,
    SCALAR_PARITY_RTOL,
    TRANSPORT_PARITY_ATOL,
    TRANSPORT_PARITY_RTOL,
    batched_annealed_transport_core_tf,
    batched_ledh_flow_core_tf,
    batched_ledh_pfpf_ot_value_core_tf,
)


MODULE_PATH = Path(
    "experiments/dpf_implementation/tf_tfp/filters/"
    "experimental_batched_ledh_pfpf_ot_streaming_tf.py"
)
FP32_XLA_ATOL = 2.0e-7
FP32_XLA_RTOL = 2.0e-7
FP32_SCORE_FD_ATOL = 2.0e-2
FP32_SCORE_FD_RTOL = 6.0e-2


def _value_fixture(batch_size: int = 3) -> dict[str, tf.Tensor]:
    time_steps = 3
    num_particles = 4
    rows = tf.cast(tf.range(batch_size), DTYPE)
    base_particles = tf.reshape(
        tf.constant([-0.30, -0.05, 0.20, 0.45], dtype=DTYPE),
        [1, num_particles, 1],
    )
    initial_particles = base_particles + 0.02 * rows[:, None, None]
    transition_matrix = tf.reshape(0.82 + 0.01 * rows, [batch_size, 1, 1])
    offsets = tf.reshape(
        tf.constant([-0.04, -0.01, 0.02, 0.05], dtype=DTYPE),
        [1, 1, num_particles, 1],
    )
    time_offsets = 0.015 * tf.reshape(
        tf.cast(tf.range(time_steps), DTYPE),
        [1, time_steps, 1, 1],
    )
    pre_flow_particles = (
        transition_matrix[:, None, :, :] * initial_particles[:, None, :, :]
        + offsets
        + time_offsets
    )
    return {
        "observations": tf.reshape(tf.constant([0.03, 0.08, 0.12], dtype=DTYPE), [3, 1]),
        "initial_particles": initial_particles,
        "pre_flow_particles": pre_flow_particles,
        "fixed_resampling_mask": tf.tile(
            tf.constant([[False, True, False]], dtype=tf.bool),
            [batch_size, 1],
        ),
        "transition_matrix": transition_matrix,
        "transition_covariance": tf.tile(
            tf.constant([[[0.16]]], dtype=DTYPE),
            [batch_size, 1, 1],
        ),
        "observation_covariance": tf.tile(
            tf.constant([[[0.25]]], dtype=DTYPE),
            [batch_size, 1, 1],
        ),
    }


def _observation(points: tf.Tensor) -> tf.Tensor:
    return points


def _observation_jacobian(points: tf.Tensor) -> tf.Tensor:
    batch_size = points.shape[0]
    num_particles = points.shape[1]
    if batch_size is None or num_particles is None:
        raise ValueError("test fixture requires static dimensions")
    return tf.ones([batch_size, num_particles, 1, 1], dtype=DTYPE)


def _observation_residual(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
    return observation[None, None, :] - h_ref


def _make_transition_log_density(
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
        variance = transition_covariance[:, 0, 0]
        quad = residual[:, :, 0] * residual[:, :, 0] / variance[:, None]
        return -0.5 * (
            tf.math.log(tf.constant(2.0 * np.pi, dtype=DTYPE))
            + tf.math.log(variance)[:, None]
            + quad
        )

    return _transition_log_density


def _observation_log_density(
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


def _baseline_value(fixture: dict[str, tf.Tensor]):
    return batched_ledh_pfpf_ot_value_core_tf(
        **fixture,
        observation_fn=_observation,
        observation_jacobian_fn=_observation_jacobian,
        observation_residual_fn=_observation_residual,
        transition_log_density_fn=_make_transition_log_density(
            fixture["transition_matrix"],
            fixture["transition_covariance"],
        ),
        observation_log_density_fn=_observation_log_density,
        sinkhorn_iterations=8,
        transport_gradient_mode="raw",
        transport_plan_mode="streaming",
        row_chunk_size=4,
        col_chunk_size=4,
    )


def _streaming_value(
    fixture: dict[str, tf.Tensor],
    *,
    return_history: bool = True,
    pre_flow_step_fn=None,
    prior_mean_fn=None,
):
    kwargs = dict(
        observations=fixture["observations"],
        initial_particles=fixture["initial_particles"],
        fixed_resampling_mask=fixture["fixed_resampling_mask"],
        transition_matrix=fixture["transition_matrix"],
        transition_covariance=fixture["transition_covariance"],
        observation_covariance=fixture["observation_covariance"],
        observation_fn=_observation,
        observation_jacobian_fn=_observation_jacobian,
        observation_residual_fn=_observation_residual,
        transition_log_density_fn=_make_transition_log_density(
            fixture["transition_matrix"],
            fixture["transition_covariance"],
        ),
        observation_log_density_fn=_observation_log_density,
        sinkhorn_iterations=8,
        row_chunk_size=4,
        col_chunk_size=4,
        particle_chunk_size=2,
        return_history=return_history,
    )
    if prior_mean_fn is not None:
        kwargs["prior_mean_fn"] = prior_mean_fn
    if pre_flow_step_fn is None:
        kwargs["pre_flow_particles"] = fixture["pre_flow_particles"]
    else:
        kwargs["pre_flow_step_fn"] = pre_flow_step_fn
    return streaming_batched_ledh_pfpf_ot_value_core_tf(**kwargs)


def test_nonlinear_prior_mean_hook_preserves_default_linear_flow() -> None:
    fixture = _value_fixture(2)

    def linear_prior(points: tf.Tensor) -> tf.Tensor:
        return tf.einsum("bnj,bdj->bnd", points, fixture["transition_matrix"])

    default = batched_ledh_flow_streaming_particles_tf(
        pre_flow_particles=fixture["pre_flow_particles"][:, 0, :, :],
        ancestors=fixture["initial_particles"],
        observation=fixture["observations"][0],
        transition_matrix=fixture["transition_matrix"],
        transition_covariance=fixture["transition_covariance"],
        observation_covariance=fixture["observation_covariance"],
        observation_fn=_observation,
        observation_jacobian_fn=_observation_jacobian,
        observation_residual_fn=_observation_residual,
        particle_chunk_size=2,
    )
    hooked_linear = batched_ledh_flow_streaming_particles_tf(
        pre_flow_particles=fixture["pre_flow_particles"][:, 0, :, :],
        ancestors=fixture["initial_particles"],
        observation=fixture["observations"][0],
        transition_matrix=fixture["transition_matrix"],
        transition_covariance=fixture["transition_covariance"],
        observation_covariance=fixture["observation_covariance"],
        observation_fn=_observation,
        observation_jacobian_fn=_observation_jacobian,
        observation_residual_fn=_observation_residual,
        prior_mean_fn=linear_prior,
        particle_chunk_size=2,
    )

    np.testing.assert_allclose(
        hooked_linear.post_flow_particles.numpy(),
        default.post_flow_particles.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )
    np.testing.assert_allclose(
        hooked_linear.pre_flow_log_density.numpy(),
        default.pre_flow_log_density.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )


def test_nonlinear_prior_mean_hook_changes_streaming_value_core() -> None:
    fixture = _value_fixture(2)

    def nonlinear_prior(points: tf.Tensor, _time_index: tf.Tensor) -> tf.Tensor:
        linear = tf.einsum("bnj,bdj->bnd", points, fixture["transition_matrix"])
        return linear + tf.constant(0.25, dtype=DTYPE) * tf.square(points)

    linear = _streaming_value(fixture, return_history=False)
    nonlinear = _streaming_value(
        fixture,
        return_history=False,
        prior_mean_fn=nonlinear_prior,
    )

    assert not np.allclose(
        nonlinear.log_likelihood.numpy(),
        linear.log_likelihood.numpy(),
        atol=1.0e-7,
        rtol=1.0e-7,
    )


def test_streaming_flow_chunked_matches_dense_flow_core() -> None:
    fixture = _value_fixture(3)
    dense = batched_ledh_flow_core_tf(
        pre_flow_particles=fixture["pre_flow_particles"][:, 0, :, :],
        ancestors=fixture["initial_particles"],
        observation=fixture["observations"][0],
        transition_matrix=fixture["transition_matrix"],
        transition_covariance=fixture["transition_covariance"],
        observation_covariance=fixture["observation_covariance"],
        observation_fn=_observation,
        observation_jacobian_fn=_observation_jacobian,
        observation_residual_fn=_observation_residual,
    )
    streaming = batched_ledh_flow_streaming_particles_tf(
        pre_flow_particles=fixture["pre_flow_particles"][:, 0, :, :],
        ancestors=fixture["initial_particles"],
        observation=fixture["observations"][0],
        transition_matrix=fixture["transition_matrix"],
        transition_covariance=fixture["transition_covariance"],
        observation_covariance=fixture["observation_covariance"],
        observation_fn=_observation,
        observation_jacobian_fn=_observation_jacobian,
        observation_residual_fn=_observation_residual,
        particle_chunk_size=2,
    )

    np.testing.assert_allclose(
        streaming.post_flow_particles.numpy(),
        dense.post_flow_particles.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )
    np.testing.assert_allclose(
        streaming.pre_flow_log_density.numpy(),
        dense.pre_flow_log_density.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )
    np.testing.assert_allclose(
        streaming.forward_log_det.numpy(),
        dense.forward_log_det.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )


def test_streaming_value_core_matches_existing_baseline_with_history() -> None:
    fixture = _value_fixture(4)
    baseline = _baseline_value(fixture)
    streaming = _streaming_value(fixture, return_history=True)

    np.testing.assert_allclose(
        streaming.log_likelihood.numpy(),
        baseline.log_likelihood.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )
    np.testing.assert_allclose(
        streaming.filtered_means.numpy(),
        baseline.filtered_means.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )
    np.testing.assert_allclose(
        streaming.filtered_variances.numpy(),
        baseline.filtered_variances.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )
    np.testing.assert_allclose(
        streaming.ess_by_time.numpy(),
        baseline.ess_by_time.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )


def test_streaming_value_likelihood_only_omits_history() -> None:
    fixture = _value_fixture(3)
    with_history = _streaming_value(fixture, return_history=True)
    likelihood_only = _streaming_value(fixture, return_history=False)

    np.testing.assert_allclose(
        likelihood_only.log_likelihood.numpy(),
        with_history.log_likelihood.numpy(),
        atol=0.0,
    )
    assert likelihood_only.filtered_means.shape == (0, 3, 1)
    assert likelihood_only.filtered_variances.shape == (0, 3, 1)
    assert likelihood_only.ess_by_time.shape == (0, 3)


def test_inactive_transport_dynamic_mask_skips_transport_core() -> None:
    fixture = _value_fixture(3)
    fixture["fixed_resampling_mask"] = tf.Variable(
        tf.zeros_like(fixture["fixed_resampling_mask"]),
        trainable=False,
    )

    with mock.patch(
        "experiments.dpf_implementation.tf_tfp.filters."
        "experimental_batched_ledh_pfpf_ot_streaming_tf."
        "batched_annealed_transport_core_tf",
        side_effect=AssertionError("transport core should not be called"),
    ):
        result = _streaming_value(fixture, return_history=False)

    assert np.all(np.isfinite(result.log_likelihood.numpy()))


def test_mixed_active_transport_dynamic_mask_calls_transport_core() -> None:
    fixture = _value_fixture(3)
    mask = np.zeros(fixture["fixed_resampling_mask"].shape, dtype=bool)
    mask[0, 1] = True
    fixture["fixed_resampling_mask"] = tf.Variable(mask, trainable=False)
    call_count = 0

    def sentinel_transport(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return batched_annealed_transport_core_tf(*args, **kwargs)

    with mock.patch(
        "experiments.dpf_implementation.tf_tfp.filters."
        "experimental_batched_ledh_pfpf_ot_streaming_tf."
        "batched_annealed_transport_core_tf",
        side_effect=sentinel_transport,
    ):
        result = _streaming_value(fixture, return_history=False)

    assert call_count > 0
    assert np.all(np.isfinite(result.log_likelihood.numpy()))


def test_streaming_pre_flow_callback_matches_fixed_tensor_path() -> None:
    fixture = _value_fixture(2)

    def pre_flow_step(particles: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
        del particles
        return fixture["pre_flow_particles"][:, time_index, :, :]

    fixed = _streaming_value(fixture, return_history=True)
    callback = _streaming_value(
        fixture,
        return_history=True,
        pre_flow_step_fn=pre_flow_step,
    )
    np.testing.assert_allclose(
        callback.log_likelihood.numpy(),
        fixed.log_likelihood.numpy(),
        atol=SCALAR_PARITY_ATOL,
        rtol=SCALAR_PARITY_RTOL,
    )


def test_streaming_transport_returns_no_dense_transport_matrix() -> None:
    particles = tf.constant(
        [[[-0.50], [-0.10], [0.25], [0.60]]],
        dtype=DTYPE,
    )
    log_weights = tf.math.log(tf.constant([[0.10, 0.20, 0.30, 0.40]], dtype=DTYPE))
    mask = tf.constant([True], dtype=tf.bool)
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
    assert streaming.transport_matrix.shape == (1, 0, 0)
    np.testing.assert_allclose(
        streaming.particles.numpy(),
        dense.particles.numpy(),
        atol=FP32_XLA_ATOL,
        rtol=FP32_XLA_RTOL,
    )


def test_streaming_value_core_tf_function_jit_smoke() -> None:
    fixture = _value_fixture(2)
    eager = _streaming_value(fixture, return_history=False).log_likelihood

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(
        observations: tf.Tensor,
        initial_particles: tf.Tensor,
        pre_flow_particles: tf.Tensor,
        fixed_resampling_mask: tf.Tensor,
        transition_matrix: tf.Tensor,
        transition_covariance: tf.Tensor,
        observation_covariance: tf.Tensor,
    ) -> tf.Tensor:
        local_fixture = {
            "observations": observations,
            "initial_particles": initial_particles,
            "pre_flow_particles": pre_flow_particles,
            "fixed_resampling_mask": fixed_resampling_mask,
            "transition_matrix": transition_matrix,
            "transition_covariance": transition_covariance,
            "observation_covariance": observation_covariance,
        }
        return _streaming_value(local_fixture, return_history=False).log_likelihood

    graph = compiled(**fixture)
    np.testing.assert_allclose(
        graph.numpy(),
        eager.numpy(),
        atol=FP32_XLA_ATOL,
        rtol=FP32_XLA_RTOL,
    )
    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1


def test_streaming_value_and_score_no_resampling_matches_finite_difference() -> None:
    fixture = _value_fixture(2)
    fixture["fixed_resampling_mask"] = tf.zeros_like(fixture["fixed_resampling_mask"])
    theta = tf.stack(
        [
            fixture["transition_matrix"][:, 0, 0],
            tf.math.log(fixture["transition_covariance"][:, 0, 0]),
            tf.math.log(fixture["observation_covariance"][:, 0, 0]),
        ],
        axis=1,
    )

    def value_from_theta(values: tf.Tensor):
        local = dict(fixture)
        local["transition_matrix"] = values[:, 0:1, None]
        local["transition_covariance"] = tf.exp(values[:, 1:2])[:, :, None]
        local["observation_covariance"] = tf.exp(values[:, 2:3])[:, :, None]
        return _streaming_value(local, return_history=False)

    result = streaming_batched_ledh_pfpf_ot_value_and_score_tf(theta, value_from_theta)
    theta_np = theta.numpy()
    finite_difference = np.zeros_like(theta_np)
    step = 1.0e-5
    for row in range(theta_np.shape[0]):
        for param in range(theta_np.shape[1]):
            direction = np.zeros_like(theta_np)
            direction[row, param] = step
            plus = value_from_theta(tf.constant(theta_np + direction, dtype=DTYPE))
            minus = value_from_theta(tf.constant(theta_np - direction, dtype=DTYPE))
            finite_difference[row, param] = (
                plus.log_likelihood.numpy()[row] - minus.log_likelihood.numpy()[row]
            ) / (2.0 * step)

    np.testing.assert_allclose(
        result.score.numpy(),
        finite_difference,
        atol=FP32_SCORE_FD_ATOL,
        rtol=FP32_SCORE_FD_RTOL,
    )


def test_streaming_module_source_is_gpu_oriented() -> None:
    source = inspect.getsource(streaming_batched_ledh_pfpf_ot_value_core_tf)
    assert ".numpy(" not in source
    assert "tf.random" not in source
    assert "np.random" not in source
    assert "tf.while_loop" in source
    assert "for t in range" not in source
    assert "means = []" not in source
    assert "transport_plan_mode: str = \"streaming\"" in source
    assert "return_history: bool = False" in source

    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    forbidden_random_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"normal", "uniform"}:
            owner = func.value
            if isinstance(owner, ast.Attribute) and owner.attr == "random":
                forbidden_random_calls.append((func.attr, node.lineno))
    assert forbidden_random_calls == []
