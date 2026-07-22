from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.highdim.cubature_genut_adapters import (
    structural_ukf_candidate_adapter,
)
from bayesfilter.highdim.cubature_genut_candidate import (
    gaussian_genut_design,
    replicate_positive_genut,
)
from bayesfilter.highdim.cubature_genut_filter import finite_value_score
from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
    STRUCTURAL_PARAMETER_LOWER,
    STRUCTURAL_PARAMETER_UPPER,
    generate_frozen_structural_dataset_tf,
    structural_observation_log_density_dtype,
    structural_source_chart,
    structural_source_chart_dtype,
    structural_transition_residual_dtype,
    structural_transition_tangent_dtype,
    structural_transition_value,
    structural_transition_value_dtype,
    structural_truth_source,
)


N = 1002


def _noise(horizon: int) -> tuple[tf.Tensor, tf.Tensor]:
    return (
        tf.random.stateless_normal([N, 2], [20260722, 101], dtype=tf.float32),
        tf.random.stateless_normal(
            [horizon, N, 1], [20260722, 102], dtype=tf.float32
        ),
    )


def _design() -> tf.Tensor:
    return replicate_positive_genut(
        gaussian_genut_design(dim=2), num_particles=N
    )


def test_dtype_generic_primitives_preserve_established_float64_values() -> None:
    theta = tf.stack([structural_truth_source(), tf.zeros([5], tf.float64)])
    previous = tf.random.stateless_normal([2, N, 2], [11, 12], dtype=tf.float64)
    innovation = tf.random.stateless_normal([2, N, 1], [13, 14], dtype=tf.float64)

    old_physical, old_derivative = structural_source_chart(theta)
    new_physical, new_derivative = structural_source_chart_dtype(theta)
    tf.debugging.assert_equal(old_physical, new_physical)
    tf.debugging.assert_equal(old_derivative, new_derivative)
    tf.debugging.assert_equal(
        structural_transition_value(theta, previous, innovation),
        structural_transition_value_dtype(theta, previous, innovation),
    )


def test_structural_transition_tangent_matches_diagnostic_forward_accumulator() -> None:
    theta = tf.cast(structural_truth_source()[None, :], tf.float32)
    previous = tf.random.stateless_normal([1, N, 2], [21, 22], dtype=tf.float32)
    innovation = tf.random.stateless_normal([1, N, 1], [23, 24], dtype=tf.float32)
    previous_tangent = tf.random.stateless_normal(
        [1, N, 2, 5], [25, 26], dtype=tf.float32
    )
    direction = tf.constant([[0.2, -0.1, 0.05, -0.08, 0.03]], tf.float32)
    point_direction = tf.einsum("bnsp,bp->bns", previous_tangent, direction)
    with tf.autodiff.ForwardAccumulator(
        (theta, previous), (direction, point_direction)
    ) as accumulator:
        output = structural_transition_value_dtype(theta, previous, innovation)
    automatic = accumulator.jvp(output)
    manual = tf.einsum(
        "bnsp,bp->bns",
        structural_transition_tangent_dtype(
            theta, previous, innovation, previous_tangent
        ),
        direction,
    )
    tf.debugging.assert_near(manual, automatic, atol=2e-6, rtol=2e-6)


def test_adapter_uses_scalar_process_noise_and_preserves_transition_support() -> None:
    adapter = structural_ukf_candidate_adapter()
    theta = tf.cast(structural_truth_source(), tf.float32)
    initial, process = _noise(2)
    particles = adapter.initial_value(theta, initial)
    tangent = adapter.initial_tangent(theta, initial)
    current = adapter.transition_value(theta, particles, process[1], tf.constant(1))
    current_tangent = adapter.transition_tangent(
        theta, particles, process[1], tangent, tf.constant(1)
    )

    assert process.shape == (2, N, 1)
    assert current.shape == (N, 2)
    assert current_tangent.shape == (N, 2, 5)
    residual = structural_transition_residual_dtype(
        theta[None, :], particles[None, :, :], current[None, :, :]
    )
    tf.debugging.assert_near(residual, tf.zeros_like(residual), atol=2e-6)


def test_artificial_k_process_noise_is_detected_by_structural_residual() -> None:
    adapter = structural_ukf_candidate_adapter()
    theta = tf.cast(structural_truth_source(), tf.float32)
    initial, process = _noise(2)
    particles = adapter.initial_value(theta, initial)
    current = adapter.transition_value(theta, particles, process[1], tf.constant(1))
    wrong = current + tf.constant([0.0, 0.02], tf.float32)[None, :]
    residual = adapter.transition_residual(theta, particles, wrong, tf.constant(1))
    tf.debugging.assert_greater(
        tf.reduce_max(tf.abs(residual)), tf.constant(1.9e-2, tf.float32)
    )


def test_time_zero_value_uses_initial_law_without_transition() -> None:
    adapter = structural_ukf_candidate_adapter()
    theta = tf.cast(structural_truth_source(), tf.float32)
    _states, observations64 = generate_frozen_structural_dataset_tf()
    observations = tf.cast(observations64[:1], tf.float32)
    initial, process = _noise(1)
    value, _score, diagnostics = finite_value_score(
        adapter,
        theta,
        observations,
        initial,
        process,
        _design(),
        transition_before_first_observation=False,
    )
    particles = adapter.initial_value(theta, initial)
    pointwise = structural_observation_log_density_dtype(
        theta[None, :], particles[None, :, :], observations[None, 0, :]
    )[0]
    expected = tf.reduce_logsumexp(pointwise) - tf.math.log(tf.cast(N, tf.float32))
    tf.debugging.assert_near(value, expected, atol=2e-5, rtol=2e-5)
    tf.debugging.assert_equal(
        diagnostics["max_transition_residual"], tf.constant(0.0, tf.float32)
    )


def test_t2_recursive_score_matches_same_scalar_finite_difference() -> None:
    adapter = structural_ukf_candidate_adapter()
    theta = tf.cast(structural_truth_source(), tf.float32)
    _states, observations64 = generate_frozen_structural_dataset_tf()
    observations = tf.cast(observations64[:2], tf.float32)
    initial, process = _noise(2)
    design = _design()

    def evaluate(values: tf.Tensor):
        return finite_value_score(
            adapter,
            values,
            observations,
            initial,
            process,
            design,
            epsilon=2.0,
            sinkhorn_steps=4,
            ridge=1e-5,
            transition_before_first_observation=False,
        )

    value, score, diagnostics = evaluate(theta)
    steps = tf.constant([2e-3, 2e-3, 2e-3, 2e-3, 2e-3], tf.float32)
    finite_difference = []
    for index in range(5):
        direction = tf.one_hot(index, 5, dtype=tf.float32)
        plus = evaluate(theta + steps[index] * direction)[0]
        minus = evaluate(theta - steps[index] * direction)[0]
        finite_difference.append((plus - minus) / (2.0 * steps[index]))
    finite_difference = tf.stack(finite_difference)
    tf.debugging.assert_near(score, finite_difference, atol=1e-2, rtol=1e-2)
    assert bool(tf.math.is_finite(value).numpy())
    assert float(diagnostics["max_transition_residual"].numpy()) < 2e-5


def test_runtime_adapter_has_no_autodiff_fd_or_duplicated_structural_constants() -> None:
    source = inspect.getsource(structural_ukf_candidate_adapter)
    assert "GradientTape" not in source
    assert "ForwardAccumulator" not in source
    assert "finite_difference" not in source
    assert "STRUCTURAL_PARAMETER_LOWER" not in source
    assert "STRUCTURAL_PARAMETER_UPPER" not in source
    assert "tf.while_loop" not in source
