from __future__ import annotations

import tensorflow as tf

from bayesfilter.highdim.sv_mixture_cut4 import ksc_1998_log_chi_square_mixture
from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
    generate_frozen_exact_sv_dataset_tf,
)
from bayesfilter.testing.ksc_gaussian_sum_ukf_neutra_target_tf import (
    ksc_gaussian_sum_ukf_likelihood_value_score_status,
)
from bayesfilter.testing.ksc_ukf_neutra_target_tf import transformed_ksc_observations


def _evaluate(theta: tf.Tensor, *, horizon: int = 4, cap: int = 32):
    _, raw = generate_frozen_exact_sv_dataset_tf(horizon=horizon)
    mixture = ksc_1998_log_chi_square_mixture()
    return ksc_gaussian_sum_ukf_likelihood_value_score_status(
        theta,
        transformed_observations=transformed_ksc_observations(raw),
        mixture_weights=mixture.weights,
        mixture_means=mixture.means,
        mixture_variances=mixture.variances,
        component_cap=cap,
    )


def test_gaussian_sum_score_matches_same_program_finite_difference() -> None:
    theta = tf.constant([[0.0, 0.0], [0.2, -0.3]], tf.float64)
    value, score, status = _evaluate(theta)
    columns = []
    for coordinate in range(2):
        direction = tf.one_hot(coordinate, 2, dtype=tf.float64)[None, :]
        plus = _evaluate(theta + 1.0e-5 * direction)[0]
        minus = _evaluate(theta - 1.0e-5 * direction)[0]
        columns.append((plus - minus) / 2.0e-5)
    tf.debugging.assert_near(score, tf.stack(columns, axis=1), atol=2e-6, rtol=2e-6)
    tf.debugging.assert_all_finite(value, "value")
    tf.debugging.assert_equal(status["status_code"], tf.zeros([2], tf.int32))


def test_gaussian_sum_is_batch_native_and_retains_multiple_components() -> None:
    theta = tf.constant([[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]], tf.float64)
    value, score, status = _evaluate(theta, cap=16)
    assert value.shape == (3,)
    assert score.shape == (3, 2)
    assert bool(tf.reduce_all(status["maximum_active_component_count"] > 1).numpy())
    tf.debugging.assert_near(
        status["minimum_retained_mass_fraction"],
        tf.ones([3], tf.float64),
        atol=2e-14,
        rtol=2e-14,
    )
    assert bool(
        tf.reduce_all(status["minimum_premerge_top_weight_mass_fraction"] < 1.0).numpy()
    )


def test_gaussian_sum_cpu_xla_has_fixed_observation_loop() -> None:
    theta = tf.constant(
        [[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]],
        tf.float64,
    )

    _, raw = generate_frozen_exact_sv_dataset_tf(horizon=20)
    mixture = ksc_1998_log_chi_square_mixture()
    transformed = transformed_ksc_observations(raw)

    @tf.function(
        input_signature=[tf.TensorSpec([None, 2], tf.float64)],
        jit_compile=True,
    )
    def compiled(values):
        return ksc_gaussian_sum_ukf_likelihood_value_score_status(
            values,
            transformed_observations=transformed,
            mixture_weights=mixture.weights,
            mixture_means=mixture.means,
            mixture_variances=mixture.variances,
            component_cap=32,
        )

    value, score, status = compiled(theta)
    tf.debugging.assert_all_finite(value, "XLA Gaussian-sum value")
    tf.debugging.assert_all_finite(score, "XLA Gaussian-sum score")
    tf.debugging.assert_equal(status["status_code"], tf.zeros([5], tf.int32))


def test_gaussian_sum_cpu_xla_accepts_dynamic_batch_signature() -> None:
    _, raw = generate_frozen_exact_sv_dataset_tf(horizon=20)
    mixture = ksc_1998_log_chi_square_mixture()
    transformed = transformed_ksc_observations(raw)

    @tf.function(
        input_signature=[tf.TensorSpec([None, 2], tf.float64)],
        jit_compile=True,
    )
    def compiled(values):
        return ksc_gaussian_sum_ukf_likelihood_value_score_status(
            values,
            transformed_observations=transformed,
            mixture_weights=mixture.weights,
            mixture_means=mixture.means,
            mixture_variances=mixture.variances,
            component_cap=32,
        )

    for size in (2, 5):
        value, score, status = compiled(tf.zeros([size, 2], tf.float64))
        assert value.shape == (size,)
        assert score.shape == (size, 2)
        tf.debugging.assert_equal(status["status_code"], tf.zeros([size], tf.int32))
