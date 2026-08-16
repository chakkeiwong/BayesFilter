from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.testing.posterior_predictive_tf import (
    PosteriorPredictiveDiagnosticError,
    posterior_predictive_energy_test,
    sample_empirical_posterior_predictive_paths,
)
from bayesfilter.testing.two_sample_energy_tf import (
    whole_path_energy_permutation_test,
)


DTYPE = tf.float64


def _gaussian_simulator(noise_scale: float, horizon: int = 1):
    scale = tf.constant(noise_scale, DTYPE)

    def simulator(parameters: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        parameters = tf.convert_to_tensor(parameters, DTYPE)
        noise = tf.random.stateless_normal(
            [int(parameters.shape[0]), horizon],
            seed,
            dtype=DTYPE,
            alg="philox",
        )
        return parameters[:, :1] + scale * noise

    return simulator


def _normal_archive(count: int, mean: float, scale: float, seed: tuple[int, int]) -> tf.Tensor:
    return (
        tf.constant(mean, DTYPE)
        + tf.constant(scale, DTYPE)
        * tf.random.stateless_normal([count, 1], seed, dtype=DTYPE, alg="philox")
    )


def _mixture_archive(
    count: int,
    *,
    left_weight: float,
    left_mean: float,
    right_mean: float,
    component_scale: float,
    seed: tuple[int, int],
) -> tf.Tensor:
    root = tf.constant(seed, tf.int32)
    choose = tf.random.stateless_uniform(
        [count, 1],
        tf.random.experimental.stateless_fold_in(root, 1, alg="philox"),
        dtype=DTYPE,
        alg="philox",
    ) < tf.constant(left_weight, DTYPE)
    component_mean = tf.where(
        choose, tf.constant(left_mean, DTYPE), tf.constant(right_mean, DTYPE)
    )
    noise = tf.random.stateless_normal(
        [count, 1],
        tf.random.experimental.stateless_fold_in(root, 2, alg="philox"),
        dtype=DTYPE,
        alg="philox",
    )
    return component_mean + tf.constant(component_scale, DTYPE) * noise


def _analytic_mixture_predictive(
    count: int,
    *,
    left_weight: float,
    left_mean: float,
    right_mean: float,
    predictive_scale: float,
    seed: tuple[int, int],
) -> tf.Tensor:
    return _mixture_archive(
        count,
        left_weight=left_weight,
        left_mean=left_mean,
        right_mean=right_mean,
        component_scale=predictive_scale,
        seed=seed,
    )


def test_one_independent_posterior_index_is_aligned_with_every_path() -> None:
    posterior = tf.reshape(tf.cast(tf.range(7), DTYPE), [7, 1])

    def identity_simulator(parameters: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        del seed
        return tf.concat((parameters, parameters + 100.0), axis=1)

    first = sample_empirical_posterior_predictive_paths(
        posterior,
        path_count=25,
        posterior_seed=(20260809, 101),
        simulator_seed=(20260809, 102),
        conditional_simulator=identity_simulator,
    )
    second = sample_empirical_posterior_predictive_paths(
        posterior,
        path_count=25,
        posterior_seed=(20260809, 101),
        simulator_seed=(20260809, 102),
        conditional_simulator=identity_simulator,
    )
    tf.debugging.assert_equal(first.posterior_indices, second.posterior_indices)
    tf.debugging.assert_equal(first.paths, second.paths)
    tf.debugging.assert_equal(
        first.selected_parameters, tf.gather(posterior, first.posterior_indices)
    )
    tf.debugging.assert_equal(first.paths[:, :1], first.selected_parameters)
    assert first.path_count == 25 > first.posterior_draw_count
    assert first.sampling_with_replacement is True
    assert int(tf.size(tf.unique(first.posterior_indices).y)) > 1


def test_unimodal_empirical_predictive_matches_analytic_gaussian_and_detects_shift() -> None:
    posterior_mean = 0.7
    posterior_scale = 0.8
    simulator_scale = 0.6
    predictive_scale = (posterior_scale**2 + simulator_scale**2) ** 0.5
    posterior = _normal_archive(
        32768, posterior_mean, posterior_scale, (20260809, 201)
    )
    bank = sample_empirical_posterior_predictive_paths(
        posterior,
        path_count=512,
        posterior_seed=(20260809, 202),
        simulator_seed=(20260809, 203),
        conditional_simulator=_gaussian_simulator(simulator_scale),
    )
    analytic = _normal_archive(
        512, posterior_mean, predictive_scale, (20260809, 204)
    )
    null = whole_path_energy_permutation_test(
        bank.paths,
        analytic,
        permutation_count=999,
        seed=(20260809, 205),
        permutation_batch_size=100,
        jit_compile=True,
    )
    shifted = whole_path_energy_permutation_test(
        bank.paths,
        analytic + 2.0,
        permutation_count=999,
        seed=(20260809, 206),
        permutation_batch_size=100,
        jit_compile=True,
    )
    assert float(null.p_value) >= 0.01
    assert float(shifted.p_value) < 0.01
    assert float(tf.abs(tf.reduce_mean(bank.paths) - posterior_mean)) < 0.15
    assert float(
        tf.abs(tf.math.reduce_std(bank.paths) - predictive_scale)
    ) < 0.15


def test_multimodal_empirical_predictive_matches_analytic_mixture_and_selects_both_modes() -> None:
    left_weight = 0.7
    left_mean = -3.0
    right_mean = 4.0
    posterior_scale = 0.35
    simulator_scale = 0.45
    predictive_scale = (posterior_scale**2 + simulator_scale**2) ** 0.5
    posterior = _mixture_archive(
        32768,
        left_weight=left_weight,
        left_mean=left_mean,
        right_mean=right_mean,
        component_scale=posterior_scale,
        seed=(20260809, 301),
    )
    bank = sample_empirical_posterior_predictive_paths(
        posterior,
        path_count=512,
        posterior_seed=(20260809, 302),
        simulator_seed=(20260809, 303),
        conditional_simulator=_gaussian_simulator(simulator_scale),
    )
    analytic = _analytic_mixture_predictive(
        512,
        left_weight=left_weight,
        left_mean=left_mean,
        right_mean=right_mean,
        predictive_scale=predictive_scale,
        seed=(20260809, 304),
    )
    result = whole_path_energy_permutation_test(
        bank.paths,
        analytic,
        permutation_count=999,
        seed=(20260809, 305),
        permutation_batch_size=100,
        jit_compile=True,
    )
    selected_left_fraction = tf.reduce_mean(
        tf.cast(bank.selected_parameters[:, 0] < 0.0, DTYPE)
    )
    assert float(result.p_value) >= 0.01
    assert 0.60 < float(selected_left_fraction) < 0.80


@pytest.mark.parametrize("alternative", ("wrong_weight", "collapsed_mode"))
def test_multimodal_diagnostic_detects_wrong_weights_and_mode_collapse(
    alternative: str,
) -> None:
    posterior_scale = 0.30
    simulator_scale = 0.35
    correct = _analytic_mixture_predictive(
        512,
        left_weight=0.7,
        left_mean=-4.0,
        right_mean=4.0,
        predictive_scale=(posterior_scale**2 + simulator_scale**2) ** 0.5,
        seed=(20260809, 401),
    )
    left_weight = 0.2 if alternative == "wrong_weight" else 1.0
    archive = _mixture_archive(
        32768,
        left_weight=left_weight,
        left_mean=-4.0,
        right_mean=4.0,
        component_scale=posterior_scale,
        seed=(20260809, 402 if alternative == "wrong_weight" else 403),
    )
    bank = sample_empirical_posterior_predictive_paths(
        archive,
        path_count=512,
        posterior_seed=(20260809, 404),
        simulator_seed=(20260809, 405),
        conditional_simulator=_gaussian_simulator(simulator_scale),
    )
    result = whole_path_energy_permutation_test(
        bank.paths,
        correct,
        permutation_count=999,
        seed=(20260809, 406),
        permutation_batch_size=100,
        jit_compile=True,
    )
    assert float(result.p_value) < 0.01


def test_high_level_diagnostic_uses_disjoint_arms_and_detects_wrong_truth() -> None:
    posterior = _normal_archive(8192, 2.5, 0.05, (20260809, 501))
    result = posterior_predictive_energy_test(
        posterior,
        tf.constant([0.0], DTYPE),
        path_count=128,
        posterior_seed=(20260809, 502),
        posterior_simulator_seed=(20260809, 503),
        truth_simulator_seed=(20260809, 504),
        permutation_seed=(20260809, 505),
        conditional_simulator=_gaussian_simulator(0.2, horizon=3),
        permutation_count=999,
        permutation_batch_size=100,
        jit_compile=True,
    )
    assert result.posterior_predictive.paths.shape == (128, 3)
    assert result.true_paths.shape == (128, 3)
    assert float(result.energy.p_value) < 0.01


def test_contract_rejects_seed_alias_shape_nonfinite_and_many_paths_per_row() -> None:
    posterior = tf.zeros([4, 1], DTYPE)
    simulator = _gaussian_simulator(1.0)
    with pytest.raises(PosteriorPredictiveDiagnosticError, match="disjoint"):
        sample_empirical_posterior_predictive_paths(
            posterior,
            path_count=4,
            posterior_seed=(1, 2),
            simulator_seed=(1, 2),
            conditional_simulator=simulator,
        )
    with pytest.raises(PosteriorPredictiveDiagnosticError, match="finite"):
        sample_empirical_posterior_predictive_paths(
            tf.constant([[float("nan")]], DTYPE),
            path_count=4,
            posterior_seed=(1, 2),
            simulator_seed=(1, 3),
            conditional_simulator=simulator,
        )
    with pytest.raises(PosteriorPredictiveDiagnosticError, match="one complete path"):
        sample_empirical_posterior_predictive_paths(
            posterior,
            path_count=4,
            posterior_seed=(1, 2),
            simulator_seed=(1, 3),
            conditional_simulator=lambda rows, seed: tf.zeros([4, 2, 3], DTYPE),
        )
    with pytest.raises(PosteriorPredictiveDiagnosticError, match="true_parameter"):
        posterior_predictive_energy_test(
            posterior,
            tf.zeros([2], DTYPE),
            path_count=4,
            posterior_seed=(1, 2),
            posterior_simulator_seed=(1, 3),
            truth_simulator_seed=(1, 4),
            permutation_seed=(1, 5),
            conditional_simulator=simulator,
            permutation_count=9,
            jit_compile=False,
        )


def test_multimodal_fixture_has_the_declared_closed_form_moments() -> None:
    weight = tf.constant(0.7, DTYPE)
    left = tf.constant(-3.0, DTYPE)
    right = tf.constant(4.0, DTYPE)
    scale = tf.constant(0.8, DTYPE)
    analytic_mean = weight * left + (1.0 - weight) * right
    analytic_variance = (
        tf.square(scale)
        + weight * tf.square(left - analytic_mean)
        + (1.0 - weight) * tf.square(right - analytic_mean)
    )
    sample = _analytic_mixture_predictive(
        100000,
        left_weight=float(weight),
        left_mean=float(left),
        right_mean=float(right),
        predictive_scale=float(scale),
        seed=(20260809, 601),
    )
    assert float(tf.abs(tf.reduce_mean(sample) - analytic_mean)) < 0.05
    assert float(tf.abs(tf.math.reduce_variance(sample) - analytic_variance)) < 0.12
