import pytest
import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_conditional_reference_tf import (
    finite_value_and_analytical_score,
    finite_value_and_autodiff_score,
    make_conditional_innovation_cloud,
)


def test_conditional_innovation_cloud_is_batch_native_and_finite() -> None:
    theta = tf.zeros([3], tf.float64)
    cloud = make_conditional_innovation_cloud(
        theta=theta, sample_count=32, seed=93001, role="test"
    )
    assert cloud.z0.shape == (32, 18)
    assert cloud.z1.shape == (32, 18)
    assert bool(tf.reduce_all(tf.math.is_finite(cloud.z0)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(cloud.z1)).numpy())


def test_conditional_reference_analytical_score_matches_same_program_autodiff() -> None:
    theta = tf.constant([0.0, 0.0, 0.0], tf.float64)
    cloud = make_conditional_innovation_cloud(
        theta=theta, sample_count=256, seed=93002, role="parity"
    )
    analytical = finite_value_and_analytical_score(
        theta, cloud.initial_noise, cloud.transition_noise
    )
    autodiff = finite_value_and_autodiff_score(
        theta, cloud.initial_noise, cloud.transition_noise
    )
    tf.debugging.assert_near(analytical["log_value"], autodiff["log_value"], atol=1e-12, rtol=1e-12)
    tf.debugging.assert_near(analytical["score"], autodiff["score"], atol=1e-8, rtol=1e-8)
    assert float(analytical["effective_sample_size"].numpy()) >= 128.0


def test_conditional_reference_reproducible_for_fixed_seed() -> None:
    theta = tf.zeros([3], tf.float64)
    first = make_conditional_innovation_cloud(
        theta=theta, sample_count=16, seed=93003, role="same"
    )
    second = make_conditional_innovation_cloud(
        theta=theta, sample_count=16, seed=93003, role="same"
    )
    tf.debugging.assert_equal(first.initial_noise, second.initial_noise)
    tf.debugging.assert_equal(first.transition_noise, second.transition_noise)
    tf.debugging.assert_equal(first.z1, second.z1)


def test_conditional_reference_has_finite_nonzero_theta_derivative() -> None:
    origin = tf.zeros([3], tf.float64)
    cloud = make_conditional_innovation_cloud(
        theta=origin, sample_count=128, seed=93005, role="nonzero"
    )
    theta = tf.constant([0.02, -0.01, 0.03], tf.float64)
    result = finite_value_and_autodiff_score(
        theta, cloud.initial_noise, cloud.transition_noise
    )
    assert bool(tf.reduce_all(tf.math.is_finite(result["score"])).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(result["log_value"])).numpy())


def test_conditional_reference_rejects_wrong_sample_shape() -> None:
    with pytest.raises(ValueError, match="sample_count must be at least two"):
        make_conditional_innovation_cloud(
            theta=tf.zeros([3], tf.float64), sample_count=1, seed=93004, role="bad"
        )
