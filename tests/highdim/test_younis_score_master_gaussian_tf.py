"""Independent Gaussian identities and diagnostic FD, with GPU hidden by tests."""
import math

import pytest
import tensorflow as tf

from bayesfilter.score_study.gaussian_tf import (
    make_gaussian_kernel, make_particle_kernel, parameterized_model,
)


THETA = tf.constant([0.62, -0.8, -0.6, 0.9, 0.25, -0.3], tf.float64)
OBSERVATIONS = tf.constant([[0.5], [-0.2], [0.8]], tf.float64)


def close(actual, expected, rtol=2e-7, atol=2e-8):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert int(tf.size(actual)) > 0
    tf.debugging.assert_near(actual, expected, rtol=rtol, atol=atol)


def fd(function, theta, h=1e-4):
    results = []
    for direction in tf.unstack(tf.eye(int(theta.shape[0]), dtype=theta.dtype)):
        terms = [function(theta + (k * h) * direction) for k in (-2., -1., 1., 2.)]
        results.append((terms[0] - 8 * terms[1] + 8 * terms[2] - terms[3]) / (12 * h))
    return tf.stack(results)


def test_scalar_one_observation_closed_form():
    model = parameterized_model(THETA, 1, 1)
    value, score, _, _, _ = make_gaussian_kernel(1, 1, 6, jit_compile=False)(OBSERVATIONS[:1], *model)
    A, _, H, _, mean, _, P, _, Q, _, R, _ = model
    prediction = float((A @ mean[:, None])[0, 0])
    variance = float((H @ (A @ P @ tf.transpose(A) + Q) @ tf.transpose(H) + R)[0, 0])
    residual = 0.5 - float(H[0, 0]) * prediction
    expected = -0.5 * (math.log(2 * math.pi * variance) + residual**2 / variance)
    assert abs(float(value) - expected) < 1e-12
    assert bool(tf.reduce_all(tf.math.is_finite(score)))


@pytest.mark.parametrize("dimension", [1, 2])
def test_all_six_parameter_dependencies_against_same_scalar(dimension):
    kernel = make_gaussian_kernel(dimension, 1, 6, jit_compile=False)
    def value(theta):
        return kernel(OBSERVATIONS, *parameterized_model(theta, dimension))[0]
    result = kernel(OBSERVATIONS, *parameterized_model(THETA, dimension))
    close(result[1], fd(value, THETA))
    assert bool(tf.reduce_all(tf.abs(result[1]) > 1e-4))
    assert float(result[4]) > 0
    assert kernel.experimental_get_tracing_count() == 1


def test_general_matrix_tangents_include_every_input():
    # Each physical input gets an arbitrary direction, independently of the
    # six-parameter convenience model. This checks the general oracle API.
    model = list(parameterized_model(THETA, 2, 1))
    for i in range(0, len(model), 2):
        direction = tf.reshape(tf.cast(tf.range(tf.size(model[i])) + 1, tf.float64), model[i].shape) * 0.07
        if len(direction.shape) == 2 and i in (6, 8, 10):
            direction = (direction + tf.transpose(direction)) / 2
        model[i + 1] = direction[None, ...]
    kernel = make_gaussian_kernel(2, 1, 1, jit_compile=False)
    actual = kernel(OBSERVATIONS, *model)[1][0]
    def value(t):
        shifted = [model[i] + t[0] * model[i+1][0] if i % 2 == 0 else model[i] for i in range(len(model))]
        return kernel(OBSERVATIONS, *shifted)[0]
    close(actual, fd(value, tf.zeros([1], tf.float64))[0])


def test_unscented_consumer_matches_kalman_affine_moments():
    arguments = (OBSERVATIONS, *parameterized_model(THETA, 2))
    exact = make_gaussian_kernel(2, 1, 6, jit_compile=False)(*arguments)
    unscented = make_gaussian_kernel(2, 1, 6, jit_compile=False, unscented=True)(*arguments)
    for expected, actual in zip(exact, unscented):
        close(actual, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("adapted", [False, True])
@pytest.mark.parametrize("resampling", [False, True])
def test_particle_initial_law_and_recursive_tangents(adapted, resampling):
    initial = tf.random.stateless_normal([24, 2], [7, 9], dtype=tf.float64)
    noises = tf.random.stateless_normal([3, 24, 2], [7, 10], dtype=tf.float64)
    kernel = make_particle_kernel(2, 1, 24, 3, jit_compile=True, adapted=adapted, resampling=resampling)
    args = (OBSERVATIONS, initial, noises)
    if resampling:
        args += (tf.random.stateless_uniform([3, 24], [7, 11], dtype=tf.float64),)
    value = lambda theta: kernel(theta, *args)[0]
    result = kernel(THETA, *args)
    assert result[0].shape == () and result[1].shape == (6,) and result[2].shape == ()
    close(result[1], fd(value, THETA))
    assert 1 <= float(result[2]) <= 24 + 1e-9
    assert kernel.experimental_get_tracing_count() == 1
