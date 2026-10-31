"""Real dual-parameter target mechanics; no posterior or tuning admission."""
from pathlib import Path
import runpy

import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import canonical_batch_fused_value_score
from bayesfilter.inference.ledh_dual_parameter_target import (
    DualParameterLEDHTarget, make_dual_parameter_target,
)

_fixture = runpy.run_path(str(Path(__file__).parents[1] / 'integration' /
                             'test_ledh_failure_contract.py'))['fixture']


@pytest.fixture
def target_fixture():
    model, inputs, controls = _fixture(horizon=2)
    controls = dict(controls)
    ridge = controls.pop('reset_ridge')
    target = make_dual_parameter_target(
        model, *inputs, damping_ratio=100., base_reset_ridge=ridge,
        base_lm_damping=.01, **controls)
    return target, tf.constant([.6], tf.float64)


def test_dual_parameter_target_value_is_exact(target_fixture):
    target, theta = target_fixture
    value, _, status = target.value_score_and_status(theta)
    expected, _, _ = canonical_batch_fused_value_score(
        target.model, theta[None], tf.ones([1, 1, 1], theta.dtype),
        target.initial_states, target.initial_covariances, target.noises,
        target.observations, **target.exact_params)
    assert bool(status['value_valid'])
    tf.debugging.assert_equal(value, expected[0])
    tf.debugging.assert_equal(target(theta), expected[0])
    tf.debugging.assert_equal(target.value_only(theta), expected[0])


def test_dual_parameter_target_score_is_biased(target_fixture):
    target, theta = target_fixture
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = target(theta)
    actual = tape.gradient(value, theta)
    _, expected, status = canonical_batch_fused_value_score(
        target.model, theta[None], tf.ones([1, 1, 1], theta.dtype),
        target.initial_states, target.initial_covariances, target.noises,
        target.observations, **target.biased_params)
    assert bool(status['force_valid'][0])
    tf.debugging.assert_equal(actual, expected[0])
    tf.debugging.assert_equal(target.score_only(theta), expected[0])


def test_dual_parameter_target_determinism(target_fixture):
    target, theta = target_fixture
    first = target.value_score_and_status(theta)
    second = target.value_score_and_status(theta)
    for a, b in zip(tf.nest.flatten(first), tf.nest.flatten(second)):
        tf.debugging.assert_equal(a, b)


def test_dual_parameter_target_shape_consistency(target_fixture):
    target, theta = target_fixture
    value, score, _ = target.value_score_and_status(theta)
    assert value.shape == ()
    assert score.shape == theta.shape
    # The custom-gradient HMC callable is scalar; diagnostic methods accept batches.
    batch = tf.stack([theta, theta + .1])
    assert target.value_only(batch).shape == (2,)
    assert target.score_only(batch).shape == (2, 1)
    assert target.score_only(batch, exact=True).shape == (2, 1)


@pytest.mark.parametrize('ratio', [1., 10., 100., 1000.])
def test_make_dual_parameter_target_factory(target_fixture, ratio):
    reference, theta = target_fixture
    controls = dict(reference.exact_params)
    controls.pop('reset_ridge')
    controls.pop('correction_lm_damping')
    target = make_dual_parameter_target(
        reference.model, reference.initial_states, reference.initial_covariances,
        reference.noises, reference.observations, damping_ratio=ratio,
        base_reset_ridge=1e-5, base_lm_damping=.01, **controls)
    assert target.exact_params['reset_ridge'] == 1e-5
    assert target.exact_params['correction_lm_damping'] == .01
    assert target.biased_params['reset_ridge'] == 1e-5 * ratio
    assert target.biased_params['correction_lm_damping'] == .01 * ratio
    assert target.biased_params['pairwise_steps'] > 0
    assert target.biased_params['coordinate_cap'] > 0
    value, score, status = target.value_score_and_status(theta)
    assert bool(status['program_valid'])
    tf.debugging.assert_all_finite(value, 'factory value')
    tf.debugging.assert_all_finite(score, 'factory force')


def test_dual_parameter_requires_frozen_noises(target_fixture):
    target, _ = target_fixture
    with pytest.raises(TypeError, match='noises must be a Tensor'):
        DualParameterLEDHTarget(
            target.model, target.initial_states, target.initial_covariances,
            None, target.observations, exact_params=target.exact_params,
            biased_params=target.biased_params)


def test_dual_parameter_requires_matching_keys(target_fixture):
    target, _ = target_fixture
    with pytest.raises(ValueError, match='Parameter key mismatch'):
        DualParameterLEDHTarget(
            target.model, target.initial_states, target.initial_covariances,
            target.noises, target.observations, exact_params=target.exact_params,
            biased_params={'reset_ridge': .001})
