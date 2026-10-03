"""Real-transition and independent-energy checks for composed-kernel validation."""
from dataclasses import replace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.testing.inference_validation.engines import invariance, mechanics
from bayesfilter.testing.inference_validation.procedures import FrozenTransition
from bayesfilter.testing.inference_validation.targets import ValidationTarget


@pytest.mark.parametrize("epsilon", [.3, .6])
def test_independent_energy_oracle_detects_reversed_ratio_at_weak_settings(design, tmp_path, epsilon):
    baseline = mechanics.run(design(step_size=epsilon), tmp_path/"baseline")
    noop = mechanics.run(design(step_size=epsilon, control="noop"), tmp_path/"noop")
    defective = mechanics.run(design(step_size=epsilon, control="wrong_energy"), tmp_path/"defective")
    assert baseline["finding"] == noop["finding"] == "mechanics_passed"
    assert baseline["metropolis_log_ratio_observed"] == noop["metropolis_log_ratio_observed"]
    assert defective["finding"] == "mechanics_discrepancy"
    assert not defective["metropolis_log_ratio_passed"]
    np.testing.assert_allclose(defective["metropolis_log_ratio_observed"],
                               -np.array(defective["metropolis_log_ratio_expected"]), atol=1.e-12)
    assert defective["metropolis_state_selection_passed"]


@pytest.mark.parametrize("control", ["baseline", "wrong_energy", "identity", "two_cycle"])
def test_power_composes_complete_transitions_with_the_recorded_substep_seeds(control):
    target = ValidationTarget("gaussian", jit_compile=False)
    kernel = FrozenTransition(target, chains=4, step_size=.6, leapfrog_steps=5,
                              control=control, jit_compile=False)
    assert kernel.powered_step(1) is kernel.step
    q = tf.constant([[-1., .3], [.4, .8], [-.2, -.7], [1., 2.]], tf.float64)
    seed = tf.constant([101, 202], tf.int32)
    expected = q
    for index in range(8):
        expected, expected_ratio = kernel.step(expected, tf.random.experimental.stateless_fold_in(seed, index))
    powered = kernel.powered_step(8)
    actual, actual_ratio = powered(q, seed)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_ratio, expected_ratio)
    assert kernel.powered_step(8) is powered
    checked_state, checked_ratio, healthy = kernel.powered_step(8, with_health=True)(q, seed)
    np.testing.assert_array_equal(checked_state, actual)
    np.testing.assert_array_equal(checked_ratio, actual_ratio)
    assert bool(healthy)
    checked_one, ratio_one, healthy_one = kernel.powered_step(1, with_health=True)(q, seed)
    one, original_ratio = kernel.step(q, seed)
    np.testing.assert_array_equal(checked_one, one)
    np.testing.assert_array_equal(ratio_one, original_ratio)
    assert bool(healthy_one)
    powered(q + .1, tf.constant([102, 203], tf.int32))
    assert powered.experimental_get_tracing_count() == 1


def test_power_one_preserves_the_original_invariance_experiment(design, tmp_path):
    original = design("invariance", replications=16)
    explicit = replace(original, options={"kernel_power": 1})
    a = invariance.run(original, tmp_path/"original")
    b = invariance.run(explicit, tmp_path/"explicit")
    assert a == b


def test_even_power_preserves_the_stationary_but_immobile_cycle_counterexample(design, tmp_path):
    result = invariance.run(design("invariance", control="two_cycle", replications=32,
                                  options={"kernel_power": 8}), tmp_path)
    assert result["fraction_moved"] == 0.
    assert result["kernel_power"] == 8
    assert result["native_transitions_per_arm"] == 56
    assert not result["mixing_established"]


@pytest.mark.parametrize("power", [0, -1, True, 1.5, 2**31])
def test_invalid_kernel_powers_fail_at_design_time(design, power):
    with pytest.raises(ValueError, match="kernel_power"):
        design("invariance", options={"kernel_power": power})


def test_kernel_power_cannot_silently_thin_a_posterior_pipeline(design):
    with pytest.raises(ValueError, match="invariance only"):
        design("accuracy", route="ordinary", options={"kernel_power": 8})


def test_composition_health_preserves_a_nonfinite_intermediate_ratio():
    target = ValidationTarget("gaussian", jit_compile=False)
    kernel = FrozenTransition(target, chains=4, step_size=.6, leapfrog_steps=5,
                              jit_compile=False)
    @tf.function(input_signature=kernel.step.input_signature, autograph=False)
    def defective_step(q, seed):
        ratio = tf.where(q[:, 0] == 0., tf.constant(float("nan"), tf.float64),
                         tf.constant(0., tf.float64))
        return q + 1., ratio
    kernel.step = defective_step
    state, last_ratio, healthy = kernel.powered_step(8, with_health=True)(
        tf.zeros([4, 2], tf.float64), tf.constant([101, 202], tf.int32))
    assert bool(tf.reduce_all(tf.math.is_finite(state)))
    assert bool(tf.reduce_all(tf.math.is_finite(last_ratio)))
    assert not bool(healthy)


def test_invariance_honors_an_expired_deadline(design, tmp_path):
    with pytest.raises(TimeoutError, match="deadline"):
        invariance.run(design("invariance"), tmp_path, deadline=0.)


def test_invariance_does_not_count_nonfinite_substeps_as_detection(design, tmp_path, monkeypatch):
    def bad_power(self, power, *, with_health=False):
        return lambda q, seed: (q, tf.zeros([len(q)], tf.float64), tf.constant(False))
    monkeypatch.setattr(FrozenTransition, "powered_step", bad_power)
    with pytest.raises(FloatingPointError, match="nonfinite invariance evidence"):
        invariance.run(design("invariance"), tmp_path)
    assert not (tmp_path/"invariance.json").exists()
