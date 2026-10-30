"""Focused contracts for the M13 endpoint-corrected mechanics stage."""
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_closure_program import PASS_CAPS, jobs
from bayesfilter.inference.neural_force_hmc import PositionOnlyForceEvaluation
import bayesfilter.testing.surrogate_hmc_m13_endpoint_mechanics as mechanics
from bayesfilter.testing.surrogate_hmc_m13_endpoint_mechanics import (
    PLAN, PRIOR_SD, endpoint_components, force_components, prior_log_density,
)


def test_master_registers_endpoint_mechanics_with_its_own_bounded_pass():
    selected = jobs("m13-endpoint-mechanics")
    assert {row["name"] for row in selected} == {"review", "m13-endpoint-mechanics"}
    worker = next(row for row in selected if row["name"] == "m13-endpoint-mechanics")
    assert worker["device"] == "gpu"
    assert worker["plan"] == PLAN
    assert PASS_CAPS["m13-endpoint-mechanics"] == {
        "cpu_worker_seconds": 800.0, "gpu_seconds": 1200.0,
    }


def test_force_is_gradient_of_negative_log_posterior_with_canonical_score():
    class FakeTarget:
        def compile(self, batch_size, **kwargs):
            assert batch_size == 1
            def evaluate(position):
                value = tf.reduce_sum(position * position, axis=-1)
                score = tf.broadcast_to(tf.constant([[3.0, -4.0]], tf.float32), tf.shape(position))
                valid = tf.ones([tf.shape(position)[0]], tf.bool)
                return value, score, {"program_valid": valid}
            return evaluate

    force, _ = force_components(FakeTarget())
    position = tf.constant([[0.25, -0.5]], tf.float32)
    result = force(position)
    expected = -tf.constant([[3.0, -4.0]], tf.float32) + position / tf.cast(tf.square(PRIOR_SD), tf.float32)
    assert isinstance(result, PositionOnlyForceEvaluation)
    tf.debugging.assert_near(result.force, expected, atol=1e-6, rtol=1e-6)
    assert bool(tf.reduce_all(result.force_valid))


def test_endpoint_plan_keeps_mechanics_and_posterior_claims_separate():
    text = Path(PLAN).read_text()
    assert "separately declared endpoint potential" in text
    assert "not a qualified R^2 posterior oracle" in text
    assert "no HMC tuning authority" in text
    tf.debugging.assert_near(
        prior_log_density(tf.constant([0.0, 0.0], tf.float64)),
        tf.constant(-2.0 * tf.math.log(tf.constant(PRIOR_SD, tf.float64) * tf.sqrt(tf.constant(2.0 * 3.141592653589793, tf.float64))), tf.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_endpoint_adapter_returns_one_potential_per_position_row(monkeypatch):
    def fake_reference(_fixture, _particles, _replicates, full_support):
        assert full_support is True

        def evaluate(theta, _seed):
            return {
                "log_likelihood": tf.constant([[-1.0]], tf.float64),
                "valid": tf.constant([True]),
            }

        return evaluate

    monkeypatch.setattr(mechanics, "trajectory_importance_reference", fake_reference)
    endpoint, _ = endpoint_components(object())
    result = endpoint(tf.constant([[0.0, 0.0]], tf.float32))
    assert result.potential.shape == (1,)
    assert result.status["value_valid"].shape == (1,)
    assert bool(tf.reduce_all(result.status["value_valid"]))
