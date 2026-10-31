"""Endpoint domain-failure semantics and complete-cost regressions."""
from types import SimpleNamespace
import json

import pytest
import tensorflow as tf

from bayesfilter.inference.neural_force_hmc import FrozenPositionOnlyForce, FrozenTargetPotential
from bayesfilter.testing import surrogate_hmc_sv_inference_consumer as candidate
from bayesfilter.testing.surrogate_hmc_sv_consumer import SVConsumer
from scripts.qualify_surrogate_hmc_sv_endpoint import complete_forecast, remaining


def costs():
    return [dict(compilation_included=i == 0, cpu_worker_seconds=float(i+1),
                 wall_seconds=float(i+2), healthy=True) for i in range(5)]


def test_forecast_counts_complete_work_and_worst_steady_cost():
    result = complete_forecast(costs(), costs())
    assert result["transitions_per_four_chain_batch"] == 3586
    assert result["required"]["cpu_worker_seconds"] == 1.5*(3586*5+256*5+240+90)
    assert result["required"]["wall_seconds"] == 1.5*(3586*6+256*6+240+90)
    assert result["admission"] is False


def test_failed_attempts_share_stage_reservation_and_are_not_free_retries(tmp_path):
    base = tmp_path/"docs/plans/artifacts"
    def charge(name, cpu, gpu):
        path = base/name
        path.mkdir(parents=True)
        (path/"accounting.json").write_text(json.dumps(dict(exit_code=2,
            charged_this_run=dict(cpu_worker_seconds=cpu, gpu_seconds=gpu))))
    charge("surrogate-hmc-all-models-test-sv20-endpoint-01", 100., 150.)
    assert remaining(tmp_path) == (350., 450.)
    charge("surrogate-hmc-all-models-test-sv20-endpoint-02", 10., 20.)
    assert remaining(tmp_path) == (0., 0.)


@pytest.mark.parametrize("field,value", [("healthy", False), ("cpu_worker_seconds", 0.),
    ("wall_seconds", float("nan")), ("compilation_included", True)])
@pytest.mark.parametrize("which", [0, 1])
def test_forecast_rejects_invalid_transition_or_predictive_cost(field, value, which):
    rows = [costs(), costs()]
    rows[which][3][field] = value
    with pytest.raises(ValueError):
        complete_forecast(*rows)


def test_guard_rejects_finite_excess_loss_and_nonfinite_score():
    valid = candidate.reference_validity(tf.zeros([4], tf.float64),
        tf.constant([[0., 0.], [0., 0.], [float("nan"), 0.], [0., 0.]], tf.float64),
        tf.constant([0., 1.01e-9, 0., float("inf")], tf.float64))
    assert valid.numpy().tolist() == [True, False, False, False]


def test_endpoint_guard_flags_numerical_failure_without_false_support(monkeypatch):
    """Execute the actual wrapper with a finite but inaccurate reference row."""
    def reference(theta, observations):
        return dict(log_likelihood=tf.reduce_sum(theta, axis=1), score=tf.ones_like(theta),
                    maximum_predicted_domain_loss=tf.constant([0., 2e-9], tf.float64))
    force = FrozenPositionOnlyForce(lambda z: z, identity="canonical-field-test")
    base = SVConsumer(fixture=object(), reference=SimpleNamespace(physical_value_score=reference),
        force=force, endpoint=FrozenTargetPotential(lambda z: z, identity="base-endpoint-test"),
        exact_force=force, evaluate_force=object(), evaluate_reference=object(), manifest={})
    monkeypatch.setattr(candidate, "make_sv_consumer", lambda *args, **kwargs: base)
    consumer = candidate.make_sv_inference_consumer(dict(partition="claim", attempt=1,
        replicate=0, raw_observations=[.3]*20), controls={}, batch_size=2)
    # Explicit CPU reference exception: exercise this graph's Python body to
    # avoid XLA compilation in a small injected-failure unit test.
    result = consumer.evaluate_reference_diagnostics.python_function(tf.zeros([2, 2], tf.float64))
    # Endpoint closure calls the compiled function; disable JIT only within the
    # test's global eager-functions context and restore it in finally.
    previous = tf.config.functions_run_eagerly()
    try:
        tf.config.run_functions_eagerly(True)
        endpoint = consumer.endpoint.function(tf.zeros([2, 2], tf.float64))
        exact = consumer.exact_force.function(tf.zeros([2, 2], tf.float64))
    finally:
        tf.config.run_functions_eagerly(previous)
    assert result[3].numpy().tolist() == [True, False]
    assert endpoint.status["numerical_failure"].numpy().tolist() == [False, True]
    assert endpoint.status["outside_declared_support"].numpy().tolist() == [False, False]
    assert bool(tf.math.is_finite(endpoint.potential[0]))
    assert bool(tf.math.is_nan(endpoint.potential[1]))
    assert exact.force_valid.numpy().tolist() == [True, False]
    assert consumer.force is base.force
    assert consumer.evaluate_force is base.evaluate_force


@pytest.mark.parametrize("change", [dict(partition="calibration"), dict(attempt=2), dict(replicate=1)])
def test_claim_scope_required_before_consumer_construction(change):
    data = dict(partition="claim", attempt=1, replicate=0)
    data.update(change)
    with pytest.raises(ValueError, match="original SV20"):
        candidate.make_sv_inference_consumer(data, controls={})
