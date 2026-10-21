"""Complete LG2 cost accounting and physical retained-diagnostic checks."""
import json
import math
from pathlib import Path

import pytest

from scripts.qualify_surrogate_hmc_linear_endpoint import complete_forecast, remaining


def timings(costs):
    return [dict(cpu_worker_seconds=cpu, wall_seconds=wall,
                 compilation_included=i == 0, healthy=True)
            for i, (cpu, wall) in enumerate(costs)]


def test_complete_forecast_includes_nomination_and_whole_diagnostic_chunks():
    transitions = timings([(99., 99.), (1., 4.), (3., 2.), (2., 1.), (2., 3.)])
    diagnostics = timings([(99., 99.), (.1, .4), (.3, .2), (.2, .1), (.2, .3)])
    forecast = complete_forecast(transitions, diagnostics)
    assert forecast["required"]["cpu_worker_seconds"] == pytest.approx(1.5*(3586*3+8*.3+330))
    assert forecast["required"]["wall_seconds"] == pytest.approx(1.5*(3586*4+8*.4+330))
    assert forecast["admission"] is False


def test_measured_compilation_overhead_cannot_be_hidden_by_fixed_allowance():
    transitions = timings([(1000., 500.), (3., 4.), (3., 4.), (3., 4.), (3., 4.)])
    diagnostics = timings([(300., 50.), (.3, .4), (.3, .4), (.3, .4), (.3, .4)])
    forecast = complete_forecast(transitions, diagnostics)
    assert forecast["compilation_allowance_seconds"] == pytest.approx(
        dict(cpu_worker_seconds=1296.7, wall_seconds=545.6))
    assert forecast["required"]["cpu_worker_seconds"] == pytest.approx(1.5*(3586*3+8*.3+1296.7+90))
    assert forecast["required"]["wall_seconds"] == pytest.approx(1.5*(3586*4+8*.4+545.6+90))


@pytest.mark.parametrize("failure", ["missing", "unhealthy", "zero", "nan"])
def test_incomplete_or_invalid_costs_cannot_qualify(failure):
    rows = timings([(1., 1.)]*5)
    if failure == "missing":
        rows.pop()
    elif failure == "unhealthy":
        rows[-1]["healthy"] = False
    else:
        rows[-1]["wall_seconds"] = 0. if failure == "zero" else math.nan
    with pytest.raises(ValueError):
        complete_forecast(rows, rows)


def test_family_spending_includes_tests_and_mixed_parent_overhead(tmp_path):
    base = tmp_path/"docs/plans/artifacts"
    for name, payload in {
        "lg2-tests": dict(charged_this_run=dict(cpu_worker_seconds=1500., gpu_seconds=0.)),
        "mixed": dict(charged_this_run=dict(cpu_worker_seconds=60., gpu_seconds=40.), jobs=[
            dict(command=["python", "scripts/calibrate_surrogate_hmc_linear_controls.py"],
                 cpu_worker_seconds=10., gpu_seconds=20.),
            dict(command=["python", "unrelated.py"], cpu_worker_seconds=40., gpu_seconds=20.)]),
    }.items():
        path = base/f"surrogate-hmc-all-models-test-{name}-01/accounting.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload))
    assert remaining(tmp_path) == (280., 450.)
    path = base/"surrogate-hmc-all-models-test-lg2-endpoint-01/accounting.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(charged_this_run=dict(cpu_worker_seconds=1., gpu_seconds=1.))))
    assert remaining(tmp_path) == (0., 0.)


def test_retained_functionals_keep_chart_units_axes_and_kalman_predictions():
    import tensorflow as tf
    from bayesfilter.testing.surrogate_hmc_linear_inference_metrics import make_functionals
    from bayesfilter.testing.surrogate_hmc_linear_reference import make_linear_reference

    root = Path(__file__).resolve().parents[1]
    base = root/"docs/plans/artifacts/surrogate-hmc-all-models-20260920-lg2-reference-01/reference/claim"
    claim = json.loads((base/"data.json").read_text())
    reference = json.loads((base/"reference.json").read_text())
    points = tf.constant([[-2., -1.], [.5, 1.], [2., -.5], [-.5, 2.]], tf.float64)
    samples = tf.tile(points[None], [250, 1, 1])
    physical, values, valid = make_functionals(claim, reference)(samples)
    assert physical.shape == (250, 4, 2) and values.shape == (250, 4, 13)
    assert bool(tf.reduce_all(valid))
    expected = tf.constant([.5, -1.], tf.float64)+tf.constant([.7, 2.], tf.float64)*tf.sigmoid(points)
    tf.debugging.assert_near(physical, tf.tile(expected[None], [250, 1, 1]), atol=1e-14, rtol=1e-14)
    means, scales, cuts = [tf.constant(reference[key], tf.float64) for key in
                          ("posterior_mean", "posterior_sd", "marginal_quantiles")]
    standardized = (expected-means)/scales
    a, b = standardized[:, 0], standardized[:, 1]
    moments = tf.stack([a, b, a*a, b*b, a*b], axis=1)
    cdfs = tf.cast(expected[:, :, None] <= cuts[None], tf.float64)
    tf.debugging.assert_near(values[:, :, :5], tf.tile(moments[None], [250, 1, 1]), atol=1e-12, rtol=1e-12)
    tf.debugging.assert_equal(values[:, :, 5:11], tf.tile(tf.reshape(cdfs, [1, 4, 6]), [250, 1, 1]))
    _, predictive = make_linear_reference("LG2").physical_statistics(
        expected, tf.cast(tf.constant(claim["observations"], tf.float32), tf.float64))
    tf.debugging.assert_near(values[:, :, -2:], tf.tile(predictive[None], [250, 1, 1]), atol=1e-11, rtol=1e-11)
