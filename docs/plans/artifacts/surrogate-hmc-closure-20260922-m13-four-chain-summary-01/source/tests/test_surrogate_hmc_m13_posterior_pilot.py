"""Independent mathematical checks and master call-chain for the M13 pilot."""
import math

import pytest
import tensorflow as tf

from bayesfilter.testing import surrogate_hmc_m13_posterior_pilot as pilot


def test_constant_likelihood_recovers_independent_prior_and_evidence():
    axis, integrate = pilot.make_integrator(65, 5*pilot.PRIOR_SD, 4)
    result = integrate(tf.zeros([65, 65, 4], tf.float64))
    metrics = result["metrics"].numpy().tolist()
    assert bool(result["finite"])
    assert metrics[:2] == pytest.approx([0., 0.], abs=1e-12)
    assert metrics[2:4] == pytest.approx([pilot.PRIOR_SD]*2, abs=1e-5)
    assert metrics[4] == pytest.approx(0., abs=1e-12)
    assert float(result["log_evidence"]) == pytest.approx(2*math.log(math.erf(5/math.sqrt(2))), abs=1e-8)
    assert metrics[5:7] == pytest.approx([math.exp(pilot.PRIOR_SD**2/2)]*2, abs=3e-5)
    assert tf.reduce_max(result["mcse"]) == 0


def test_tilted_gaussian_recovers_analytic_posterior_and_constant_shift_invariance():
    axis, integrate = pilot.make_integrator(65, 6*pilot.PRIOR_SD, 4)
    q, r = tf.meshgrid(axis, axis, indexing="ij")
    logits = (q*.4-r*.7)[..., None]+tf.zeros([1, 1, 4], tf.float64)
    expected = [.4*pilot.PRIOR_SD**2, -.7*pilot.PRIOR_SD**2]
    result, shifted = integrate(logits), integrate(logits+100.)
    assert result["metrics"][:2].numpy().tolist() == pytest.approx(expected, abs=1e-6)
    assert result["metrics"][2:4].numpy().tolist() == pytest.approx([pilot.PRIOR_SD]*2, abs=1e-6)
    tf.debugging.assert_near(result["metrics"], shifted["metrics"], atol=1e-12)
    assert float(shifted["log_evidence"]-result["log_evidence"]) == pytest.approx(100.)


def test_missing_or_invalid_node_refuses_posterior_summary():
    logs = tf.zeros([9, 9, 4], tf.float64)
    valid = tf.ones([9, 9, 4], tf.bool)
    for bad_logs, bad_valid in ((tf.tensor_scatter_nd_update(logs, [[4, 4, 0]], [float("nan")]), valid),
        (logs, tf.tensor_scatter_nd_update(valid, [[4, 4, 0]], [False]))):
        result = pilot.summarize_grid(bad_logs, bad_valid)
        assert result["status"] == "INVALID_GRID_NO_POSTERIOR_SUMMARY"
        assert not result["posterior_reference_qualified"]
        assert "fine" not in result


def test_valid_grid_still_cannot_qualify_posterior_or_endpoint():
    logs = tf.zeros([17, 17, 4], tf.float64)
    result = pilot.summarize_grid(logs, tf.ones_like(logs, tf.bool))
    assert result["complete_valid_grid"]
    assert not result["posterior_reference_qualified"]
    assert not result["endpoint_qualified"]
    assert result["tail_bound"] == "NOT_ESTABLISHED"
    assert result["paired_grid_difference_mcse"] == [0.]*len(pilot.NAMES)


def test_pilot_master_uses_existing_pool_and_review():
    from bayesfilter.inference import surrogate_hmc_closure_program as master
    stages = master.jobs("m13-posterior-pilot")
    assert [s["name"] for s in stages] == ["review", "m13-posterior-pilot"]
    assert stages[1]["plan"] == pilot.PLAN
    assert stages[1]["cpu"]+stages[0]["cpu"] <= 554.930702
    assert stages[1]["timeout"] <= 581.189720343
    assert "tests/test_surrogate_hmc_m13_posterior_pilot.py" in stages[0]["command"]


def test_worker_reaches_posterior_pilot(monkeypatch, tmp_path):
    from bayesfilter.testing import surrogate_hmc_closure_diagnostics as worker
    calls = []
    monkeypatch.setattr(pilot, "posterior_pilot_stage", lambda path: calls.append(path) or
                        dict(status="mechanics", qualified=False, inference_launched=False))
    output = tmp_path/"result.json"
    worker.main(["--stage", "m13-posterior-pilot",
        "--device", "cpu", "--output", str(output)])
    assert calls == [output]

@pytest.mark.parametrize("valid", [True, False])
def test_stage_calls_full_support_reference_and_never_promotes(monkeypatch, tmp_path, valid):
    calls = []
    def factory(fixture, particles, replicates, *, full_support=False):
        assert full_support and particles == 4096 and replicates == 8
        assert fixture.horizon == 20
        def evaluate(theta, seed):
            calls.append(theta.numpy().tolist())
            return dict(log_likelihood=tf.zeros([8], tf.float64),
                valid=tf.fill([8], valid), minimum_ess=tf.ones([8], tf.float64)*100,
                proposal_maximum_normalized_weight=tf.ones([8], tf.float64)*.01,
                proposal_first_diagnostics=tf.zeros([10], tf.float64),
                proposal_full_diagnostics=tf.zeros([10], tf.float64))
        return evaluate
    monkeypatch.setattr(pilot, "ORDER", 9)
    monkeypatch.setattr(pilot, "trajectory_importance_reference", factory)
    report = pilot.posterior_pilot_stage(tmp_path/"report.json")
    assert len(calls) == 81
    assert report["valid_node_count"] == (81 if valid else 0)
    assert not report["posterior_reference_qualified"] and not report["endpoint_qualified"]
    assert not report["inference_launched"]
    assert report["summary"]["complete_valid_grid"] is valid
