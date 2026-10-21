"""Master wiring, budget isolation, and actual trajectory reference call chain."""
import json
import math
from dataclasses import replace

import pytest
import tensorflow as tf

from bayesfilter.testing import surrogate_hmc_m13_reference_extension as extension


def test_extension_has_separate_bounded_reservation_and_review():
    from bayesfilter.inference import surrogate_hmc_closure_program as master
    stages = master.jobs("m13-reference-extension")
    assert [s["name"] for s in stages] == ["review", "m13-reference-extension"]
    assert stages[1]["plan"] == extension.PLAN
    assert stages[1]["device"] == "gpu"
    assert "tests/test_surrogate_hmc_m13_wrapped_reference.py" in stages[0]["command"]
    receipts = [dict(path=f"/tmp/surrogate-hmc-closure-20260922-{name}-01/accounting.json",
                     charged_this_run=dict(cpu_worker_seconds=cpu, gpu_seconds=gpu))
                for name, cpu, gpu in (("m13-reference", 100., 200.),
                                       ("m13-reference-extension", 20., 30.))]
    assert master.remaining_pass_reservation(receipts, "m13-reference-extension") == {
        "cpu_worker_seconds": 580., "gpu_seconds": 570.}
    assert master.remaining_pass_reservation(receipts, "m13-reference") == {
        "cpu_worker_seconds": 2300., "gpu_seconds": 3400.}
    with pytest.raises(ValueError, match="separate"):
        master.jobs("m13-reference-extension,m13-controls")


def test_worker_dispatch_and_cpu_manifest(tmp_path, monkeypatch):
    from bayesfilter.testing import surrogate_hmc_closure_diagnostics as worker
    calls = []
    def run(output):
        calls.append(output)
        return dict(status="WIRING_ONLY", qualified=False)
    monkeypatch.setattr(extension, "reference_extension_stage", run)
    path = tmp_path/"report.json"
    worker.main(["--stage", "m13-reference-extension", "--device", "cpu", "--output", str(path)])
    result = json.loads(path.read_text())
    assert calls == [path] and result["gpu_intentionally_hidden"]


@pytest.mark.parametrize("valid", [True, False])
def test_stage_never_promotes_fixed_point_evidence_to_posterior(tmp_path, monkeypatch, valid):
    calls = []
    def factory(fixture, particles, replicates, *, full_support=False):
        assert particles == 4096 and replicates == 8
        def run(theta, seed):
            calls.append((full_support, theta.numpy().tolist(), seed.numpy().tolist()))
            return dict(log_likelihood=tf.zeros([8], tf.float64), valid=tf.fill([8], valid),
                minimum_ess=tf.ones([8], tf.float64)*100,
                proposal_maximum_normalized_weight=tf.ones([8], tf.float64)*.01,
                proposal_first_diagnostics=tf.zeros([10], tf.float64),
                proposal_full_diagnostics=tf.zeros([10], tf.float64))
        return run
    monkeypatch.setattr(extension, "trajectory_importance_reference", factory)
    report = extension.reference_extension_stage(tmp_path/"report.json")
    assert len(calls) == 7
    assert report["all_points_valid"] == valid
    assert not report["qualified"] and not report["posterior_reference_qualified"]
    assert not report["likelihood_reference_qualified"] and not report["inference_launched"]
    assert report["prior_support"] == "R^2"
    assert calls[0][1:] == calls[1][1:] and calls[2][1:] == calls[3][1:]
    assert report["status"].endswith("COMPLETED" if valid else "REPAIR_REQUIRED")


def test_actual_trajectory_graph_calls_extended_density_and_angular_score(monkeypatch):
    from experiments.dpf_implementation.tf_tfp.fixtures.range_bearing_tf import build_range_bearing_fixture_tf
    from bayesfilter.testing import surrogate_hmc_m13_wrapped_reference as wrapped
    from bayesfilter.testing import surrogate_hmc_m13_reference as old
    from bayesfilter.testing.surrogate_hmc_m13_trajectory_reference import trajectory_importance_reference
    fixture = build_range_bearing_fixture_tf()
    fixture = replace(fixture, horizon=2, observations=fixture.observations[:2])
    calls = {"density": 0, "score": 0}
    density, angular = wrapped.observation_log_density, wrapped.angular_terms
    def checked_density(*args):
        calls["density"] += 1
        return density(*args)
    def checked_angular(*args):
        calls["score"] += 1
        return angular(*args)
    def forbidden(*args):
        raise AssertionError("extended trajectory reached restricted density")
    monkeypatch.setattr(wrapped, "observation_log_density", checked_density)
    monkeypatch.setattr(wrapped, "angular_terms", checked_angular)
    monkeypatch.setattr(old, "observation_log_density", forbidden)
    evaluate = trajectory_importance_reference(fixture, 8, 2, full_support=True)
    result = evaluate(tf.constant([0., math.log(100.)], tf.float64),
                      tf.constant([20260924, 71], tf.int32))
    assert calls["density"] > 0 and calls["score"] > calls["density"]
    tf.debugging.assert_all_finite(result["log_likelihood"], "extended T2 reference")
    tf.debugging.assert_equal(result["valid"], True)
