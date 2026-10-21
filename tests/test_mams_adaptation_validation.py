"""Validation lifecycle regressions; no stochastic performance claims."""
import json
import time

import pytest

from scripts import validate_mams_adaptation as validation


@pytest.mark.parametrize("failure,status", [
    ("pilot_budget", "pilot_budget_exhausted"),
    ("unavailable", "pilot_unavailable"),
    ("execution_budget", "budget_exhausted"),
])
def test_expected_failure_preserves_members_and_continues_targets(tmp_path, monkeypatch, failure, status):
    from bayesfilter.inference.hmc_mams_trajectory_proposals import MAMSPilotBudgetExceeded, MAMSPilotUnavailable
    from bayesfilter.runtime.execution_budget import ExecutionBudgetExceeded
    exceptions = dict(pilot_budget=MAMSPilotBudgetExceeded, unavailable=MAMSPilotUnavailable,
                      execution_budget=ExecutionBudgetExceeded)
    called = []

    def numerical(target, output, seed, deadline, progress, **kwargs):
        called.append(target)
        if target == "gaussian":
            progress["rows"].extend([
                dict(candidate_id="v1", tuning_state="verified", status="posterior_running"),
                dict(candidate_id="v2", tuning_state="verified", status="posterior_not_started"),
                dict(candidate_id="failed", tuning_state="rejected", status="not_verified")])
            raise exceptions[failure]("test allocation")
        progress["status"] = "completed"

    monkeypatch.setattr(validation, "_run_case", numerical)
    result = validation.execute_matrix(["gaussian", "mixture"], tmp_path, 1,
        time.monotonic()+30, use_xla=False, source_hashes={})
    assert called == ["gaussian", "mixture"]
    first, second = result["rows"]
    assert first["status"] == status
    assert first["rows"][0]["status"] == "posterior_interrupted_" + status
    assert first["rows"][1]["status"] == "posterior_not_started_" + status
    assert first["rows"][2]["status"] == "not_verified"
    assert second["status"] == "completed"
    assert not result["release_qualified"]


def test_shared_failure_stops_later_targets_but_preserves_denominator(tmp_path, monkeypatch):
    called = []

    def numerical(target, output, seed, deadline, progress, **kwargs):
        called.append(target)
        raise ValueError("correction identity mismatch")

    monkeypatch.setattr(validation, "_run_case", numerical)
    result = validation.execute_matrix(["gaussian", "mixture"], tmp_path, 1,
        time.monotonic()+30, use_xla=False, source_hashes={})
    assert called == ["gaussian"]
    assert result["status"] == "repair_required"
    assert len(result["rows"]) == 2
    assert result["rows"][0]["error_type"] == "ValueError"
    assert result["rows"][1]["status"] == "not_started_repair_required"


def test_hard_stop_recovers_created_and_unstarted_members_from_checkpoint(tmp_path):
    directory = tmp_path / "gaussian"
    tuning = directory / "search" / "tuning"
    tuning.mkdir(parents=True)
    validation.write_json(directory / "progress.json", dict(target="gaussian", status="tuning", rows=[]))
    validation.write_json(tuning / "tuning_checkpoint.json", dict(result=dict(
        candidates=[dict(candidate_id="a"), dict(candidate_id="b")],
        candidate_states=dict(a="verified", b="validating"),
        verified_candidate_ids=["a"], completion_status="partial_budget")))
    result = validation.finalize_interrupted_matrix(tmp_path, ["gaussian", "mixture"],
                                                   status="supervisor_budget_exhausted")
    first, second = result["rows"]
    assert first["verified_count"] == 1
    assert len(first["rows"]) == 2
    assert first["rows"][0]["status"] == "posterior_not_started_supervisor_budget_exhausted"
    assert first["rows"][1]["status"] == "not_verified"
    assert second["status"] == "not_started_supervisor_budget_exhausted"
    assert json.loads((tmp_path/"result.json").read_text()) == result


def test_expired_matrix_does_not_run_or_omit_any_target(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("expired allocation must not execute")
    result = validation.execute_matrix(["gaussian", "LGSSM", "NL-ACC"], tmp_path, 1,
        time.monotonic()-1, use_xla=False, source_hashes={}, case_runner=forbidden)
    assert len(result["rows"]) == 3
    assert all(r["status"] == "not_started_budget_exhausted" for r in result["rows"])


def test_residual_funnel_uses_requested_compile_mode_and_exact_transformed_score():
    import numpy as np  # Independent diagnostic only.
    import tensorflow as tf
    target, _, _, _ = validation.fixture("residual_funnel", False)
    assert not target._batch.function_spec.jit_compile
    points = tf.constant([[0., 1., -2.], [-1.3, -.4, .7], [2., 1., .2]], tf.float64)
    value, score = target.log_prob_and_grad(points)
    q = points.numpy()
    delta = .5*np.tanh(q[:, 0]/2)
    derivative = .25 / np.cosh(q[:, 0]/2)**2
    r2 = np.sum(q[:, 1:]**2, axis=1)*np.exp(2*delta)
    expected_score = np.column_stack((-q[:, 0]+derivative*(2-r2), -q[:, 1:]*np.exp(2*delta)[:, None]))
    np.testing.assert_allclose(value, -.5*(q[:, 0]**2+r2)+2*delta, atol=1e-14)
    np.testing.assert_allclose(score, expected_score, atol=1e-14)
