import numpy as np
import pytest
from bayesfilter.testing.inference_validation.designs import ScenarioSpec

from bayesfilter.testing.inference_validation.engines.acceptance import acceptance_marks, run


def test_beta_refresh_preserves_stationary_mean_and_creates_dependence():
    # Independent generative law, distinct from the tested acceptance estimator.
    draws = acceptance_marks(np.random.default_rng(392), 20000, .7, 20., .9)
    assert abs(draws.mean() - .7) < .01
    assert np.corrcoef(draws[1:,0], draws[:-1,0])[0,1] > .85


def test_screen_calibration_runs_real_stages_and_keeps_missing_denominator(design, tmp_path):
    d = design("acceptance", route="controller", replications=2,
        options={"acceptance_mean":.7, "acceptance_concentration":2000.,
                 "acceptance_persistence":0., "evidence_rungs":[1,2,4]})
    result = run(d, tmp_path)
    assert result["completed"] == result["verified"] == 2
    assert all([look["stage"] for look in r["looks"]] == ["measurement", "verification"] for r in result["rows"])
    assert not result["numerical_hmc"]
    incomplete = run(d, tmp_path/"expired", deadline=0.001)
    assert incomplete["completed"] == 0 and incomplete["planned"] == 2
    assert incomplete["finding"] == "incomplete"
    with pytest.raises(ValueError, match="declare finite"):
        design("acceptance", route="controller")


def test_actual_hmc_acceptance_uses_stationary_gaussian_transitions(design, tmp_path):
    from bayesfilter.testing.inference_validation.engines.acceptance import run
    actual = design("acceptance", target="gaussian", route="frozen", replications=2,
                    scenario=ScenarioSpec("gaussian", "frozen", start="reference"),
                    draws=64, step_size=.3, leapfrog_steps=5)
    result = run(actual, tmp_path)
    assert result["finding"] == "actual_hmc_acceptance_measured"
    assert result["assessed"] == 2
    assert result["numerical_hmc"] and not result["automatic_preparation"]
    assert all(row["evidence"]["evidence_validity"] == "valid" for row in result["rows"])


def test_actual_hmc_acceptance_rejects_non_gaussian_and_keeps_synthetic_contract(design):
    with pytest.raises(ValueError, match="Gaussian target"):
        design("acceptance", target="funnel", route="frozen")
    with pytest.raises(ValueError, match="CPU diagnostic controller"):
        design("acceptance", target="gaussian", route="controller", device="gpu")


def test_prepared_calibration_calls_public_tuner_and_issues_fresh_receipts(design, tmp_path, monkeypatch):
    import bayesfilter.inference as inference
    calls = []
    public = inference.tune_hmc_kernel
    def capture(**kwargs):
        calls.append(kwargs)
        return public(**kwargs)
    monkeypatch.setattr(inference, "tune_hmc_kernel", capture)
    d = design("acceptance", route="prepared", replications=2, measurement_draws=64,
        step_size=1.5, leapfrog_steps=5, options={"evidence_rungs":[1,2,4], "reference_anchors":256})
    result = run(d, tmp_path)
    assert len(calls) == 2
    assert result["completed"] == result["planned"] == 2
    assert result["reference"]["independent_energy_check_passed"]
    assert not result["reference"]["reference_used_for_tuning"]
    assert result["public_tuner"] == "tune_hmc_kernel"
    for kwargs in calls:
        np.testing.assert_array_equal(kwargs["initial_position"], [[-1.,-1.],[-.3,-.3],[.4,.4],[1.,1.]])
        assert not kwargs["config"].pilot_enabled
        assert kwargs["config"].max_candidates == 1
    assert calls[0]["candidate_set_adapter"].scope.search_id != calls[1]["candidate_set_adapter"].scope.search_id
    assert all(r["inventory"]["finding"] == "inventory_passed" for r in result["rows"])


def test_acceptance_routes_enforce_initialization_and_reference_contract(design):
    with pytest.raises(ValueError, match="reference starts"):
        design("acceptance", route="frozen")
    with pytest.raises(ValueError, match="reference_anchors"):
        design("acceptance", route="prepared", options={"evidence_rungs":[1,2,4]})
    with pytest.raises(ValueError, match="must not supply tuning starts"):
        design("acceptance", scenario=ScenarioSpec("gaussian", "prepared", start="reference"),
               options={"evidence_rungs":[1,2,4], "reference_anchors":256})


def test_public_acceptance_reference_vetoes_an_incorrect_energy_ratio(design,tmp_path,monkeypatch):
    import tensorflow as tf
    from bayesfilter.testing.inference_validation.engines import acceptance_hmc
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    original = acceptance_hmc.FrozenTransition
    def incorrect_transition(*args,**kwargs):
        transition = original(*args,**kwargs)
        audit_step = transition.audit_step
        def incorrect_ratio(*step_args):
            row = dict(audit_step(*step_args))
            row["log_accept_ratio"] = row["log_accept_ratio"] + tf.constant(.1,tf.float64)
            return row
        transition.audit_step = incorrect_ratio
        return transition
    monkeypatch.setattr(acceptance_hmc,"FrozenTransition",incorrect_transition)
    d = design("acceptance",route="prepared",options={"reference_anchors":32,"evidence_rungs":[1,2]})
    with pytest.raises(ValueError,match="independent endpoint energies"):
        acceptance_hmc.stationary_reference(d,ValidationTarget("gaussian",jit_compile=False),tmp_path)
    assert not (tmp_path/"reference.json").exists()


def test_public_acceptance_expired_deadline_does_not_issue_receipts(design,tmp_path):
    d = design("acceptance",route="prepared",options={"reference_anchors":32,"evidence_rungs":[1,2]})
    with pytest.raises(TimeoutError,match="deadline"):
        run(d,tmp_path,deadline=.001)
    assert not list(tmp_path.glob("replication-*"))
