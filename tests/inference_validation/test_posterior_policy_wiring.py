"""Public call-chain checks for declared posterior diagnostics, not calibration."""
from dataclasses import replace

import numpy as np
import pytest

from bayesfilter.testing.inference_validation.engines.stopping import arm_quantities
from bayesfilter.testing.inference_validation.catalog import get_target
from bayesfilter.testing.inference_validation.procedures import execute_pipeline
from bayesfilter.testing.inference_validation.storage import read_json, read_tensor


@pytest.mark.parametrize("key,value", [
    ("posterior_precision_settings", None), ("posterior_precision_settings", {"batch_size": True}),
    ("posterior_precision_settings", {"batch_size": 0}),
    ("posterior_precision_settings", {"min_batches": 1}),
    ("posterior_precision_settings", {"lugsail_r": 1.5}),
    ("posterior_precision_settings", {"lugsail_c": float("nan")}),
    ("posterior_precision_settings", {"lugsail_c": 1}),
    ("posterior_precision_settings", {"jit_compile": False}),
    ("posterior_assessment_settings", {"warmup_consecutive_checks": 0}),
    ("posterior_assessment_settings", {"retained_bulk_ess_min": True}),
    ("posterior_assessment_settings", {"warmup_tail_ess_min": -1}),
    ("posterior_assessment_settings", {"retained_tail_ess_min": float("inf")}),
    ("posterior_assessment_settings", {"warmup_rhat_max": 8}),
])
def test_bad_settings_rejected_before_execution(design, key, value):
    with pytest.raises(ValueError, match=key):
        design(options={key: value})


def test_comparator_matches_independent_batch_formula_and_unavailable_batches():
    values = np.random.default_rng(793).normal(size=(960, 4, 2))
    def lrv(b):
        return b * values.reshape(960 // b, b, 4, 2).mean(axis=1).var(axis=0, ddof=1)
    expected = np.sqrt(((lrv(12) - .25 * lrv(6)) / .75).sum(axis=0) / (960 * 16))
    result = arm_quantities(values, get_target("gaussian"), {}, None, jit_compile=False,
                            batch_size=12, min_batches=80, lugsail_r=2, lugsail_c=.25)
    assert result["x:mean"]["mcse"] == pytest.approx(expected[0], rel=1e-12)
    assert result["y:mean"]["mcse"] == pytest.approx(expected[1], rel=1e-12)
    unavailable = arm_quantities(values, get_target("gaussian"), {}, None, jit_compile=False,
                                 batch_size=12, min_batches=81)
    assert not unavailable["x:mean"]["available"]


def test_public_stopped_fixed_policy_and_restart_identity(design, tmp_path, monkeypatch):
    from bayesfilter.testing.inference_validation.engines.pipeline import run
    settings = {"batch_size": 8, "min_batches": 4, "lugsail_r": 2, "lugsail_c": .25}
    readiness = {"warmup_bulk_ess_min": 1., "warmup_tail_ess_min": 1.,
                 "retained_bulk_ess_min": 1., "retained_tail_ess_min": 1.,
                 "warmup_consecutive_checks": 2}
    d = design("stopping", "gaussian", "prepared", replications=1, draws=512,
        posterior_cap=1024, step_size=1.3, l_grid=(2, 3),
        options={"posterior_members": "selected", "member_rule": "first_verified",
            "posterior_precision_settings": settings, "posterior_assessment_settings": readiness,
            "acceptance_policy": {"practical_region": (.41, .99), "repair_region": (.405, .995)},
            "search": {"pilot_enabled": False, "refinement_rounds": 0,
                       "total_budget_units": 24, "repair_reserve_units": 4, "evidence_rungs": (1,)},
            "fixed_comparator": {"warmup_results": 128, "retained_results": 128}})
    result = run(d, tmp_path)
    fit = tmp_path / "replication-0000"
    pipeline = read_json(fit / "pipeline.json")
    member = next(m for m in pipeline["members"] if m["status"] == "assessed")
    policy = member["posterior"]["config"]["assessment_policy"]
    assert all(policy[k] == v for k, v in readiness.items())
    assert all(policy["precision"][k] == v for k, v in settings.items())
    assert member["recorded_retained_count"] > 0
    pair = result["replications"][0]["members"][0]["stopping_pair"]
    for arm, path in (("stopped", member["draws_path"]),
                      ("fixed", member["fixed_comparator"]["draws_path"])):
        independent = arm_quantities(read_tensor(path), get_target("gaussian"), {}, None,
                                      jit_compile=False, **settings)
        assert pair[arm] == independent
    last = member["posterior"]["retained_checks"][-1]
    mean = next(t for t in last[last["diagnostic_role"]]["precision"]["targets"]
                if t["name"] == "x" and t["kind"] == "mean")
    assert pair["stopped"]["x:mean"]["mcse"] == pytest.approx(mean["mcse"], rel=1e-12)
    assert execute_pipeline(d, fit)["verified_candidate_ids"] == pipeline["verified_candidate_ids"]
    changed = replace(d, options={**d.options, "posterior_precision_settings": {**settings, "batch_size": 16}})
    assert changed.identity != d.identity
    from bayesfilter.testing.inference_validation.engines.pipeline import run_replication
    with pytest.raises(ValueError, match="identity changed"):
        run_replication(changed, tmp_path, 0)
    with pytest.raises(ValueError, match="identity|checkpoint|scope|policy"):
        execute_pipeline(changed, fit)
    for key, value in (("posterior_assessment_settings", {**readiness, "warmup_consecutive_checks": 3}),
                       ("posterior_precision_method", "autocorrelation")):
        with pytest.raises(ValueError, match="identity changed"):
            execute_pipeline(replace(d, options={**d.options, key: value}), fit)
    from bayesfilter.testing.inference_validation import execution
    with monkeypatch.context() as patch:
        patch.setattr(execution, "source_state", lambda: {"identity": "changed-source"})
        with pytest.raises(ValueError, match="identity changed"):
            run_replication(d, tmp_path, 0)
    (fit / "fit_identity.json").unlink()
    with pytest.raises(ValueError, match="legacy pipeline"):
        execute_pipeline(d, fit)
