"""Independent arithmetic, missingness and public-route diagnostic checks."""
from dataclasses import replace
import math

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_precision import mean_precision
from bayesfilter.testing.inference_validation.designs import ScenarioSpec
from bayesfilter.testing.inference_validation.engines import reference_mean as engine
from bayesfilter.testing.inference_validation.storage import read_json, write_tensor


@pytest.mark.parametrize("kind", ["iid", "persistent", "antithetic", "ties", "negative", "constant"])
def test_independent_variance_against_tensorflow(kind):
    rng = np.random.default_rng(93874)
    x = rng.normal(size=(1003, 4, 1))
    if kind in {"persistent", "antithetic"}:
        rho = .9 if kind == "persistent" else -.6
        for t in range(1, len(x)):
            x[t] = rho * x[t-1] + math.sqrt(1-rho**2) * x[t]
    if kind == "ties":
        x = np.round(x, 1)
    if kind == "constant":
        x[:] = 2.
    if kind == "negative":
        x = np.broadcast_to(np.tile([1., 1., 1., -1., -1., -1.], 200)[:, None, None], (1200, 4, 1))
    options = {"batch_size": 6 if kind == "negative" else 30, "min_batches": 20}
    ref = engine.independent_lugsail(x, **options)
    actual = mean_precision(tf.constant(x), method="lugsail", jit_compile=False, **options)
    assert ref["available"] == bool(actual["valid"][0])
    np.testing.assert_allclose(ref["per_chain_long_run_variance"],
                               actual["per_chain_long_run_variance"][:, 0], rtol=2e-12, atol=1e-13)
    if ref["available"]:
        assert ref["mcse"] == pytest.approx(float(actual["mcse"][0]), rel=2e-12)
        assert ref["estimate"] == pytest.approx(float(actual["estimate"][0]), abs=1e-13)
    else:
        assert ref["mcse"] is None
    assert ref["unused_terminal_draws"] == len(x) % options["batch_size"]


def test_insufficient_batches_are_unavailable():
    result = engine.independent_lugsail(np.ones((25, 4, 1)))
    assert not result["available"] and result["reason"] == "insufficient_complete_batches"
    with pytest.raises(ValueError, match="finite"):
        engine.independent_lugsail(np.full((500, 4, 1), np.nan))


def pipeline_fixture(tmp_path):
    x = np.random.default_rng(172).normal(1.92, .4, size=(1600, 4, 1))
    path = write_tensor(tmp_path / "draws.tensor", tf.constant(x))
    member = {"candidate_id": "a", "status": "assessed", "draws_path": str(path),
              "warmup_exclusion_matches": True, "posterior": {"passed": True,
              "warmup_cap_hit": False, "retained_cap_hit": False, "hard_vetoes": [],
              "config": {"retained_seed": [11, 12]}}}
    return {"data": [2.]*6, "completion": "complete", "verified_candidate_ids": ["b", "a"],
            "selection": {"rule": "first_verified", "scope": "selected", "candidate_ids": ["a"]},
            "members": [member]}, x


def assess(pipeline):
    return engine.alarm_from_pipeline("0", pipeline, alpha=.01, mcse_sd_max=.05, tau=2., sigma=1.)


def test_reference_independent_of_shifted_draws_and_runtime_mcse(tmp_path):
    pipeline, x = pipeline_fixture(tmp_path)
    row = assess(pipeline)
    assert row["reference_mean"] == pytest.approx(1.92)
    assert row["reference_sd"] == pytest.approx(.4)
    assert row["status"] == "qualified" and not row["alarm"]
    assert row["cutoff"] == pytest.approx(2.5758293035489004)
    # A fabricated runtime MCSE must not replace independent draw arithmetic.
    pipeline["members"][0]["posterior"]["mcse"] = 99.
    path = write_tensor(tmp_path / "shifted.tensor", tf.constant(x+.1))
    pipeline["members"][0]["draws_path"] = str(path)
    defect = assess(pipeline)
    assert defect["alarm"] and defect["reference_mean"] == row["reference_mean"]
    assert defect["independent_precision"]["mcse"] == pytest.approx(row["independent_precision"]["mcse"])


@pytest.mark.parametrize("veto", ["warmup_cap_hit", "retained_cap_hit", "hard_vetoes", "passed", "warmup_exclusion"])
def test_unqualified_fits_cannot_be_detections(tmp_path, veto):
    pipeline, _ = pipeline_fixture(tmp_path)
    member = pipeline["members"][0]
    if veto == "warmup_exclusion":
        member["warmup_exclusion_matches"] = False
    else:
        member["posterior"][veto] = (["invalid_energy"] if veto == "hard_vetoes" else veto != "passed")
    row = assess(pipeline)
    assert row["status"] == "unavailable" and row["alarm"] is None


def test_empty_selection_and_changed_member_rule(tmp_path):
    pipeline, _ = pipeline_fixture(tmp_path)
    pipeline["verified_candidate_ids"] = []
    pipeline["selection"]["candidate_ids"] = []
    assert assess(pipeline)["reason"] == "no_verified_member"
    pipeline["selection"]["rule"] = "best_mcse"
    with pytest.raises(ValueError, match="predetermined"):
        assess(pipeline)


def alarm_records(n, alarms=0):
    return [{"fit_id": str(i), "detector": engine.DETECTOR, "output_rule": engine.OUTPUT_RULE,
             "alpha": .01, "mcse_sd_max": .05, "status": "qualified", "alarm": i < alarms,
             "z": 3. if i < alarms else 0., "stream": [57, i]} for i in range(n)]


def summarize(rows, n, control):
    return engine.summarize_alarms(rows, [str(i) for i in range(n)], control=control,
                                    alpha=.01, mcse_sd_max=.05)


def test_complete_denominators_and_exact_screen_boundaries():
    assert summarize(alarm_records(128, 6), 128, "baseline")["rate_screen_passed"]
    assert not summarize(alarm_records(128, 7), 128, "baseline")["rate_screen_passed"]
    assert summarize(alarm_records(64, 58), 64, "location_shift")["rate_screen_passed"]
    assert not summarize(alarm_records(64, 57), 64, "location_shift")["rate_screen_passed"]
    null = summarize(alarm_records(120), 128, "baseline")
    assert null["conservative_numerator"] == 8 and not null["rate_screen_passed"]
    defect = summarize(alarm_records(57, 57), 64, "location_shift")
    assert defect["conservative_numerator"] == 57 and not defect["rate_screen_passed"]
    assert defect["unavailable"] == 7 and defect["finding"] == "calibration_incomplete"


def test_rate_response_uses_full_inventory_screen_instead_of_any_alarm(design):
    from bayesfilter.testing.inference_validation.controls import response
    d = normal_design(design)
    null = summarize(alarm_records(128, 6), 128, "baseline")
    assert null["finding"] == "discrepancy_detected"
    assert response(d, null)["status"] == "reference_mean_rate_screen_passed"
    missing = summarize(alarm_records(120), 128, "baseline")
    assert response(d, missing)["status"] == "reference_mean_rate_screen_not_passed"
    assert response(d, missing)["unavailable"] == 8
    assert not response(d, null)["power_established"]


def test_completed_but_unqualified_fit_is_not_complete_assessment(design):
    from bayesfilter.testing.inference_validation.designs import seed_for
    d = normal_design(design)
    row = engine.unavailable_record(d, 0, "retained_cap")
    row["data_seed"] = list(seed_for(d.seed, d.design_id, 0, "data"))
    result = engine.summarize_replications(d, [row])
    assert result["completed_fits"] == 1 and result["all_fits_recorded"]
    assert not result["assessment_complete"] and not result["execution_failures"]
    assert result["unavailable"] == result["conservative_numerator"] == 1


def test_duplicate_corrupt_and_changed_policy_records_fail():
    rows = alarm_records(2)
    for changed in ([rows[0], rows[0]], [rows[0], {**rows[1], "stream": rows[0]["stream"]}],
                    [rows[0], {**rows[1], "alpha": .05}],
                    [rows[0], {**rows[1], "z": float("nan")}],
                    [rows[0], {**rows[1], "status": "unavailable", "alarm": True}]):
        with pytest.raises(ValueError):
            summarize(changed, 2, "baseline")


def normal_design(design, **overrides):
    values = dict(engine="reference_mean", target="normal_conjugate", route="ordinary",
        scenario=ScenarioSpec("normal_conjugate", "ordinary", parameters={"tau":2., "sigma":1., "n":6}),
        alpha=.01, replications=1, draws=512, posterior_cap=2048, mcse_tolerance=.02,
        options={"posterior_members":"selected", "member_rule":"first_verified",
                 "reference_mean_alarm":{"mcse_sd_max":.05}})
    values.update(overrides)
    return design(**values)


def test_design_binds_estimator_scale_selection_and_data(design):
    base = normal_design(design)
    for changes in ({"posterior_members":"all"}, {"member_rule":"declared_l_first"},
                    {"data":[2.]*6}, {"posterior_precision_method":"autocorrelation"},
                    {"reference_mean_alarm":{"mcse_sd_max":.2}}):
        with pytest.raises(ValueError, match="reference_mean"):
            replace(base, options={**base.options, **changes})
    with pytest.raises(ValueError, match="absolute MCSE"):
        replace(base, mcse_tolerance=.05)
    with pytest.raises(ValueError, match="only supported"):
        replace(base, engine="accuracy")
    with pytest.raises(ValueError, match="nested power"):
        design("power", "normal_conjugate", "ordinary", options={"calibration_design":base.payload()})


def test_public_full_fit_and_cached_policy_identity(design, tmp_path):
    # Explicitly small CPU mechanics integration. Broad scientific calibration
    # uses the native-search campaign, not these permissive acceptance settings.
    base = normal_design(design)
    d = replace(base, budget_seconds=240, mcse_tolerance=.2, l_grid=(3, 5), options={
        **base.options, "reference_mean_alarm":{"mcse_sd_max":.5},
        "bootstrap_initialization_rounds":5, "metric_evidence_policy":"finite_window",
        "metric_probe_num_results":16, "preparation_max_restarts":1,
        "acceptance_policy":{"practical_region":(.41,.99), "repair_region":(.405,.995)},
        "search":{"pilot_enabled":False, "refinement_rounds":0,
                  "total_budget_units":32, "repair_reserve_units":4, "evidence_rungs":(1,)}})
    report = engine.run(d, tmp_path)
    assert report["recorded"] == report["planned"] == 1
    row = report["records"][0]
    pipeline = read_json(tmp_path / "replication-0000/pipeline.json")
    assert row["data"] == pipeline["data"] and len(row["data"]) == 6
    assert row["reference_mean"] == pytest.approx(sum(row["data"])/6.25)
    assert pipeline["verified_candidate_ids"]
    assert row["candidate_id"] == sorted(pipeline["verified_candidate_ids"])[0]
    assert row["status"] == "qualified"
    again = engine.run(d, tmp_path)
    assert again == report
    with pytest.raises(ValueError, match="identity changed"):
        engine.run(replace(d, alpha=.02), tmp_path)
