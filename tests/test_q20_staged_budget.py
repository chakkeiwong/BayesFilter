"""Staged spending and resource pauses, with no q20 scientific claims."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bayesfilter.inference.q20_master_program import forecast_campaign, run_estimation_attempts, posterior_chunk_forecast
from bayesfilter.inference.q20_stage_budget import allocate_stage, repair_remaining, StageBudgetPause, BudgetedCheckpoint
from bayesfilter.inference.q20_production_config import protocol_template, digest
from tests.test_q20_estimation_objective import harness, prices


def test_forecast_uses_actual_tuning_and_repair_limits():
    config = protocol_template()
    measured = prices(config, "neutra")
    for row in measured["hmc"]:
        row.update(first_seconds=135., steady_seconds=29.)
    quote = forecast_campaign(config, measured)
    assert quote["reserves"]["tuning"] == 28800.
    assert quote["reserves"]["localized_repair"] == 7200.
    assert "minimum_complete_seconds" not in quote
    assert "configured_cap_reservation_seconds" not in quote
    r = config["reference"]
    rows = r["banks"] * r["rungs"][1]
    startup = measured["worker_initialization_seconds"] + 2*config["execution"]["termination_grace_seconds"]
    expected = config["budget"]["forecast_safety_factor"] * (startup
        + measured["reference_batch_first_seconds"] + rows/r["batch_size"]*measured["reference_batch_steady_seconds"]
        + rows*measured["reference_analysis_seconds_per_row"])
    assert quote["reference_first_assessment_seconds"] == expected
    assert quote["posterior_first_assessment_seconds"] > quote["posterior_first_chunk_seconds"]


def test_allocator_debits_actual_time_and_never_renews_a_stage_or_repair_hold():
    config = protocol_template()
    state = {"stages": {}, "attempts": [], "stage_limits": {"tune": 100.}}
    campaign = SimpleNamespace(config=config, state=state, remaining=lambda: 80.)
    assert allocate_stage(campaign, "tune", 9999., protected_seconds=20.) == 60.
    state["attempts"].append({"stage": "tune", "status": "failed", "elapsed_seconds": 60.})
    assert allocate_stage(campaign, "tune", 9999.) == 100.
    assert repair_remaining(campaign) == 7140.
    state["attempts"].append({"stage": "sample-ensemble", "status": "failed", "elapsed_seconds": 7000.})
    assert repair_remaining(campaign) == 140.
    with pytest.raises(StageBudgetPause):
        allocate_stage(campaign, "new", 100., protected_seconds=80.)
    state["stages"]["tune"] = {}
    assert allocate_stage(campaign, "tune", 9999., protected_seconds=1000.) == 100.


def test_expensive_caps_do_not_prevent_affordable_master_stages(tmp_path):
    campaign, stage, finish, calls = harness(tmp_path)
    limits = {}
    def costly(name, request, **kwargs):
        result = stage(name, request, **kwargs)
        if request["stage"] == "price":
            for row in result["result"]["hmc"]:
                # Sampling caps exceed the entire allowance, but the next
                # bounded chunk and the training floor are affordable.
                row["steady_seconds"] = .2 if row["L"] == 3 else .25
        if "reserve" in kwargs:
            limits[name] = kwargs["reserve"]
        return result
    result = run_estimation_attempts(campaign, costly, finish)
    assert result["details"]["method"] == "neutra"
    assert limits["tune-neutra-beta1"] <= 28800.
    assert limits["sample-neutra"] < campaign.remaining()
    assert limits["train-neutra"] < 100.


def test_cheap_sampling_cannot_hide_unaffordable_required_tuning_cohort(tmp_path):
    campaign, stage, finish, calls = harness(tmp_path)
    def costly(name, request, **kwargs):
        result = stage(name, request, **kwargs)
        if request["stage"] == "price":
            for row in result["result"]["hmc"]:
                row["steady_seconds"] = .2 if row["L"] == 3 else 29.
        return result
    with pytest.raises(StageBudgetPause):
        run_estimation_attempts(campaign, costly, finish)
    assert [request["stage"] for _, request in calls] == ["price"]


@pytest.mark.parametrize("status", ["partial_budget", "budget_bound", "budget_paused"])
def test_partial_tuning_cannot_select_member_or_fallback(tmp_path, status):
    campaign, stage, finish, calls = harness(tmp_path)
    def paused(name, request, **kwargs):
        result = stage(name, request, **kwargs)
        if request["stage"] == "tune":
            result["result"]["status"] = status
        return result
    with pytest.raises(StageBudgetPause):
        run_estimation_attempts(campaign, paused, finish)
    assert not any(r["stage"] in {"sample", "ensemble"} for _, r in calls)
    assert not any("ensemble" in name for name, _ in calls)


def test_selected_trajectory_cost_controls_chunk_allowance():
    config = protocol_template()
    measured = prices(config, "neutra")
    for row in measured["hmc"]:
        row["steady_seconds"] = float(row["L"])
    short = posterior_chunk_forecast(config, measured, {"neutra-beta1": {"L": 3}})
    middle = posterior_chunk_forecast(config, measured, {"neutra-beta1": {"L": 10}})
    long = posterior_chunk_forecast(config, measured, {"neutra-beta1": {"L": 25}})
    assert short < middle == long
    with pytest.raises(ValueError, match="coverage"):
        posterior_chunk_forecast(config, measured, {"neutra-beta1": {"L": 30}})


def test_checkpoint_pauses_before_new_work_and_replays_without_new_compute(tmp_path, monkeypatch):
    import tensorflow as tf
    from bayesfilter.inference import q20_stage_budget
    clock = [0.]
    monkeypatch.setattr(q20_stage_budget, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    def compute():
        clock[0] += 4.
        return tf.constant([1., 2.], tf.float64)
    with BudgetedCheckpoint(tmp_path/"chunks", {"target": "fixture"}, deadline=8., chunk_seconds=1., safety_factor=2.) as store:
        original = store.run("first", {"seed": 1}, compute)
        with pytest.raises(StageBudgetPause):
            store.run("second", {"seed": 2}, compute)
        assert store.contains("first") and not store.contains("second")
    with BudgetedCheckpoint(tmp_path/"chunks", {"target": "fixture"}, deadline=0.) as store:
        replayed = store.run("first", {"seed": 1}, lambda: pytest.fail("numerical work repeated"))
        tf.debugging.assert_equal(original, replayed)
        with pytest.raises(StageBudgetPause):
            store.run("second", {"seed": 2}, compute)


@pytest.mark.parametrize("kind,status", [
    ("tune", "partial_budget"), ("tune", "budget_bound"), ("sample", "budget_paused"),
    ("reference", "budget_paused"), ("train", "partial_validation_budget"),
])
def test_supervisor_does_not_cache_budget_pauses(tmp_path, monkeypatch, kind, status):
    from tests.test_q20_campaign_runtime import campaign
    from bayesfilter.inference.q20_campaign_runtime import atomic_json
    c = campaign(tmp_path)
    c.config["cpu_reference"] = True
    def executed(name, command, **kwargs):
        folder = tmp_path/"attempt"
        (folder/"worker/data").mkdir(parents=True)
        atomic_json(folder/"worker/worker-result.json", {"completed": True, "result": {"status": status},
                    "status": status, "wall_seconds": .1})
        assert kwargs["request"]["cooperative_seconds"] < 5.
        attempt = {"directory": str(folder), "status": "completed", "stage": name, "elapsed_seconds": .1}
        c.state["attempts"].append(attempt)
        return attempt
    monkeypatch.setattr(c, "execute", executed)
    with c.locked():
        result = c.numerical_stage(kind, {"stage": kind}, cap_seconds=5., diagnostic=False)
        assert result["budget_paused"] and not result["completed"]
        assert kind not in c.state["stages"]


@pytest.mark.parametrize("method", ["neutra", "ensemble"])
def test_training_handoff_waits_for_full_floor_and_all_temperatures(tmp_path, method):
    from tests.test_q20_production_repair import tiny_protocol, four_dimensional_bridge
    from bayesfilter.inference.q20_production_training import run_training_cohort
    from bayesfilter.inference.q20_production_config import method_betas
    config = tiny_protocol()
    config["training"].update(rungs=[1, 2, 4], cohort_min_updates=2)
    result = run_training_cohort(config, four_dimensional_bridge(), tmp_path/"training",
        memory_policy={"mode": "tiny_cpu_reference"}, max_seconds=120., method=method, stop_when_trial_ready=True)
    assert result["method_complete"]
    cohort = json.loads(Path(result["checkpoint"]).read_text())["cohort"]
    for item in cohort.values():
        assert item["session"]["level_updates"] == 2
        assert set(item["exports"]) == {str(beta) for beta in method_betas(config, method)}
        assert max(a["updates"] for a in item["assessments"]) == 2


def test_reference_pause_resumes_preserved_chunks_exactly(tmp_path, monkeypatch):
    from tests.test_q20_production_repair import tiny_protocol, four_dimensional_bridge
    from bayesfilter.inference.q20_production_reference import run_reference
    from bayesfilter.inference import q20_stage_budget
    config, bridge = tiny_protocol(), four_dimensional_bridge()
    config["reference"].update(banks=4, rungs=[64, 128], batch_size=32, ess_min=1., minimum_tail_rows=1)
    baseline = run_reference(config, bridge, tmp_path/"full")
    original = q20_stage_budget.BudgetedCheckpoint.run
    def pause_after_first(store, key, inputs, compute):
        if store.records:
            raise StageBudgetPause("fixture allocation consumed after first chunk")
        return original(store, key, inputs, compute)
    with monkeypatch.context() as patch:
        patch.setattr(q20_stage_budget.BudgetedCheckpoint, "run", pause_after_first)
        with pytest.raises(StageBudgetPause):
            run_reference(config, bridge, tmp_path/"partial")
    committed = tmp_path/"partial/chunks/committed/bank-0-offset-0/bundle.sha256"
    checksum = committed.read_text()
    resumed = run_reference(config, bridge, tmp_path/"resumed", resume_chunks=tmp_path/"partial/chunks")
    assert baseline == resumed
    assert (tmp_path/"resumed/chunks/committed/bank-0-offset-0/bundle.sha256").read_text() == checksum


def test_public_tuning_partial_budget_resumes_without_new_allowance(tmp_path):
    from tests.test_q20_master_integration import protocol
    from tests.test_q20_production_repair import four_dimensional_bridge
    from bayesfilter.inference.q20_production_training import run_training_cohort
    from bayesfilter.inference.q20_production_hmc import tune_scope
    from bayesfilter.inference.hmc_candidate_set_checkpoint import load_numerical_tuning_checkpoint
    config, bridge = protocol(), four_dimensional_bridge()
    trained = run_training_cohort(config, bridge, tmp_path/"train", memory_policy={"mode": "tiny_cpu_reference"},
                                 max_seconds=120., method="neutra", stop_when_trial_ready=True)
    cohort = json.loads(Path(trained["checkpoint"]).read_text())["cohort"]
    export = next(iter(cohort.values()))["exports"]["1.0"]
    partial = tune_scope(config, bridge, tmp_path/"partial", method="neutra", training_export=export,
                         max_work_items=0, max_seconds=120.)
    assert partial["status"] == "partial_budget"
    _, before = load_numerical_tuning_checkpoint(partial["tuning_checkpoint"], adapter=bridge.fixed_beta_adapter(1.))
    assert 0 < before.config.max_wall_time_seconds < 120.
    finished = tune_scope(config, bridge, tmp_path/"resume", method="neutra", training_export=export,
                          resume=partial["tuning_checkpoint"], max_seconds=120.)
    assert finished["status"] == "complete"
    assert finished["verified_members"]
    _, after = load_numerical_tuning_checkpoint(finished["tuning_checkpoint"], adapter=bridge.fixed_beta_adapter(1.))
    assert before.config == after.config
    assert after.result().search_state["elapsed_seconds"] >= before.result().search_state["elapsed_seconds"]
