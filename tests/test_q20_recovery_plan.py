"""Recovery and exact pricing engineering checks; tiny CPU fixtures only."""
import json
from pathlib import Path
import time

import pytest

from bayesfilter.inference.q20_pricing import PricingLedger
from bayesfilter.inference.q20_stage_budget import StageBudgetPause


def test_interrupted_pricing_reuses_complete_blocks_and_rejects_changed_scope(tmp_path):
    calls = []
    first = PricingLedger(tmp_path/"first", {"target": "known"})
    assert first.run("one", {"L": 3}, lambda: calls.append("one") or 4.) == 4.
    with pytest.raises(RuntimeError):
        first.run("two", {"L": 5}, lambda: (_ for _ in ()).throw(RuntimeError("interrupted")))
    resumed = PricingLedger(tmp_path/"resume", {"target": "known"}, resume=first.root)
    assert resumed.run("one", {"L": 3}, lambda: pytest.fail("repeated computation")) == 4.
    assert resumed.run("two", {"L": 5}, lambda: calls.append("two") or 6.) == 6.
    assert calls == ["one", "two"]
    with pytest.raises(ValueError, match="scope"):
        PricingLedger(tmp_path/"wrong", {"target": "different"}, resume=first.root)
    with pytest.raises(ValueError, match="inputs"):
        resumed.run("one", {"L": 25}, lambda: 0.)
    record = next(p for p in first.root.glob("*.json") if p.name != "identity.json")
    raw = json.loads(record.read_text()); raw["value"] = 99.
    record.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="checksum"):
        first.run("one", {"L": 3}, lambda: 0.)


def test_pricing_checks_next_block_cost_but_can_replay_without_funds(tmp_path):
    store = PricingLedger(tmp_path/"price", {"target": "known"}, deadline=time.monotonic()+1.)
    with pytest.raises(StageBudgetPause):
        store.run("large", {}, lambda: pytest.fail("unfunded"), reserve_seconds=2.)
    assert json.loads((store.root/"pending.json").read_text())["required_seconds"] == 2.
    store.run("saved", {}, lambda: 1.)
    store.deadline = 0.
    assert store.run("saved", {}, lambda: pytest.fail("repeated")) == 1.


def test_actual_pricing_resume_skips_saved_training_and_hmc(tmp_path, monkeypatch):
    from tests.test_q20_master_integration import protocol
    from tests.test_q20_production_repair import four_dimensional_bridge
    from bayesfilter.inference.q20_master_stages import dispatch
    from bayesfilter.inference import q20_production_training
    from bayesfilter.inference.hmc import ReusableFullChainHMCRunner
    config, bridge = protocol(), four_dimensional_bridge()
    run = PricingLedger.run
    def interrupt(self, key, inputs, compute, **kwargs):
        if key == "reference":
            raise StageBudgetPause("forced interruption after HMC prices")
        return run(self, key, inputs, compute, **kwargs)
    monkeypatch.setattr(PricingLedger, "run", interrupt)
    with pytest.raises(StageBudgetPause):
        dispatch(config, bridge, tmp_path/"first", {"stage": "price"}, {"mode": "tiny_cpu_reference"})
    monkeypatch.setattr(PricingLedger, "run", run)
    def forbidden(*args, **kwargs):
        pytest.fail("completed training/HMC measurement repeated")
    monkeypatch.setattr(q20_production_training, "price_training", forbidden)
    monkeypatch.setattr(ReusableFullChainHMCRunner, "run", forbidden)
    result = dispatch(config, bridge, tmp_path/"resumed", {"stage": "price",
        "pricing_resume": str(tmp_path/"first/pricing-ledger")}, {"mode": "tiny_cpu_reference"})
    assert result["status"] == "estimation_method_cost_measurements"
    assert result["hmc"] and result["reference_batch_first_seconds"] > 0


def test_terminal_contention_preserves_worker_result(tmp_path, monkeypatch):
    from bayesfilter.inference import q20_master_stages as stages
    from bayesfilter.inference import q20_pricing
    from tests.test_q20_production_repair import tiny_protocol
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    calls = []
    monkeypatch.setattr(stages, "dispatch", lambda *a: calls.append(1) or {"status": "computed", "answer": 42})
    monkeypatch.setattr(q20_pricing, "resource_observation", lambda _: {"quality": "contention_observed", "receipt": {"other_gpu_pids": [123]}})
    result = stages.run_worker({"stage": "price", "fixture": True, "config": tiny_protocol()}, tmp_path/"worker")
    assert calls == [1] and result["completed"] and result["numerical_completed"]
    assert result["terminal_resource"]["quality"] == "contention_observed"
    assert json.loads((tmp_path/"worker/data/result.json").read_text())["answer"] == 42
    assert json.loads((tmp_path/"worker/worker-result.json").read_text())["completed"]


def test_exact_selected_price_binds_members_and_can_veto_affordability():
    from tests.test_q20_estimation_objective import prices
    from bayesfilter.inference.q20_production_config import protocol_template
    from bayesfilter.inference.q20_master_program import selected_procedure_forecast
    config = protocol_template(); pricing = prices(config, "neutra")
    selected = {"status": "selected_procedure_priced", "method": "neutra", "members": ["one"],
        "first_seconds": 40., "steady_seconds": 2., "worker_initialization_seconds": 10.}
    quote = selected_procedure_forecast(config, pricing, selected, {"neutra-beta1": {"path": "one"}})
    assert quote["first_assessment_seconds"] > 12000.
    assert quote["chunk_seconds"] > 4000.
    with pytest.raises(ValueError, match="bind"):
        selected_procedure_forecast(config, pricing, selected, {"neutra-beta1": {"path": "other"}})


def test_only_contended_selected_timing_retries_within_one_cumulative_cap(tmp_path, monkeypatch):
    from tests.test_q20_campaign_operations import driver, FakeCampaign
    monkeypatch.setattr(driver.time, "sleep", lambda _: None)
    campaign = FakeCampaign(tmp_path, ["pass", "pass"])
    name = "price-selected-neutra"
    campaign.state["recovery_plan"] = {"diagnostic_blocks": {
        name: {"minimum_seconds": 10., "maximum_seconds": 100.}}}
    original = campaign.numerical_stage
    def numerical(*args, **kwargs):
        result = original(*args, **kwargs)
        result["terminal_resource"] = {"quality": "contention_observed" if campaign.calls == 1 else "no_contention_observed"}
        return result
    campaign.numerical_stage = numerical
    job = {"deadline_epoch": time.time()+1000., "deadline_local": "test", "poll_seconds": 1., "max_recovery_attempts_per_phase": 1}
    result = driver.execute_phase(campaign, job, name, {"stage": "price-selected"}, diagnostic=True)
    assert result["completed"] and campaign.calls == 2
    assert campaign.state["stage_limits"][name] == 100.
    assert campaign.state["attempts"][1]["cap_seconds"] == 98.
    assert campaign.state["spent_seconds"] == 4.


def test_cached_losses_do_not_erase_restore_and_reassessment_cost(tmp_path, monkeypatch):
    from tests.test_q20_campaign_operations import driver, FakeCampaign
    from bayesfilter.inference import q20_training_resume
    from bayesfilter.inference.q20_production_config import training_cohort
    campaign = FakeCampaign(tmp_path, [])
    campaign.state["recovery_plan"] = {"active": True}
    candidates = training_cohort(campaign.config, method="neutra")
    saved = {"cohort": {c["id"]: {"exports": {}, "session": {"map": {"beta": 1.}, "level_updates": 512}}
                        for c in candidates}}
    monkeypatch.setattr(q20_training_resume, "read_training_checkpoint", lambda *args: saved)
    result = {"completed": True, "result": {"training_quote": {"minimum_cohort_seconds": 0.},
        "training": {"rows": [{"width": w, "beta": 1., "batch_size": 32, "setup_seconds": 10., "heldout_first_seconds": 4.}
                              for w in campaign.config["training"]["widths"]]}}}
    priced = driver.reassessment_pricing(campaign, {"stage": "price", "method": "neutra", "training_checkpoint": "saved"}, result)
    assert priced["result"]["training_quote"]["minimum_cohort_seconds"] == 2*len(candidates)*18.
    assert result["result"]["training_quote"]["minimum_cohort_seconds"] == 0.


def test_explicit_repaired_attempt_keeps_prior_charge_and_cumulative_ceiling(tmp_path):
    from tests.test_q20_campaign_operations import driver, FakeCampaign
    campaign = FakeCampaign(tmp_path, ["pass"])
    campaign.state["attempts"] = [{"stage": "train-neutra", "status": "completed", "diagnostic": False,
        "elapsed_seconds": 23., "failure_classification": "budget_pause", "directory": str(tmp_path/"old")}]
    campaign.state["spent_seconds"] = 23.
    campaign.state["stage_limits"]["train-neutra"] = 123.
    campaign.state["recovery_plan"] = {"repair_attempt_baselines": {"train-neutra": 1}}
    job = {"deadline_epoch": time.time()+1000., "deadline_local": "test", "poll_seconds": 1., "max_recovery_attempts_per_phase": 1}
    result = driver.execute_phase(campaign, job, "train-neutra", {"stage": "train"}, reserve=123.)
    assert result["completed"] and campaign.calls == 1
    assert campaign.state["attempts"][-1]["cap_seconds"] == 100.
    assert campaign.state["spent_seconds"] == 25.


def test_actual_finite_cohort_fresh_verification_and_selected_price(tmp_path):
    from tests.test_q20_master_integration import protocol
    from tests.test_q20_production_repair import four_dimensional_bridge
    from bayesfilter.inference.q20_production_training import run_training_cohort
    from bayesfilter.inference.q20_production_hmc import tune_scope, sample_member
    config, bridge = protocol(), four_dimensional_bridge()
    trained = run_training_cohort(config, bridge, tmp_path/"train", memory_policy={"mode": "tiny_cpu_reference"}, max_seconds=120., method="neutra")
    cohort = json.loads(Path(trained["checkpoint"]).read_text())["cohort"]
    export = next(iter(cohort.values()))["exports"]["1.0"]
    tuned = tune_scope(config, bridge, tmp_path/"tune", method="neutra", training_export=export,
        explicit_cohort={"id": "finite-repair-fixture", "epsilon": .3},
        max_seconds=config["tuning"]["max_wall_seconds"]+60.)
    checkpoint = json.loads((tmp_path/"tune/tuning/tuning_checkpoint.json").read_text())["result"]
    assert len(checkpoint["candidates"]) == len(config["tuning"]["l_grid"])
    assert all(c["epsilon"] == .3 for c in checkpoint["candidates"])
    assert checkpoint["verification_receipts"] and tuned["verified_members"]
    assert checkpoint["config"]["max_wall_time_seconds"] > config["tuning"]["max_wall_seconds"]
    member = next(iter(tuned["verified_members"].values()))
    priced = sample_member(config, bridge, tmp_path/"selected", member_path=member, label="cost", pricing_only=True)
    assert priced["first_seconds"] > 0 and priced["steady_seconds"] > 0
    assert priced["members"] == [member] and priced["transitions_per_call"] == config["execution"]["pricing_transitions"]
    assert not (tmp_path/"selected/retained.tensor").exists()
