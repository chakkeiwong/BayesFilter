"""CPU-only mechanics for training repair, not q20 quality evidence."""
import copy
import json
from pathlib import Path
import sys
import time

import pytest
import tensorflow as tf

from bayesfilter.inference.q20_production_config import digest, scoped_seed, training_cohort
from bayesfilter.inference.neutra_training_protocol import assess_training_rung, export_weighted_transport, transport_parity
from bayesfilter.inference.q20_production_training import new_session, scope_for, run_training_cohort, source_snapshot
from bayesfilter.inference.q20_training_repair import repair_session, clipping_envelope
from tests.test_q20_production_repair import tiny_protocol, four_dimensional_bridge


def test_missing_distinct_increment_cannot_create_plateau():
    d = assess_training_rung(baseline={"lower": -2., "upper": -1.}, increment=None,
        reliability=True, prior_plateaus=0, at_cap=False, minimum_improvement=.04,
        maximum_half_width=.02, plateau_comparisons=2)
    assert not d["distinct_increment_observed"]
    assert not d["plateau_observed"]
    assert not d["continuing_improvement"]
    assert not d["at_training_cap"]


def test_requested_rung_covers_all_ensemble_temperatures(tmp_path):
    config, bridge = tiny_protocol(), four_dimensional_bridge()
    result = run_training_cohort(config, bridge, tmp_path/"ensemble", memory_policy={},
                                 max_seconds=60, method="ensemble", stop_after_rung=1)
    cohort = json.loads(Path(result["checkpoint"]).read_text())["cohort"]
    assert all(i["session"]["map"]["beta"] == 1. for i in cohort.values())
    assert all(set(i["exports"]) == {"0.5", "1.0"} for i in cohort.values())
    for item in cohort.values():
        for beta, path in item["exports"].items():
            exported = json.loads(Path(path).read_text())
            report = exported["assessment"]["post_training"]
            assert report["beta"] == float(beta)
            assert report["training_state_hash"] == exported["training_checkpoint_hash"]
            assert report["numerical_check_passed"]
            assert report["geometry"]["rows"] == config["validation"]["reliability_rows"]
            assert not report["posterior_qualified"]


def test_eligible_checkpoint_continues_to_next_rung(tmp_path):
    config, bridge = tiny_protocol(), four_dimensional_bridge()
    result = run_training_cohort(config, bridge, tmp_path/"first", memory_policy={},
        max_seconds=60, method="neutra", stop_after_rung=1)
    state = json.loads(Path(result["checkpoint"]).read_text())
    for item in state["cohort"].values():
        item["status"] = "hmc_trial_nominee"
    state.pop("checkpoint_hash")
    path = tmp_path/"nominee.json"
    path.write_text(json.dumps({**state, "checkpoint_hash": digest(state)}))
    second = run_training_cohort(config, bridge, tmp_path/"second", memory_policy={},
        max_seconds=60, method="neutra", resume=path)
    final = json.loads(Path(second["checkpoint"]).read_text())
    assert all(i["session"]["level_updates"] == 2 for i in final["cohort"].values())
    assert all(i["last_distinct_assessment"]["decision"]["distinct_increment_observed"] for i in final["cohort"].values())


def test_reissued_distinct_comparison_does_not_count_another_plateau(tmp_path):
    config, bridge = tiny_protocol(), four_dimensional_bridge()
    config["validation"].update(minimum_improvement=1e6, maximum_half_width=1e6)
    result = run_training_cohort(config, bridge, tmp_path/"first", memory_policy={},
                                 max_seconds=60, method="neutra")
    state = json.loads(Path(result["checkpoint"]).read_text())
    counts = {name: item["plateaus"] for name, item in state["cohort"].items()}
    assert set(counts.values()) == {2}
    for item in state["cohort"].values():
        item["historical_assessments"] = item["assessments"]
        item["assessments"], item["exports"] = [], {}
    state.pop("checkpoint_hash")
    path = tmp_path/"imported.json"
    path.write_text(json.dumps({**state, "checkpoint_hash": digest(state)}))
    result = run_training_cohort(config, bridge, tmp_path/"reissued", memory_policy={},
                                 max_seconds=60, method="neutra", resume=path)
    after = json.loads(Path(result["checkpoint"]).read_text())["cohort"]
    assert {name: item["plateaus"] for name, item in after.items()} == counts
    assert all(i["assessments"][-1]["reused_distinct_comparison"] for i in after.values())


def test_legacy_endpoint_gets_post_training_checks_without_retraining(tmp_path):
    config, bridge = tiny_protocol(), four_dimensional_bridge()
    result = run_training_cohort(config, bridge, tmp_path/"first", memory_policy={},
                                 max_seconds=60, method="neutra")
    state = json.loads(Path(result["checkpoint"]).read_text())
    before = {name: item["session"] for name, item in state["cohort"].items()}
    for item in state["cohort"].values():
        for assessment in item["assessments"]:
            assessment.pop("post_training", None)
    state.pop("checkpoint_hash")
    path = tmp_path/"old-assessments.json"
    path.write_text(json.dumps({**state, "checkpoint_hash": digest(state)}))
    result = run_training_cohort(config, bridge, tmp_path/"rechecked", memory_policy={},
                                 max_seconds=60, method="neutra", resume=path)
    after = json.loads(Path(result["checkpoint"]).read_text())["cohort"]
    assert {name: item["session"] for name, item in after.items()} == before
    assert all(item["historical_assessments"] for item in after.values())
    assert all(item["assessments"][-1]["post_training"]["numerical_check_passed"] for item in after.values())


def test_identity_depth_extension_preserves_nonzero_map_and_adam_prefix():
    config, bridge = tiny_protocol(), four_dimensional_bridge()
    candidate = training_cohort(config, method="neutra")[0]
    scope = scope_for(config, bridge, candidate, sources={}, memory_policy={})
    first = new_session(config, bridge, candidate, scope=scope)
    first = first.next_beta(1., root_seed=scoped_seed(config, "train", candidate["id"], 1.), preflight_seed=(2, 7))
    first.advance(3)
    saved = first.checkpoint()
    config["training"].update(stages=4, gradient_clip_norm=160.)
    new_scope = scope_for(config, bridge, candidate, sources={}, memory_policy={})
    repaired, _ = repair_session(config, bridge, saved, scope=new_scope, candidate=candidate)
    _, frozen = export_weighted_transport(first.trainer.transport,
        target_signature=bridge.fixed_beta_adapter(1.).adapter_signature(), training_state_hash=saved["state_hash"], transport_id="test")
    z = tf.random.stateless_normal([32, 4], (25, 7), dtype=tf.float64)
    assert transport_parity(repaired.trainer.transport, frozen, z, rtol=1e-9, atol=1e-10)["passed"]
    state = repaired.checkpoint()
    assert state["optimizer"][:len(saved["optimizer"])] == saved["optimizer"]
    assert state["iteration"] == saved["iteration"]
    assert state["rng_index"] == saved["rng_index"]
    repaired.advance(1)
    assert repaired.checkpoint()["iteration"] == 4
    config["training"]["stages"] = 3
    with pytest.raises(ValueError, match="paired identity"):
        repair_session(config, bridge, saved, scope=new_scope, candidate=candidate)


def test_envelope_uses_observed_norms_and_rejects_nonfinite():
    saved = {"history": [{"status": "accepted", "gradient_norm": 6.}]}
    assert clipping_envelope(saved, [[3., 4.]])["cap"] == 6.
    with pytest.raises(ValueError):
        clipping_envelope(saved, [[float("nan")]])


def test_queue_reserves_sum_and_accounts_overlap(tmp_path):
    from bayesfilter.inference.q20_campaign_runtime import Campaign, CampaignBudgetError
    from bayesfilter.inference.q20_parallel_training import execute_queue
    config = tiny_protocol()
    config["execution"]["termination_grace_seconds"] = .1
    repo = Path(__file__).resolve().parents[1]
    c = Campaign(tmp_path/"campaign", repo=repo, config=config,
                 allowance={"campaign_remaining_seconds": 30., "diagnostic_remaining_seconds": 15.})
    jobs = [{"stage": f"task{i}", "command": [sys.executable, "-c", "import time; time.sleep(1.2)"],
             "request": {}, "diagnostic": False, "cap_seconds": 10.} for i in range(2)]
    with c.locked():
        with pytest.raises(CampaignBudgetError, match="aggregate"):
            execute_queue(c, [{**j, "cap_seconds": 20.} for j in jobs], [0, 1])
        assert not c.state["attempts"]
        t = time.monotonic()
        results = execute_queue(c, jobs, [0, 1])
        wall = time.monotonic()-t
        assert all(a["status"] == "completed" for a in results)
        assert c.state["spent_seconds"] == sum(a["elapsed_seconds"] for a in results)
        assert c.state["spent_seconds"] > wall
        assert c.state["diagnostic_spent_seconds"] == 0


def test_master_retries_an_alternative_map_before_ensemble(tmp_path):
    from tests.test_q20_estimation_objective import harness
    from bayesfilter.inference.q20_master_program import run_estimation_attempts
    campaign, stage, finish, calls = harness(tmp_path)
    first = True
    def fail_first(name, request, **kwargs):
        nonlocal first
        result = stage(name, request, **kwargs)
        if request["stage"] == "tune" and first:
            first = False
            result["result"]["verified_members"] = {}
        return result
    result = run_estimation_attempts(campaign, fail_first, finish)
    assert result["details"]["method"] == "neutra"
    tuned = [r for _, r in calls if r["stage"] == "tune"]
    assert len(tuned) == 2
    assert tuned[0]["training_export"] != tuned[1]["training_export"]


def test_master_returns_failed_map_cohort_to_training(tmp_path):
    from tests.test_q20_estimation_objective import harness
    from bayesfilter.inference.q20_master_program import run_estimation_attempts
    campaign, stage, finish, calls = harness(tmp_path)
    repaired = False
    def repair_after_failures(name, request, **kwargs):
        nonlocal repaired
        result = stage(name, request, **kwargs)
        if request["stage"] == "train":
            repaired = "repair-u" in name
            p = Path(result["result"]["checkpoint"])
            cohort = json.loads(p.read_text())
            for item in cohort["cohort"].values():
                item["session"] = {"level_updates": request["stop_after_rung"]}
            p.write_text(json.dumps(cohort))
        if request["stage"] == "tune" and not repaired:
            result["result"]["verified_members"] = {}
        return result
    result = run_estimation_attempts(campaign, repair_after_failures, finish)
    assert result["details"]["method"] == "neutra"
    training = [r for _, r in calls if r["stage"] == "train"]
    assert [r["stop_after_rung"] for r in training] == [512, 2048]
    assert all(not r["stop_when_trial_ready"] for r in training)


def test_worker_honors_assigned_device_before_tf_import(tmp_path, monkeypatch):
    from bayesfilter.inference import q20_master_stages as stages
    from bayesfilter.inference.q20_gpu_runtime import GPUResourceUnavailable
    config = tiny_protocol()
    config["cpu_reference"] = False
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    seen = []
    def select(*, requested):
        seen.append(requested)
        raise GPUResourceUnavailable({"status": "FAILED", "test": "no_device_initialization"})
    monkeypatch.setattr(stages, "select_worker_gpu", select)
    stages.run_worker({"config": config, "stage": "repair-training", "gpu": 2}, tmp_path/"worker")
    assert seen == [2]


def test_repair_worker_writes_resumable_cohort_and_disposable_pilot(tmp_path):
    import hashlib
    from bayesfilter.inference.q20_training_repair import run_repair_arm
    from bayesfilter.inference.q20_training_resume import read_training_checkpoint
    config, bridge = tiny_protocol(), four_dimensional_bridge()
    result = run_training_cohort(config, bridge, tmp_path/"parent", memory_policy={},
                                 max_seconds=60, method="neutra")
    parent = Path(result["checkpoint"])
    candidate = training_cohort(config, method="neutra")[0]["id"]
    request = {"checkpoint": str(parent), "checkpoint_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        "candidate": candidate, "cooperative_seconds": 60., "updates": 2, "arm": "control", "sanity_only": False}
    actual = run_repair_arm(config, bridge, tmp_path/"worker", request=request, memory_policy={})
    assert actual["status"] == "repair_tranche_complete"
    report = actual["assessment"]["post_training"]
    assert report["numerical_check_passed"]
    assert actual["next_action"] == report["next_action"]
    assert report["clipping"]["observed_updates"] == request["updates"]
    exported = json.loads(Path(actual["export"]).read_text())
    assert exported["assessment"]["post_training"] == report
    cohort = read_training_checkpoint(actual["checkpoint"], config, sources=source_snapshot())
    assert cohort["cohort"][candidate]["session"]["iteration"] == 4
    request.update(sanity_only=True)
    pilot = run_repair_arm(config, bridge, tmp_path/"pilot", request=request, memory_policy={})
    assert not pilot["calibration_complete"]
    assert not pilot["assessment"]["decision"]["hmc_trial_eligible"]
    assert not pilot["assessment"]["post_training"]["hmc_trial_eligible"]
    assert not read_training_checkpoint(pilot["checkpoint"], config)["cohort"][candidate]["exports"]
