"""CPU reference checks for an administrative deadline extension, not q20 evidence."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

_path = Path(os.environ.get("Q20_CONTINUATION_TEST_DRIVER",
    str(Path(__file__).resolve().parents[1] / "scripts/q20_deadline_continuation.py")))
_spec = importlib.util.spec_from_file_location("q20_deadline_driver", _path)
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)


def state():
    from bayesfilter.inference.q20_production_config import protocol_template
    return {"config": protocol_template(), "campaign_limit": 100000., "spent_seconds": 30000.,
            "stages": {}, "attempts": [{"stage": driver.TUNING_STAGE, "status": "completed",
                                        "elapsed_seconds": 28000.}]}


def quote():
    return {"posterior_first_assessment_seconds": 20000.,
            "reference_first_assessment_seconds": 4000., "reserves": {"assessment": 1000.}}


def test_deadline_extension_preserves_total_funds_and_reserves():
    current = state()
    original = deepcopy(current)
    result = driver.allocation(current, quote(), 1000000., 100.)
    assert result["protected_downstream_seconds"] == 32200.
    assert result["additional_tuning_seconds"] == 37800.
    assert result["tuning_cumulative_limit_seconds"] == 65800.
    assert result["additional_campaign_seconds"] == 0.
    assert current == original
    clipped = driver.allocation(current, quote(), 1000., 100.)
    assert clipped["additional_tuning_seconds"] == 895.


def test_expired_or_unfunded_extension_does_not_launch():
    current = state()
    with pytest.raises(ValueError, match="no funded"):
        driver.allocation(current, quote(), 100., 100.)
    current["spent_seconds"] = 90000.
    with pytest.raises(ValueError, match="no funded"):
        driver.allocation(current, quote(), 1000000., 100.)
    current["attempts"][0]["status"] = "running"
    with pytest.raises(ValueError, match="unsettled"):
        driver.allocation(current, quote(), 1000000., 100.)


@pytest.mark.parametrize("status,reason,eligible", [
    ("ESTIMATION_BUDGET_PAUSED", "tune-neutra-beta1: partial_tuning_checkpointed", True),
    ("ESTIMATION_BUDGET_PAUSED", "sample-neutra: budget_paused", False),
    ("MASTER_INFRASTRUCTURE_FAILURE", "tune-neutra-beta1", False),
    ("STABLE_ESTIMATE_OBTAINED", "", False),
])
def test_only_tuning_budget_pause_permits_extension(status, reason, eligible):
    assert driver.eligible_predecessor({"status": status, "details": {"reason": reason}}) is eligible


def test_qualification_paths_reuse_only_identical_evidence(tmp_path):
    previous, fresh = tmp_path/"previous.json", tmp_path/"fresh.json"
    previous.write_text('{"checksum":"same"}\n')
    fresh.write_bytes(previous.read_bytes())
    attempt = tmp_path/"attempt"
    attempt.mkdir()
    (attempt/"request.json").write_text(json.dumps({"qualification_path": str(previous)}))
    campaign = SimpleNamespace(state={"attempts": [{"stage": "train-neutra", "directory": str(attempt)}]})
    result = driver.canonical_request(campaign, "train-neutra", {"qualification_path": str(fresh)})
    assert result["qualification_path"] == str(previous)
    fresh.write_text('{"checksum":"different"}\n')
    with pytest.raises(ValueError, match="qualification changed"):
        driver.canonical_request(campaign, "train-neutra", {"qualification_path": str(fresh)})


def test_prepare_campaign_preserves_ledger_and_rejects_unsettled_parent(tmp_path, monkeypatch):
    from bayesfilter.inference import q20_campaign_runtime
    from bayesfilter.inference.q20_production_config import digest
    current = state()
    current.update(status="ESTIMATION_BUDGET_PAUSED", details={"reason": driver.TUNING_STAGE},
                   config_hash=digest(current["config"]), sources={"test": "source"},
                   stage_limits={driver.TUNING_STAGE: 28800.})
    attempt = tmp_path/"old-attempt"
    checkpoint = attempt/"worker/data/tuning/tuning_checkpoint.json"
    driver.write_json(checkpoint, {"result": {"completion_status": "partial_budget"}})
    current["attempts"][0].update(directory=str(attempt), pid=None)
    receipt = tmp_path/"completed.json"
    driver.write_json(receipt, {"completed": True})
    current["stages"]["train-neutra"] = {"result_path": str(receipt),
        "result_sha256": driver.checksum(receipt), "artifact_hashes": {str(receipt): driver.checksum(receipt)}}
    parent = tmp_path/"parent"
    driver.write_json(parent/"campaign.json", current)
    driver.write_json(parent/"forecast-neutra.json", quote())
    parent_bytes = (parent/"campaign.json").read_bytes()
    job = {"parent_campaign": str(parent), "source_root": str(tmp_path),
        "config_hash": current["config_hash"], "campaign_limit_seconds": 100000.,
        "minimum_settled_seconds": 1000., "deadline_epoch": 1000000.,
        "deadline_local": "test deadline", "output_root": str(tmp_path/"new"),
        "plan_file": "test reference plan", "authority": "test fixture"}
    monkeypatch.setattr(q20_campaign_runtime, "source_snapshot", lambda root: current["sources"])
    monkeypatch.setattr(driver.time, "time", lambda: 100.)
    _, copied = driver.prepare_campaign(job)
    assert copied["attempts"] == current["attempts"]
    assert copied["stages"] == current["stages"]
    assert copied["spent_seconds"] == current["spent_seconds"]
    assert copied["campaign_limit"] == current["campaign_limit"]
    assert copied["config"] == current["config"]
    assert (parent/"campaign.json").read_bytes() == parent_bytes
    current["attempts"][0]["status"] = "running"
    driver.write_json(parent/"campaign.json", current)
    with pytest.raises(ValueError, match="unsettled"):
        driver.prepare_campaign({**job, "output_root": str(tmp_path/"another")})
    current["attempts"][0]["status"] = "completed"
    driver.write_json(parent/"campaign.json", current)
    receipt.write_text('{}')
    with pytest.raises(ValueError, match="receipt changed"):
        driver.prepare_campaign({**job, "output_root": str(tmp_path/"another")})


def test_real_checkpoint_extension_matches_uninterrupted_numerics(tmp_path):
    import os
    import subprocess
    import sys
    import tensorflow as tf
    from tests.test_q20_master_integration import protocol
    from tests.test_q20_production_repair import four_dimensional_bridge
    from bayesfilter.inference.q20_production_training import run_training_cohort
    from bayesfilter.inference.q20_production_hmc import tune_scope, sample_member
    from bayesfilter.inference.hmc_candidate_set_checkpoint import load_numerical_tuning_checkpoint
    config, bridge = protocol(), four_dimensional_bridge()
    trained = run_training_cohort(config, bridge, tmp_path/"train",
        memory_policy={"mode": "tiny_cpu_reference"}, max_seconds=120.,
        method="neutra", stop_when_trial_ready=True)
    cohort = driver.read_json(trained["checkpoint"])["cohort"]
    export = next(iter(cohort.values()))["exports"]["1.0"]
    partial = tune_scope(config, bridge, tmp_path/"partial", method="neutra",
                         training_export=export, max_work_items=1, max_seconds=120.)
    old_bytes = Path(partial["tuning_checkpoint"]).read_bytes()
    _, before = load_numerical_tuning_checkpoint(partial["tuning_checkpoint"], adapter=bridge.fixed_beta_adapter(1.))
    assert before.result().observations, "test must preserve already-computed evidence"
    request = {"stage": "tune", "method": "neutra", "beta": 1., "training_export": export,
        "resume_checkpoint": partial["tuning_checkpoint"], "cooperative_seconds": 240.,
        "deadline_extension": {"deadline_epoch": driver.time.time()+300.,
                               "driver_sha256": driver.checksum(driver.__file__)}}
    continued = driver.resume_extended_tuning(config, bridge, tmp_path/"continued", request)
    assert continued["status"] == "complete"
    assert continued["verified_members"]
    _, after = load_numerical_tuning_checkpoint(continued["tuning_checkpoint"], adapter=bridge.fixed_beta_adapter(1.))
    old_config, new_config = before.config.payload(), after.config.payload()
    assert new_config.pop("max_wall_time_seconds") > old_config.pop("max_wall_time_seconds")
    assert old_config == new_config
    assert after.result().budget_used_units >= before.result().budget_used_units
    assert after.result().search_state["elapsed_seconds"] >= before.result().search_state["elapsed_seconds"]
    assert Path(partial["tuning_checkpoint"]).read_bytes() == old_bytes
    # Exercise the actual child-process entry and unchanged worker bootstrap.
    from bayesfilter.inference import q20_production_config
    child_request = {**request, "config": config, "fixture": True,
        "deadline_extension": {**request["deadline_extension"],
            "source_root": str(Path(q20_production_config.__file__).resolve().parents[2])}}
    request_path = tmp_path/"worker-request.json"
    driver.write_json(request_path, child_request)
    child = subprocess.run([sys.executable, str(driver.__file__), "worker", "--request",
        str(request_path), "--output-dir", str(tmp_path/"worker")],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "TF_FORCE_GPU_ALLOW_GROWTH": "true"},
        capture_output=True, text=True, timeout=120.)
    assert child.returncode == 0, child.stderr[-6000:]
    child_result = driver.read_json(tmp_path/"worker/worker-result.json")
    assert child_result["result"]["verified_member_parameters"] == continued["verified_member_parameters"]
    assert driver.read_json(tmp_path/"worker/manifest.json")["gpu_intentionally_hidden"]
    baseline = tune_scope(config, bridge, tmp_path/"baseline", method="neutra", training_export=export)
    assert baseline["verified_member_parameters"] == continued["verified_member_parameters"]
    key = sorted(continued["verified_members"])[0]
    a = sample_member(config, bridge, tmp_path/"sample-baseline", member_path=baseline["verified_members"][key], label="same-seeds")
    b = sample_member(config, bridge, tmp_path/"sample-continued", member_path=continued["verified_members"][key], label="same-seeds")
    tf.debugging.assert_equal(a["private_retained_raw"], b["private_retained_raw"])
