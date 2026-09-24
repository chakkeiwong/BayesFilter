"""CPU/reference checks of historical recovery, never q20 learning evidence."""
import copy
import json
from pathlib import Path

import pytest

from bayesfilter.inference.q20_checkpoint_migration import audit_migration, migrate_training_checkpoint
from bayesfilter.inference.q20_production_config import digest
from bayesfilter.inference.q20_training_resume import read_training_checkpoint
from tests.test_q20_production_repair import four_dimensional_bridge
from tests.test_q20_master_integration import protocol

REPO = Path(__file__).resolve().parents[1]


def historical(config):
    old = copy.deepcopy(config)
    old["schema"] = "bayesfilter.q20.production_protocol.v2"
    old["comparison"] = {**old.pop("assessment"), "methods": ["identity", "neutra"], "confirmation_replicates": 3}
    old.pop("estimation")
    return old


@pytest.mark.parametrize("field", ["target", "training", "seed", "assessment"])
def test_migration_rejects_scientific_scope_changes_before_loading_state(tmp_path, field):
    config = protocol()
    old = historical(config)
    if field == "target":
        config[field]["prior_sd"] = 3.
    elif field == "training":
        config[field]["learning_rates"] = [.002]
    elif field == "seed":
        config[field][0] += 1
    else:
        config[field]["mean_margin_sd"] *= 2
    with pytest.raises(ValueError):
        audit_migration(config, old, tmp_path/"missing.json", REPO, REPO)


def test_migration_restores_state_without_old_admission_or_floor_credit(tmp_path):
    from bayesfilter.inference.q20_production_training import run_training_cohort, training_config
    from bayesfilter.inference.neutra_training_protocol import TrainingSession
    from bayesfilter.inference.q20_campaign_costs import training_reservation
    from tests.test_q20_campaign_costs import measured_row
    config, bridge = protocol(), four_dimensional_bridge()
    old = historical(config)
    memory = {"mode": "tiny_cpu_reference"}
    trained = run_training_cohort(old, bridge, tmp_path/"old", memory_policy=memory,
        max_seconds=120., calibration_only=True)
    original = Path(trained["checkpoint"]).read_bytes()
    prior = read_training_checkpoint(trained["checkpoint"], old)
    migrated = migrate_training_checkpoint(config, bridge, tmp_path/"migrated",
        previous_config=old, checkpoint=trained["checkpoint"], previous_root=REPO,
        memory_policy=memory, reference_bridge=bridge)
    state = read_training_checkpoint(migrated["checkpoint"], config)
    assert Path(trained["checkpoint"]).read_bytes() == original
    assert not (tmp_path/"migrated/validation-cache").exists()
    assert migrated["source_import"]["current_scope_update_credit"] == 0
    for name, item in state["cohort"].items():
        before, after = prior["cohort"][name]["session"], item["session"]
        assert not item["assessments"] and not item["exports"] and item["plateaus"] == 0
        assert after["level_updates"] == 0
        assert after["parent_hash"] == before["state_hash"]
        assert after["map"]["variables"] == before["map"]["variables"]
        for key in ("optimizer", "rng_index", "root_seed", "iteration", "history"):
            assert after[key] == before[key]
        # With the same numerical fixture, the next Adam update is exactly
        # preserved; scope migration must not silently restart optimizer/RNG.
        args = dict(bridge=bridge, config=training_config(config, after["scope"]["candidate"]), preflight_seed=(91, 3))
        a = TrainingSession.restore(before, expected_scope=before["scope"], **args)
        b = TrainingSession.restore(after, expected_scope=after["scope"], **args)
        a.advance(1)
        b.advance(1)
        assert a.checkpoint()["map"]["variables"] == b.checkpoint()["map"]["variables"]
        assert a.checkpoint()["optimizer"] == b.checkpoint()["optimizer"]
    rows = [{**measured_row(4, b), "batch_size": 8} for b in (.5, 1.)]
    quote = training_reservation(config, rows, checkpoint=migrated["checkpoint"])
    assert quote["scenarios"]["floor_first_bank"]["credited_updates"] == 0
    assert quote["scenarios"]["floor_first_bank"]["target_validation_rows"] > training_reservation(config, rows)["scenarios"]["floor_first_bank"]["target_validation_rows"]
    resumed = run_training_cohort(config, bridge, tmp_path/"continued", memory_policy=memory,
        max_seconds=120., resume=migrated["checkpoint"], method="neutra")
    current = read_training_checkpoint(resumed["checkpoint"], config)
    assert resumed["method_complete"]
    for name, item in current["cohort"].items():
        if name.startswith("direct"):
            assert item["session"]["level_updates"] >= config["training"]["cohort_min_updates"]
            assert item["exports"]
        else:
            assert item == state["cohort"][name]


def test_master_migrates_in_supervised_worker_and_replays_phase(tmp_path):
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    from bayesfilter.inference.q20_master_program import execute_master
    from bayesfilter.inference.q20_production_training import run_training_cohort
    config = protocol()
    config["reference"].update(banks=4, rungs=[64,128], batch_size=32, ess_min=1., minimum_tail_rows=1)
    old = historical(config)
    trained = run_training_cohort(old, four_dimensional_bridge(), tmp_path/"old",
        memory_policy=configure_tensorflow_gpu_memory_growth(tf, require_gpu=False), max_seconds=120., calibration_only=True)
    kwargs = dict(repo=REPO, fixture=True, training_checkpoint=trained["checkpoint"],
        previous_source_root=REPO, previous_config=old,
        allowance={"campaign_remaining_seconds": 7200., "diagnostic_remaining_seconds": 3600.})
    root = tmp_path/"campaign"
    result = execute_master(config, root, stop_after="migrate", **kwargs)
    assert result["status"] == "TRAINING_MIGRATION_COMPLETE"
    ledger = json.loads((root/"campaign.json").read_text())
    assert set(ledger["stages"]) == {"migrate-training"}
    assert execute_master(config, root, stop_after="migrate", **kwargs) == result
    assert json.loads((root/"campaign.json").read_text())["spent_seconds"] == ledger["spent_seconds"]
    finished = execute_master(config, root, **kwargs)
    # The allowance equals the protected repair reserve. Migration and pricing
    # consume it, so continuation must preserve the reserve rather than start
    # an unfunded training/assessment phase.
    assert finished["status"] == "ESTIMATION_BUDGET_PAUSED", finished
    ledger = json.loads((root/"campaign.json").read_text())
    assert "price-neutra" in ledger["stages"] and "train-neutra" not in ledger["stages"]
    assert "price-ensemble" not in ledger["stages"]
    changed = copy.deepcopy(old)
    changed["budget"]["campaign_remaining_seconds"] += 1
    with pytest.raises(ValueError, match="configuration/checksum|inputs changed"):
        execute_master(config, root, **{**kwargs, "previous_config": changed})


@pytest.mark.parametrize("method", ["neutra", "ensemble"])
def test_checkpoint_pricing_stops_before_downstream_when_training_reserve_exceeds_budget(tmp_path, monkeypatch, method):
    from bayesfilter.inference.q20_production_training import run_training_cohort
    from bayesfilter.inference.q20_master_stages import dispatch
    from bayesfilter.inference import q20_production_hmc
    config, bridge = protocol(), four_dimensional_bridge()
    memory = {"mode": "tiny_cpu_reference"}
    trained = run_training_cohort(config, bridge, tmp_path/"trained", memory_policy=memory,
        max_seconds=120., calibration_only=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("unaffordable training must stop before downstream HMC pricing")
    monkeypatch.setattr(q20_production_hmc, "draw_start_bank", forbidden)
    result = dispatch(config, bridge, tmp_path/"price", {"stage": "price", "method": method,
        "training_checkpoint": trained["checkpoint"], "reservation_limit_seconds": 1e-9}, memory)
    assert result["status"] == "unaffordable_under_declared_reservation"
    assert result["training_quote"]["checkpoint_credit_source"] == trained["checkpoint"]
    assert result["training_quote"]["scenarios"]["floor_first_bank"]["credited_updates"] > 0
    assert not list((tmp_path/"price").glob("hmc-*.json"))
