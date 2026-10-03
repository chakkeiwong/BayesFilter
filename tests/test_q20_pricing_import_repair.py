"""CPU-only engineering checks for historical ensemble pricing and continuation."""
import hashlib
import json

import pytest


def test_multitemperature_import_preserves_timing_and_map_provenance(tmp_path, monkeypatch):
    from tests.test_q20_master_integration import protocol
    from tests.test_q20_production_repair import four_dimensional_bridge
    from bayesfilter.inference.q20_master_stages import dispatch
    from bayesfilter.inference import q20_pricing, q20_production_training
    from bayesfilter.inference.hmc import ReusableFullChainHMCRunner

    config, bridge = protocol(), four_dimensional_bridge()
    memory = {"mode": "tiny_cpu_reference"}
    history = dispatch(config, bridge, tmp_path / "original",
                       {"stage": "price", "method": "ensemble"}, memory)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}")
    origin = {"result_path": str(tmp_path / "original/result.json"),
              "result_sha256": "fixture", "source_root": "fixture",
              "timing_quality": "resource_history_unknown"}
    monkeypatch.setattr(q20_pricing, "checked_historical_price", lambda *a: (history, origin))

    def forbidden(*args, **kwargs):
        pytest.fail("completed training or HMC timing was recomputed")

    monkeypatch.setattr(q20_production_training, "price_training", forbidden)
    monkeypatch.setattr(ReusableFullChainHMCRunner, "run", forbidden)
    result = dispatch(config, bridge, tmp_path / "imported", {"stage": "price",
        "method": "ensemble", "historical_pricing": str(receipt_path)}, memory)
    assert {row["beta"] for row in result["hmc"]} == {.5, 1.}
    assert result["historical_timing_origin"] == origin
    assert all(row["timing_origin"] == origin for row in result["hmc"])
    assert all(row["role"] == "initialization_map_pricing_only" for row in result["map_prices"])
    assert result["exchange_first_seconds"] > 0
    records = [json.loads(p.read_text()) for p in (tmp_path / "imported/pricing-ledger").glob("*.json")
               if p.name != "identity.json"]
    imported = [row for row in records if not row["measured_here"]]
    assert {row["key"] for row in imported} >= {"training", "reference", "analysis"}
    assert all(row["origin"] == origin and row["timing_quality"] == origin["timing_quality"]
               for row in imported)
    assert [row["key"] for row in records if row["measured_here"]] == ["exchange"]


def failed_tuning(tmp_path, config, *, status="complete", verified=False):
    from bayesfilter.inference.q20_production_config import digest
    checkpoint = tmp_path / "tuning.json"
    checkpoint.write_text(json.dumps({"result": {"completion_status": status,
        "verified_candidate_ids": ["one"] if verified else [],
        "candidate_states": {"one": "verified" if verified else "promotion_failed"}}}))
    result = tmp_path / "worker.json"
    result.write_text(json.dumps({"completed": True, "result": {"method": "neutra", "beta": 1.,
        "config_hash": digest(config), "status": status, "tuning_checkpoint": str(checkpoint),
        "verified_members": {"one": "member"} if verified else {}}}))
    return {"stage_receipt": {"result_path": str(result),
        "result_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
        "artifact_hashes": {str(checkpoint): hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}}


def test_completed_plain_rejection_preserved_while_refreshed_maps_get_fresh_trials(tmp_path):
    from tests.test_q20_estimation_objective import harness
    from bayesfilter.inference.q20_master_program import run_estimation_attempts
    campaign, stage, finish, calls = harness(tmp_path)
    campaign.state["recovery_plan"] = {"completed_tuning_failures": {
        "neutra": failed_tuning(tmp_path, campaign.config)}}
    result = run_estimation_attempts(campaign, stage, finish)
    assert result["details"]["method"] == "neutra"
    assert calls[0][0] == "price-neutra"
    assert any(request["stage"] == "train" and request["method"] == "neutra"
               for _, request in calls)
    assert any(request["stage"] == "tune" and request["method"] == "neutra"
               for _, request in calls)
    assert not any("ensemble" in name for name, _ in calls)
    assert result["details"]["attempts"][0]["status"] == "tuning_candidate_failed"


@pytest.mark.parametrize("status,verified", [("partial_budget", False), ("complete", True)])
def test_unfinished_or_successful_tuning_is_not_imported_as_failure(tmp_path, status, verified):
    from bayesfilter.inference.q20_master_program import checked_completed_tuning_failure
    from bayesfilter.inference.q20_production_config import protocol_template
    config = protocol_template()
    saved = failed_tuning(tmp_path, config, status=status, verified=verified)
    with pytest.raises(ValueError, match="completed plain tuning rejection"):
        checked_completed_tuning_failure(config, "neutra", saved)


def test_changed_negative_evidence_is_rejected(tmp_path):
    from bayesfilter.inference.q20_master_program import checked_completed_tuning_failure
    from bayesfilter.inference.q20_production_config import protocol_template
    config = protocol_template()
    saved = failed_tuning(tmp_path, config)
    (tmp_path / "tuning.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        checked_completed_tuning_failure(config, "neutra", saved)
