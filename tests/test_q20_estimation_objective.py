"""Coordinator contracts: either NeuTra method may supply the estimate."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from bayesfilter.inference.q20_master_program import run_estimation_attempts, forecast_campaign
from bayesfilter.inference.q20_production_config import (
    protocol_template, validate_protocol, method_betas, training_cohort, digest,
)


def prices(config, method):
    return {"config_hash": digest(config), "method": method, "status": "estimation_method_cost_measurements",
        "hmc": [{"beta": beta, "kind": f"chart-w{width}", "L": length,
                 "steady_seconds": .0001, "first_seconds": .001}
                for beta in method_betas(config, method) for width in config["training"]["widths"]
                for length in (min(config["tuning"]["l_grid"]), max(config["tuning"]["l_grid"]))],
        "training_quote": {"minimum_cohort_seconds": 1., "full_training_cap_seconds": 2.},
        "missing_cost_categories": [], "worker_initialization_seconds": .001,
        "reference_batch_first_seconds": .001, "reference_batch_steady_seconds": .0001,
        "reference_analysis_seconds_per_row": .000001, "posterior_analysis_seconds": .001,
        "exchange_price_role": "actual_multi_chart_mixture" if method == "ensemble" else "not_required",
        "exchange_charts_per_temperature": config["ensemble"]["charts"] if method == "ensemble" else 0,
        "exchange_first_seconds": .002 if method == "ensemble" else None,
        "exchange_steady_seconds": .0002 if method == "ensemble" else None}


def harness(tmp_path, *, plain_failure=None, reference_valid=True, worker_error=False):
    config = protocol_template()
    config.update(role="smoke", jit_compile=False, cpu_reference=True)
    campaign = SimpleNamespace(config=config, root=tmp_path,
        state={"campaign_limit": 10000., "stages": {}, "attempts": []}, remaining=lambda: 10000.)
    calls = []

    def stage(name, request, **kwargs):
        calls.append((name, request))
        folder = tmp_path / name
        folder.mkdir()
        method = request.get("method", "ensemble" if "ensemble" in name else "neutra")
        kind = request["stage"]
        plain = "ensemble" not in name
        if kind == "price":
            result = prices(config, method)
        elif kind == "train":
            cohort = {}
            for candidate in training_cohort(config, method=method):
                exports = {}
                if not (plain and plain_failure == "train"):
                    for beta in method_betas(config, method):
                        path = folder / f"{candidate['id']}-{beta}.json"
                        path.write_text(json.dumps({"assessment": {"map_reliability": {"passed": True},
                                                                   "decision": {"development_eligible": True}}}))
                        exports[str(beta)] = str(path)
                cohort[candidate["id"]] = {"exports": exports}
            path = folder / "checkpoint.json"
            path.write_text(json.dumps({"cohort": cohort}))
            result = {"checkpoint": str(path), "method": method, "method_complete": True}
        elif kind == "tune":
            assert request["method"] == "neutra"
            result = {"status": "complete", "verified_members": {} if plain and plain_failure == "tune"
                      else {"verified": str(folder / "member.json")}}
        elif kind in {"sample", "ensemble"}:
            if worker_error:
                raise ValueError("corrupt shared target evidence")
            result = {"summary": {"sequential_declared_checks_passed": not (plain and plain_failure == "sample")}}
        elif kind == "reference":
            result = {"qualified": reference_valid}
        elif kind == "assess":
            passed = reference_valid and not (plain and plain_failure == "assess")
            result = {"passed": passed, "status": "estimate_validated" if passed else
                      "reference_unqualified" if not reference_valid else "candidate_failed_checks"}
        else:
            raise AssertionError("unpermitted stage: " + kind)
        path = folder / "result.json"
        path.write_text(json.dumps(result))
        return {"result": result, "result_path": str(path), "wall_seconds": .001, "supervisor_seconds": .001}

    # The real supervisor reuses its completed reference for the second method.
    recorded_stage = stage
    cache = {}
    def cached_stage(name, request, **kwargs):
        if name not in cache:
            cache[name] = recorded_stage(name, request, **kwargs)
            campaign.state["stages"][name] = {}
        return cache[name]

    def finish(status, **details):
        return {"status": status, "details": details}
    return campaign, cached_stage, finish, calls


def test_protocol_only_permits_estimation_and_no_classical_preparation():
    config = protocol_template()
    validate_protocol(config)
    assert "comparison" not in config
    assert config["estimation"]["methods"] == ["neutra", "ensemble"]
    assert method_betas(config, "neutra") == [1.]
    assert {c["schedule"] for c in training_cohort(config, method="neutra")} == {"direct"}
    changed = copy.deepcopy(config)
    changed["estimation"]["methods"].append("classical")
    with pytest.raises(ValueError, match="first valid estimate"):
        validate_protocol(changed)
    changed = copy.deepcopy(config)
    changed["schema"] = "bayesfilter.q20.production_protocol.v2"
    with pytest.raises(ValueError, match="historical comparison protocol"):
        validate_protocol(changed)


def test_plain_success_does_not_launch_ensemble_or_comparison(tmp_path):
    campaign, stage, finish, calls = harness(tmp_path)
    result = run_estimation_attempts(campaign, stage, finish)
    assert result["details"]["method"] == "neutra"
    assert [r["stage"] for _, r in calls] == ["price", "train", "tune", "sample", "reference", "assess"]
    assert all(r.get("beta", 1.) == 1. for _, r in calls)


def test_completed_calibration_is_reused_in_estimation_pricing_and_training(tmp_path):
    campaign, stage, finish, calls = harness(tmp_path)
    saved = tmp_path/"calibration-result.json"
    checkpoint = tmp_path/"saved-calibration.json"
    checkpoint.write_text(json.dumps({"cohort": {}}))
    saved.write_text(json.dumps({"result": {"checkpoint": str(checkpoint)}}))
    campaign.state["stages"]["calibration"] = {"result_path": str(saved),
        "result_sha256": hashlib.sha256(saved.read_bytes()).hexdigest()}
    run_estimation_attempts(campaign, stage, finish)
    priced = next(request for _, request in calls if request["stage"] == "price")
    trained = next(request for _, request in calls if request["stage"] == "train")
    assert priced["training_checkpoint"] == str(checkpoint)
    assert trained["resume_checkpoint"] == str(checkpoint)


@pytest.mark.parametrize("mode", ["refresh", "repair", "checkpoint-repair"])
def test_retired_classical_phases_fail_before_creating_campaign(tmp_path, mode):
    from bayesfilter.inference.q20_master_program import execute_master
    root = tmp_path/"forbidden"
    with pytest.raises(ValueError, match="classical-preparation phases are retired"):
        execute_master(protocol_template(), root, repo=tmp_path, stop_after=mode)
    assert not root.exists()


@pytest.mark.parametrize("failure", ["train", "tune", "sample", "assess"])
def test_candidate_failure_can_be_repaired_by_ensemble(tmp_path, failure):
    campaign, stage, finish, calls = harness(tmp_path, plain_failure=failure)
    result = run_estimation_attempts(campaign, stage, finish)
    assert result["status"] == "SMOKE_ESTIMATION_CHECKS_PASSED"
    assert result["details"]["method"] == "ensemble"
    assert len(result["details"]["attempts"]) == 2
    ensemble = next(r for _, r in calls if r["stage"] == "ensemble")
    assert set(ensemble["members_by_beta"]) == {"0.5", "1.0"}
    assert all(len(paths) == 2 for paths in ensemble["members_by_beta"].values())
    assert all(r.get("method", "neutra") in {"neutra", "ensemble"} for _, r in calls)
    assert sum(r["stage"] == "reference" for _, r in calls) == 1


def test_reference_failure_requires_reference_repair_not_another_sampler(tmp_path):
    campaign, stage, finish, calls = harness(tmp_path, reference_valid=False)
    result = run_estimation_attempts(campaign, stage, finish)
    assert result["status"] == "REFERENCE_REPAIR_REQUIRED"
    assert all("ensemble" not in name for name, _ in calls)


def test_shared_invalidity_does_not_turn_into_candidate_fallback(tmp_path):
    campaign, stage, finish, calls = harness(tmp_path, worker_error=True)
    with pytest.raises(ValueError, match="corrupt shared target"):
        run_estimation_attempts(campaign, stage, finish)
    assert all("ensemble" not in name for name, _ in calls)


def test_master_interruption_records_terminal_state(tmp_path, monkeypatch):
    from pathlib import Path
    from bayesfilter.inference.q20_campaign_runtime import Campaign
    from bayesfilter.inference.q20_master_program import execute_master
    config = protocol_template()
    config.update(role="smoke", cpu_reference=True, jit_compile=False)
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(Campaign, "numerical_stage", interrupted)
    root = tmp_path/"interrupted"
    with pytest.raises(KeyboardInterrupt):
        execute_master(config, root, repo=Path(__file__).resolve().parents[1], fixture=True,
            allowance={"campaign_remaining_seconds": 1200., "diagnostic_remaining_seconds": 300.})
    assert json.loads((root/"result.json").read_text())["status"] == "MASTER_INTERRUPTED"
    assert json.loads((root/"campaign.json").read_text())["status"] == "MASTER_INTERRUPTED"


def test_forecast_only_charges_current_method_and_rejects_proxy_ensemble_price():
    config = protocol_template()
    plain = forecast_campaign(config, prices(config, "neutra"))
    ensemble_prices = prices(config, "ensemble")
    ensemble = forecast_campaign(config, ensemble_prices)
    assert plain["scope_count"] == 1
    assert ensemble["scope_count"] == 4
    assert set(plain["reserves"]) == {"tuning", "posterior", "reference", "assessment", "localized_repair"}
    ensemble_prices["exchange_price_role"] = "single_chart_exchange"
    with pytest.raises(ValueError, match="actual multi-chart"):
        forecast_campaign(config, ensemble_prices)
    incomplete = prices(config, "neutra")
    incomplete["hmc"].pop()
    with pytest.raises(ValueError, match="missing or unexpected scopes"):
        forecast_campaign(config, incomplete)


def test_forecast_reports_missing_training_diagnostic_pricing():
    config = protocol_template()
    pricing = prices(config, "neutra")
    assert forecast_campaign(config, pricing)["full_campaign_priced"] is True
    pricing["training_quote"]["missing_training_scopes"] = [[16, 32, 1.]]
    forecast = forecast_campaign(config, pricing)
    assert forecast["full_campaign_priced"] is False
    assert forecast["status"] == "partial_downstream_forecast"
    assert forecast["missing_cost_categories"] == ["training"]
    assert pricing["missing_cost_categories"] == []
