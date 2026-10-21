"""Budget and launch-boundary tests for already authorized research work."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import continue_modern_hmc_priced_comparison as continuation


def report():
    return dict(status="audited_cost_diagnostic", actual_price_worker_seconds=80.,
        rows=[dict(method=method, root_count=4 if method == "mams" else 8,
            design_file="design.json", design_sha256="sha",
            scenarios=[dict(load_multiplier=load, minimum_path_seconds=100.*load,
                            maximum_count_scenario_seconds=400.*load) for load in (1., 2., 4.)])
            for method in continuation.METHODS])


def balance():
    return dict(prior_unallocated_worker_seconds=1000., unallocated_worker_seconds=700.,
                reservations=dict(gpu_prices=100., local_checks=200.))


def test_reserves_are_not_released_or_charged_twice():
    result = continuation.allocate(report(), balance())
    assert result["child_max_wall_seconds"] == 700.
    assert result["budget_interpretation"] == "bounded_feasibility"
    assert result["unallocated_after_child_reservation_seconds"] == 0.
    assert result["local_reservation_still_held_seconds"] == 200.
    assert result["pricing_reservation_release_deferred"] is True
    assert result["unfunded_methods"] == list(continuation.METHODS[:-1])
    assert result["future_replicates_funded"] is False


def test_supervisor_overrun_reduces_new_allocation():
    prices = report()
    prices["actual_price_worker_seconds"] = 125.
    assert continuation.allocate(prices, balance())["child_max_wall_seconds"] == 675.


def test_shorter_full_count_scenario_releases_unused_child_allocation():
    prices = report()
    for row in prices["rows"][-1]["scenarios"]:
        row["maximum_count_scenario_seconds"] = 120.*row["load_multiplier"]
    result = continuation.allocate(prices, balance())
    assert result["child_max_wall_seconds"] == 480.
    assert result["unallocated_after_child_reservation_seconds"] == 220.
    assert result["budget_interpretation"] == "cost_scenarios_fit"
    assert result["resource_censoring_possible"] is True


def test_conservative_forecast_does_not_veto_bounded_work_or_reduce_members():
    prices = report()
    prices["rows"][-1]["scenarios"][0]["minimum_path_seconds"] = 800.
    result = continuation.allocate(prices, balance())
    assert result["status"] == "allocated"
    assert result["child_max_wall_seconds"] == 700.
    assert result["minimum_count_scenario_fits"] is False
    assert result["cost_forecast_is_continuation_veto"] is False
    assert result["planned_root_count"] == 4
    assert result["complete_frozen_cohort_preserved"] is True
    assert result["full_completion_guaranteed"] is False
    assert result["budget_interpretation"] == "bounded_feasibility"


def test_actual_exhausted_budget_prevents_launch():
    prices = report()
    prices["actual_price_worker_seconds"] = 850.
    result = continuation.allocate(prices, balance())
    assert result["status"] == "budget_exhausted"
    assert result["child_max_wall_seconds"] == 0.
    assert result["unallocated_after_child_reservation_seconds"] == 0.
    assert result["budget_interpretation"] == "budget_exhausted"


@pytest.mark.parametrize("mutation", ["unsettled", "omitted_method", "omitted_mams_root", "missing_load",
                                      "double_spend", "nan_balance", "nan_price"])
def test_invalid_accounting_or_inventory_stops_dependent_launch(mutation):
    prices, ledger = report(), balance()
    if mutation == "unsettled": prices["status"] = "running"
    elif mutation == "omitted_method": prices["rows"].pop(0)
    elif mutation == "omitted_mams_root": prices["rows"][-1]["root_count"] = 3
    elif mutation == "missing_load": prices["rows"][-1]["scenarios"].pop()
    elif mutation == "double_spend": ledger["unallocated_worker_seconds"] = 900.
    elif mutation == "nan_balance": ledger["unallocated_worker_seconds"] = float("nan")
    elif mutation == "nan_price": prices["rows"][-1]["scenarios"][0]["minimum_path_seconds"] = float("nan")
    with pytest.raises(ValueError):
        continuation.allocate(prices, ledger)


@pytest.mark.parametrize("child_status,exit_code", [("completed", 0), ("resource_stopped", 1)])
@pytest.mark.parametrize("minimum_forecast", [100., 800.])
@pytest.mark.parametrize("method", ["mams", "fixed_hmc"])
def test_continuation_uses_snapshot_child_and_preserves_terminal_outcomes(
        tmp_path, monkeypatch, child_status, exit_code, minimum_forecast, method):
    campaign = tmp_path / "campaign"
    (campaign / "prices-01").mkdir(parents=True)
    source = tmp_path / "immutable-source"
    source.mkdir()
    design = campaign / f"{method}.json"
    design.write_text(json.dumps(dict(method=method)))
    (campaign / "prices-01/result.json").write_text(json.dumps(dict(status="completed",
        supervisor_elapsed_seconds=80., source_root=str(source), numerical_source=dict(git_commit="frozen"))))
    (campaign / "allocation-01.json").write_text(json.dumps(balance()))
    audit = tmp_path / "deployment-audit.json"
    audit.write_text(json.dumps(dict(status="passed", gpu_supervisor_seconds=60.,
        native_work=dict(native_call_audit=[dict(native_seconds=40.)]))))
    prices = report()
    row = next(row for row in prices["rows"] if row["method"] == method)
    row["scenarios"][0]["minimum_path_seconds"] = minimum_forecast
    row.update(design_file=str(design), design_sha256=continuation.digest(design))
    monkeypatch.setattr(continuation, "build_report", lambda *a, **kw: deepcopy(prices))
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    monkeypatch.setitem(sys.modules, "scripts.price_modern_hmc_method_proposals",
                        SimpleNamespace(select_device=lambda gpu: ("1", dict(status="SHARED_ADMITTED"))))
    launched = []
    def fake_run(command, *, cwd, env, stdout, stderr):
        launched.append(command)
        assert Path(command[1]) == source / "scripts/run_modern_hmc_frozen_full_model.py"
        assert cwd == source
        assert env["CUDA_VISIBLE_DEVICES"] == "1" and env["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
        assert float(command[-1]) == 700.
        output = Path(command[command.index("--output")+1])
        output.mkdir()
        (output / "result.json").write_text(json.dumps(dict(status=child_status)))
        return SimpleNamespace(returncode=exit_code)
    monkeypatch.setattr(continuation.subprocess, "run", fake_run)
    output = tmp_path / "continuation"
    argv = ["continue", "--campaign", str(campaign), "--output", str(output),
            "--deployment-audit", str(audit), "--method", method]
    if method != "mams":
        argv += ["--completed-mams", str(campaign / "prior")]
        monkeypatch.setattr(continuation, "settle_completed_mams", lambda *a, **kw: balance())
    monkeypatch.setattr(sys, "argv", argv)
    assert continuation.main() == exit_code
    assert len(launched) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == child_status
    assert manifest["child_exit_code"] == exit_code
    assert manifest["child_worker_seconds"] >= 0.
    allocation = json.loads((output / "allocation.json").read_text())
    assert allocation["scope"].endswith(f"bounded_{method}_feasibility")
    assert allocation["complete_frozen_cohort_preserved"] is True
    assert allocation["minimum_count_scenario_fits"] is (minimum_forecast <= 700.)
    assert manifest["price_status"] == "completed"
    assert Path(manifest["child_result"]).parent.name == f"{method}-run"


def completed_campaign(tmp_path):
    campaign = tmp_path / "campaign"
    prior = campaign / "continuation-02"
    (prior / "mams-run/posterior").mkdir(parents=True)
    (campaign / "prices-01").mkdir()
    def write(path, value):
        path.write_text(json.dumps(value))
    write(campaign / "allocation-01.json", balance())
    write(campaign / "prices-01/result.json", dict(status="completed", supervisor_elapsed_seconds=80.))
    prices = report()
    prices["prices_sha256"] = continuation.digest(campaign / "prices-01/result.json")
    write(prior / "price-audit.json", prices)
    write(prior / "allocation.json", continuation.allocate(prices, balance()))
    write(prior / "manifest.json", dict(status="completed", child_exit_code=0, child_worker_seconds=300.))
    write(prior / "mams-run/manifest.json", dict(status="completed"))
    write(prior / "mams-run/result.json", dict(status="completed", method="mams", completion_status="complete"))
    write(prior / "mams-run/posterior/result.json", dict(status="completed"))
    files = [prior / name for name in ("manifest.json", "mams-run/manifest.json",
             "mams-run/result.json", "mams-run/posterior/result.json")]
    write(prior / "mams-terminal-audit-01.json", dict(status="passed", child_worker_seconds=300.,
        source_sha256={str(path.relative_to(campaign)): continuation.digest(path) for path in files}))
    return campaign, prior


def test_terminal_settlement_charges_actual_work_and_holds_both_cpu_reserves(tmp_path):
    campaign, prior = completed_campaign(tmp_path)
    settled = continuation.settle_completed_mams(campaign, prior, cpu_reservation_seconds=100.)
    assert settled["unallocated_worker_seconds"] == 320.  # 1000 - 300 - 80 - 200 - 100
    allocated = continuation.allocate(report(), settled, method="fixed_hmc")
    assert allocated["child_max_wall_seconds"] == 320.
    assert allocated["planned_root_count"] == 8
    assert allocated["unfunded_methods"] == ["randomized_hmc", "snaper"]
    assert allocated["pricing_reservation_release_deferred"] is False


@pytest.mark.parametrize("mutation", ["running", "changed_price", "changed_charge", "omitted_audit_file",
                                     "inconsistent_allocation", "negative_cpu", "nan_cpu", "overspend"])
def test_invalid_or_unsettled_charge_cannot_fund_a_comparator(tmp_path, mutation):
    campaign, prior = completed_campaign(tmp_path)
    cpu = 100.
    if mutation in {"negative_cpu", "nan_cpu", "overspend"}:
        cpu = {"negative_cpu": -1., "nan_cpu": float("nan"), "overspend": 1000.}[mutation]
    else:
        path = prior / {"running": "manifest.json", "changed_charge": "manifest.json",
            "omitted_audit_file": "mams-terminal-audit-01.json", "inconsistent_allocation": "allocation.json",
            "changed_price": "../prices-01/result.json"}[mutation]
        data = json.loads(path.read_text())
        if mutation == "running": data["status"] = "running"
        elif mutation == "changed_charge": data["child_worker_seconds"] = 1.
        elif mutation == "changed_price": data["supervisor_elapsed_seconds"] = 1.
        elif mutation == "omitted_audit_file": data["source_sha256"].pop(next(iter(data["source_sha256"])))
        elif mutation == "inconsistent_allocation": data["child_max_wall_seconds"] = 9999.
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        continuation.settle_completed_mams(campaign, prior, cpu_reservation_seconds=cpu)


@pytest.mark.parametrize("method", ["fixed_hmc", "randomized_hmc"])
def test_comparator_allocation_preserves_eight_roots_even_when_forecast_exceeds_balance(method):
    prices = report()
    row = next(row for row in prices["rows"] if row["method"] == method)
    row["scenarios"][0]["minimum_path_seconds"] = 900.
    assert continuation.allocate(prices, balance(), method=method)["planned_root_count"] == 8
    row["root_count"] = 7
    with pytest.raises(ValueError, match="complete method cohort"):
        continuation.allocate(prices, balance(), method=method)
