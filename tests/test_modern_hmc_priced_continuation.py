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


def test_insufficient_minimum_envelope_does_not_launch_or_reduce_members():
    prices = report()
    prices["rows"][-1]["scenarios"][0]["minimum_path_seconds"] = 800.
    result = continuation.allocate(prices, balance())
    assert result["status"] == "under_budgeted"
    assert result["child_max_wall_seconds"] == 0.


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
def test_continuation_uses_snapshot_child_and_preserves_terminal_outcomes(tmp_path, monkeypatch, child_status, exit_code):
    campaign = tmp_path / "campaign"
    (campaign / "prices-01").mkdir(parents=True)
    source = tmp_path / "immutable-source"
    source.mkdir()
    design = campaign / "mams.json"
    design.write_text('{"method": "mams"}')
    (campaign / "prices-01/result.json").write_text(json.dumps(dict(status="completed",
        supervisor_elapsed_seconds=80., source_root=str(source), numerical_source=dict(git_commit="frozen"))))
    (campaign / "allocation-01.json").write_text(json.dumps(balance()))
    audit = tmp_path / "deployment-audit.json"
    audit.write_text(json.dumps(dict(status="passed", gpu_supervisor_seconds=60.,
        native_work=dict(native_call_audit=[dict(native_seconds=40.)]))))
    prices = report()
    prices["rows"][-1].update(design_file=str(design), design_sha256=continuation.digest(design))
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
    monkeypatch.setattr(sys, "argv", ["continue", "--campaign", str(campaign), "--output", str(output),
                                      "--deployment-audit", str(audit)])
    assert continuation.main() == exit_code
    assert len(launched) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == child_status
    assert manifest["child_exit_code"] == exit_code
    assert manifest["child_worker_seconds"] >= 0.
    assert json.loads((output / "allocation.json").read_text())["scope"].endswith("complete_adapted_mams_cohort")
