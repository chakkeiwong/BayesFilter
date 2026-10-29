"""Main funding uses full prices and preserves original scientific work."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from scripts import run_hmc_ssm_priced_main as main
from bayesfilter.testing.inference_validation.storage import write_json, read_json, file_hash


def inventory(design):
    rows = []
    for case in range(8):
        for slot in range(4):
            d = design("accuracy", "gaussian", "ordinary", replications=1, budget_seconds=2100.,
                options={"campaign_case": f"K{case}", "data": [slot], "dataset_id": str(slot),
                    "data_seed": [slot, 0], "data_version": str(slot),
                    "isolate_fits": True,
                    "fit_process_timeout_seconds": 2090., "timeout_policy": {
                        "max_extension_seconds": 0., "gpu_admission_mode": "shared"}})
            rows.append(replace(d, design_id=f"K{case}-slot-{slot}", seed=slot).payload())
    return {"designs": rows}, {"gaussian": [{"reference_checked": True}]}


def test_whole_lane_reallocation_preserves_32_slots_and_each_numerical_design(design):
    template, datasets = inventory(design)
    prices = {"K0": {"status": "complete_declared_workload", "price_seconds": 2241.},
              "K2": {"status": "complete_declared_workload", "price_seconds": 4841.},
              "K4": {"status": "complete_declared_workload", "price_seconds": 2533.}}
    original = deepcopy(template)
    a = main.allocate_lanes(template, datasets, prices, 9 * 3600.)
    assert a["original_main_denominator"] == len(a["dispositions"]) == 32
    assert [j["design"]["options"]["campaign_case"] for j in a["jobs"]] == ["K0", "K4"] * 4
    assert a["allocated_seconds"] <= 9 * 3600.
    assert a["lanes"]["K0"]["lane_seconds"] > 4 * 2100.
    assert template == original
    for entry in a["jobs"]:
        before = next(d for d in template["designs"] if d["design_id"] == entry["design"]["design_id"])
        after = entry["design"]
        assert main.numerical_workload(before) == main.numerical_workload(after)
        assert before["seed"] == after["seed"]
        assert before["options"]["data"] == after["options"]["data"]
    assert a["posterior_outcomes_used_for_allocation"] is False


def test_incomplete_or_different_workload_never_becomes_a_cheap_price(design):
    template, datasets = inventory(design)
    for status in ["incomplete", "complete_process_incomplete_declared_workload", "unpriced_main_workload_mismatch"]:
        a = main.allocate_lanes(template, datasets, {"K0": {"status": status, "price_seconds": 1.}}, 10000.)
        assert not a["jobs"]
        assert len(a["dispositions"]) == 32
    changed = deepcopy(template["designs"][0])
    changed["measurement_draws"] += 1
    assert main.numerical_workload(changed) != main.numerical_workload(template["designs"][0])
    changed = deepcopy(template["designs"][0])
    changed["options"]["reference_settings"] = {"resolution": 321}
    assert main.numerical_workload(changed) != main.numerical_workload(template["designs"][0])


def test_completed_unfavorable_assessment_still_supplies_cost(design):
    template, datasets = inventory(design)
    price = {"status": "complete_declared_workload", "price_seconds": 100.}
    good = main.allocate_lanes(template, datasets, {"K0": {**price, "posterior_passed": True}}, 10000.)
    bad = main.allocate_lanes(template, datasets, {"K0": {**price, "posterior_passed": False}}, 10000.)
    assert good == bad


def test_each_start_checks_original_cutoff_and_entire_slot():
    config = {"campaign_started_epoch": 0.}
    assert main.may_start(config, 100., 50., now_epoch=42*3600-1., now_monotonic=50.)
    assert not main.may_start(config, 100., 50., now_epoch=42*3600., now_monotonic=50.)
    assert not main.may_start(config, 100., 50., now_epoch=1., now_monotonic=51.)


def test_settled_repairs_and_active_reservation_constrain_main(tmp_path, monkeypatch):
    pool = tmp_path / "main"
    ledger = write_json(tmp_path / "ledger.json", {"grant_gpu_seconds": 200000.,
        "remaining_gpu_seconds": 171000., "allocations": {"canonical_neutra_pricing_reserved_only": 1200.},
        "records": [{"receipt": main.ssm_campaign.C1_RECEIPT, "elapsed_seconds": 10000.},
                    {"receipt": str(main.REPAIR_ROOT / "settled/execution.json"), "elapsed_seconds": 18000.},
                    {"receipt": str(pool / "prior/execution.json"), "elapsed_seconds": 1000.}]})
    config = {"ledger": str(ledger), "campaign_started_epoch": time.time(),
        "ssm_accounting_roots": [], "prior_ssm_receipts": [],
        "ssm_gpu_cap_seconds": 20000., "root": str(tmp_path / "old")}
    assert main.available_main_seconds(config, pool / "new") == 940.
    value = read_json(ledger); value["active_reservation"] = {"service": "prior"};write_json(ledger,value)
    with pytest.raises(ValueError, match="reservation"):
        main.available_main_seconds(config, pool / "new")


@pytest.mark.parametrize("mutation", [None, "assessment", "source", "memory", "jit", "numerical_failure"])
def test_native_main_receipt_and_gpu_provenance_are_required(tmp_path, design, mutation):
    d = design("accuracy", "gaussian", "ordinary", replications=1, device="gpu")
    source = {"identity": "frozen"}
    path = tmp_path / d.design_id / "replication-0000"
    assessment = write_json(path / "independent_assessment.json", {"posterior_passed": False})
    prefix = path / "process-attempt-001"
    receipt = {"status": "complete", "assessment_sha256": file_hash(assessment)}
    manifest = {"design_identity": d.identity, "source": source,
        "runtime": {"jit_compile": True, "gpu_tensor_device": "GPU:0", "memory_policy": {
            "configured_before_logical_device_initialization": True,
            "all_physical_devices_memory_growth": True, "physical_devices": ["GPU:0"]}}}
    if mutation == "assessment": assessment.write_text('{}')
    if mutation == "source": manifest["source"] = {"identity": "other"}
    if mutation == "memory": manifest["runtime"]["memory_policy"]["all_physical_devices_memory_growth"] = False
    if mutation == "jit": manifest["runtime"]["jit_compile"] = False
    if mutation == "numerical_failure": receipt["status"] = "failed"
    write_json(str(prefix) + "-exit.json", receipt)
    write_json(str(prefix) + "-manifest.json", manifest)
    if mutation is None:
        assert main.inspect_cell(tmp_path, d, source)["status"] == "complete"
    else:
        with pytest.raises(ValueError): main.inspect_cell(tmp_path, d, source)


def test_missing_terminal_main_receipt_is_not_reported_as_success(tmp_path, design):
    d = design("accuracy", "gaussian", "ordinary")
    with pytest.raises(ValueError, match="terminal numerical receipt"):
        main.inspect_cell(tmp_path, d, {"identity": "frozen"})


@pytest.mark.parametrize("coordinator_exit", [0, 1])
def test_main_worker_uses_frozen_cli_enforces_each_start_and_settles(tmp_path, design, monkeypatch, coordinator_exit):
    template, datasets = inventory(design)
    prices = {"K0": {"status": "complete_declared_workload", "price_seconds": 100.}}
    source = {"identity": "frozen"}
    allocation = main.allocate_lanes(template, datasets, prices, 10000.)
    output = tmp_path / "main" / "r1"
    write_json(output / "allocation.json", {"source": source, "prices": prices,
        "available_seconds": 10000., "allocation": allocation})
    ledger = write_json(tmp_path / "ledger.json", {"grant_gpu_seconds": 100000.,
        "remaining_gpu_seconds": 100000., "records": []})
    config = write_json(tmp_path / "config.json", {"ledger": str(ledger), "ssm_root": str(tmp_path / "frozen")})
    args = SimpleNamespace(config=config, output=output, gpu="GPU-test", service="main-test",
        seconds=allocation["allocated_seconds"]+30., work_deadline=time.monotonic()+allocation["allocated_seconds"])
    monkeypatch.setattr(main, "checked_inputs", lambda config: (template, datasets, prices, source))
    monkeypatch.setattr(main, "available_main_seconds", lambda *a: 10000.)
    starts = iter([True, False, False, False])
    monkeypatch.setattr(main, "may_start", lambda *a: next(starts))
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == tmp_path / "frozen" / "source"
        assert "--reuse-leapfrog-graphs" in command
        suite = read_json(command[4])
        d = main.ValidationDesign.from_payload(suite["designs"][0])
        root = Path(command[command.index("--output")+1]) / d.design_id / "replication-0000"
        assessment = write_json(root / "independent_assessment.json", {"posterior_passed": False})
        write_json(root / "process-attempt-001-manifest.json", {
            "design_identity": d.identity, "source": source})
        write_json(root / "process-attempt-001-exit.json", {
            "status": "complete", "assessment_sha256": file_hash(assessment)})
        return SimpleNamespace(returncode=coordinator_exit)

    monkeypatch.setattr(main.subprocess, "run", run)
    if coordinator_exit:
        with pytest.raises(ValueError, match="coordinator failed"):
            main.worker(args)
    else:
        main.worker(args)
    result = read_json(output / "result.json")
    assert result["exit_code"] == coordinator_exit
    if not coordinator_exit:
        assert [r["status"] for r in result["outcomes"]] == ["complete"] + ["unstarted_original_deadline"]*3
    assert len(calls) == 1
    settled = read_json(ledger)
    assert not settled.get("active_reservation")
    assert settled["records"] == [read_json(output / "execution.json")]


@pytest.mark.parametrize("prior_exit", [0, 1])
def test_main_dispatch_waits_for_normal_recovery_closeout(tmp_path, design, monkeypatch, prior_exit):
    template, datasets = inventory(design)
    prices = {"K0": {"status": "complete_declared_workload", "price_seconds": 100.}}
    ledger = write_json(tmp_path / "ledger.json", {"records": []})
    config = write_json(tmp_path / "config.json", {"ledger": str(ledger), "campaign_started_epoch": time.time()})
    previous = write_json(tmp_path / "prior/result.json", {"exit_code": prior_exit})
    output = tmp_path / "main/r1"
    monkeypatch.setattr(main.sys, "argv", ["main.py", str(config), str(output),
        "--gpu", "GPU-test", "--service", "main-test", "--after-service", "prior-test",
        "--after-result", str(previous)])
    monkeypatch.setattr(main, "checked_inputs", lambda c: (template, datasets, prices, {"identity": "frozen"}))
    monkeypatch.setattr(main, "available_main_seconds", lambda *a: 10000.)
    monkeypatch.setattr(main.time, "sleep", lambda seconds: None)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "systemctl":
            return SimpleNamespace(stdout="active\n" if len(calls)==1 else "inactive\n")
        assert command[0] == "systemd-run"
        assert "--property=KillMode=control-group" in command
        assert "--property=RuntimeMaxSec=750" in command
        assert "--setenv=TF_FORCE_GPU_ALLOW_GROWTH=true" in command
        assert float(command[command.index("--work-deadline")+1]) <= main.time.monotonic()+720.
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(main.subprocess, "run", run)
    if prior_exit:
        with pytest.raises(ValueError, match="preceding recovery failed"):
            main.main()
        assert len(calls) == 2 and not output.exists()
    else:
        with pytest.raises(SystemExit) as exc: main.main()
        assert exc.value.code == 0 and len(calls) == 3
        assert read_json(output / "allocation.json")["allocation"]["original_main_denominator"] == 32
