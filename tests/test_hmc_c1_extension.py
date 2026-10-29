"""Queue, funding and source invariants without launching a GPU workload."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "docs/benchmarks/run_hmc_c1_extension_2026_09_25.py"
spec = importlib.util.spec_from_file_location("hmc_c1_extension", MODULE)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


@pytest.fixture
def prepared():
    return {"suite_id": "prepared", "designs": [
        {"design_id": "gaussian", "scenario": {"target": "gaussian"},
         "replications": 256, "seed": 11, "phase": "confirmation", "budget_seconds": 1,
         "purpose": "prepared", "options": {"posterior_settings": {"warmup_min_results": 30000}},
         "l_grid": [3, 5, 9, 13, 18, 25], "alpha": .05},
        {"design_id": "beta", "scenario": {"target": "beta_binomial"},
         "replications": 256, "seed": 12, "phase": "confirmation", "budget_seconds": 1,
         "purpose": "prepared", "options": {"data": [5, 12]},
         "l_grid": [3, 5, 9, 13, 18, 25], "alpha": .05}]}


def test_confirmation_preserves_frozen_scientific_design(prepared):
    original = copy.deepcopy(prepared)
    suite, audit = queue.confirmation_suite(prepared,
        {"gaussian": [240., 260.], "beta_binomial": [270., 300.]})
    assert prepared == original
    assert audit["forecast_gpu_seconds"] == 256*560 + 120
    assert sum(d["budget_seconds"] for d in suite["designs"]) == queue.CONFIRMATION_CAP-120
    for old, new in zip(prepared["designs"], suite["designs"]):
        assert {k:v for k,v in old.items() if k not in {"budget_seconds", "purpose"}} == {
            k:v for k,v in new.items() if k not in {"budget_seconds", "purpose"}}


@pytest.mark.parametrize("costs", [
    {"gaussian": [400, 450], "beta_binomial": [400, 450]},
    {"gaussian": [323.203126]*2, "beta_binomial": [323.203126]*2},
])
def test_unaffordable_never_reduces_denominators(prepared, costs):
    suite, audit = queue.confirmation_suite(prepared, costs)
    assert suite is None and audit["affordable"] is False
    assert [d["replications"] for d in prepared["designs"]] == [256, 256]


@pytest.mark.parametrize("costs", [
    {"gaussian": [1], "beta_binomial": [1, 2]},
    {"gaussian": [float("nan"), 2], "beta_binomial": [1, 2]},
    {"gaussian": [0, 2], "beta_binomial": [1, 2]},
])
def test_invalid_prices_fail_closed(prepared, costs):
    with pytest.raises(ValueError):
        queue.confirmation_suite(prepared, costs)


def test_pricing_is_disjoint_and_uses_same_posterior_workload(prepared):
    suite = queue.pricing_suite(prepared)
    assert len(suite["designs"]) == 4
    assert len({d["seed"] for d in suite["designs"]}) == 4
    assert not {d["seed"] for d in suite["designs"]} & {d["seed"] for d in prepared["designs"]}
    for i, d in enumerate(suite["designs"]):
        old = prepared["designs"][i//2]
        assert d["replications"] == 1 and d["phase"] == "development"
        assert d["scenario"] == old["scenario"]
        assert d["l_grid"] == old["l_grid"]
        assert {k:v for k,v in d["options"].items() if k != "plan_file"} == old["options"]


def runtime():
    return {"runtime": {"device_scope": "gpu", "jit_compile": True,
        "visible_devices": queue.GPU_UUID, "gpu_tensor_device": "device:GPU:0",
        "memory_policy": {"all_physical_devices_memory_growth": True,
            "configured_before_logical_device_initialization": True,
            "tf_force_gpu_allow_growth": "true"}}}


@pytest.mark.parametrize("field,value", [("jit_compile", False), ("visible_devices", "0"),
                                          ("gpu_tensor_device", "device:CPU:0")])
def test_runtime_requires_real_selected_gpu(field, value):
    m = runtime()
    m["runtime"][field] = value
    with pytest.raises(RuntimeError):
        queue.validate_runtime(m)


def test_pricing_rejects_censored_and_no_posterior_output(tmp_path):
    suite = {"designs": [{"design_id": "d", "scenario": {"target": "gaussian"}}]}
    cell = tmp_path / "suite/d"
    fit = cell / "replication-0000"
    assessment = fit / "independent_assessment.json"
    queue.write(assessment, {"inventory": {"failures": []}, "members": [{"retained_count": 20}]})
    queue.write(fit / "process-attempt-001-manifest.json", runtime())
    receipt = {"status": "timed_out", "exit_code": 0,
               "assessment_sha256": queue.file_hash(assessment)}
    queue.write(fit / "process-attempt-001-exit.json", receipt)
    queue.write(cell / "result.json", {"execution_status": "complete", "elapsed_seconds": 90})
    queue.write(tmp_path / "suite/run_index.json", {"jobs": {"d": {
        "status": "complete", "attempts": [{"elapsed_seconds": 91}]}}})
    with pytest.raises(RuntimeError, match="censored"):
        queue.direct_prices(tmp_path, suite)
    receipt["status"] = "complete"
    queue.write(fit / "process-attempt-001-exit.json", receipt)
    assert queue.direct_prices(tmp_path, suite) == {"gaussian": [91.]}
    queue.write(assessment, {"inventory": {"failures": []}, "members": []})
    receipt["assessment_sha256"] = queue.file_hash(assessment)
    queue.write(fit / "process-attempt-001-exit.json", receipt)
    with pytest.raises(RuntimeError, match="no retained"):
        queue.direct_prices(tmp_path, suite)


def test_finalizer_conservatively_charges_once(tmp_path):
    queue.write(tmp_path / "grant-ledger.json", {"grant_gpu_seconds": queue.GRANT})
    queue.write(tmp_path / "active-run.json", {"stage": "pricing-r1", "cap_seconds": 3660})
    queue.write(tmp_path / "status.json", {"status": "pricing-r1_running"})
    queue.finalize(tmp_path)
    queue.finalize(tmp_path)
    ledger = queue.read(tmp_path / "grant-ledger.json")
    assert ledger["charged_gpu_seconds"] == 3661
    assert ledger["remaining_gpu_seconds"] == queue.GRANT-3661
    assert len(ledger["records"]) == 1
    assert queue.read(tmp_path / "status.json")["status"] == "interrupted_requires_diagnosis"


def test_real_receipts_are_not_double_counted_or_overwritten(tmp_path):
    queue.write(tmp_path / "grant-ledger.json", {"grant_gpu_seconds": queue.GRANT})
    queue.write(tmp_path / "pricing-r1/execution.json", {"elapsed_seconds": 500, "exit_code": 0})
    queue.write(tmp_path / "confirmation-r1/execution.json", {"elapsed_seconds": 130000, "exit_code": 0})
    queue.reconcile(tmp_path)
    result = queue.reconcile(tmp_path)
    assert result["charged_gpu_seconds"] == 130500 and len(result["records"]) == 2


def test_source_manifest_detects_changed_added_and_deleted_code(tmp_path):
    path = tmp_path / "source/bayesfilter/m.py"
    path.parent.mkdir(parents=True)
    path.write_text("value = 1\n")
    queue.write(tmp_path / "source/source_snapshot.json", {"files": {"bayesfilter/m.py": queue.file_hash(path)}})
    queue.verify_source(tmp_path)
    path.write_text("value = 2\n")
    with pytest.raises(RuntimeError):
        queue.verify_source(tmp_path)


def test_missing_terminal_cell_cannot_pass(tmp_path, prepared):
    result = queue.terminal_summary(tmp_path, prepared)
    assert result["all_declared_screens_passed"] is False
    assert result["scientific_gaps_closed"] is False
    assert [c["planned"] for c in result["cells"]] == [256, 256]


@pytest.mark.parametrize("affordable", [True, False])
def test_queue_prices_before_deciding_to_confirm(tmp_path, prepared, monkeypatch, affordable):
    from types import SimpleNamespace
    root = tmp_path / "new"
    old = tmp_path / "old"
    queue.write(root / "grant-ledger.json", {"grant_gpu_seconds": queue.GRANT})
    queue.write(root / "prepared-c1-suite.json", prepared)
    queue.write(root / "manifest.json", {"prepared_suite_sha256":
                                        queue.file_hash(root / "prepared-c1-suite.json")})
    queue.write(old / "i-terminal-result.json", {"status": "C2_not_closed"})
    monkeypatch.setattr(queue, "OLD_ROOT", old)
    monkeypatch.setattr(queue, "verify_source", lambda root: None)
    monkeypatch.setattr(queue.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="inactive\n"))
    monkeypatch.setattr(queue, "gpu_snapshot", lambda: {"compute_processes": []})
    costs = {"gaussian": [250, 260], "beta_binomial": [270, 280]} if affordable else {
        "gaussian": [600, 650], "beta_binomial": [600, 650]}
    calls = []
    monkeypatch.setattr(queue, "run_stage", lambda root, name, path, cap: calls.append((name, cap)))
    monkeypatch.setattr(queue, "direct_prices", lambda *args: costs)
    monkeypatch.setattr(queue, "terminal_summary", lambda *args: {
        "status": "terminal_scientific_review_pending", "scientific_gaps_closed": False})
    queue.execute(root)
    expected = [("pricing-r1", queue.PRICING_CAP)]
    if affordable:
        expected.append(("confirmation-r1", queue.CONFIRMATION_CAP))
        assert [d["replications"] for d in queue.read(root / "confirmation-suite.json")["designs"]] == [256, 256]
        assert queue.read(root / "terminal-result.json")["scientific_gaps_closed"] is False
    else:
        assert not (root / "confirmation-suite.json").exists()
    assert calls == expected
