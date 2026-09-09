"""CPU contract checks for the Phase 9B P1 canary repair."""

from __future__ import annotations

import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "docs/benchmarks/run_ssl_lstm_q20_phase9b_p1_sequential_canary_2026_09_05.py"
)


def _runner_module():
    spec = importlib.util.spec_from_file_location("phase9b_p1_canary_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load Phase 9B P1 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_campaign_entrypoint_delegates_to_parallel_recoverable_p1(tmp_path, monkeypatch):
    module = _runner_module()
    calls = []

    def coordinator(campaign, **kwargs):
        calls.append((campaign, kwargs))
        return 0

    monkeypatch.setattr(module, "_recovery_coordinator", lambda: coordinator)
    assert module.main(["--campaign-root", str(tmp_path), "--resume", "--initialize-only"]) == 0
    assert calls == [(tmp_path, {"resume": True, "through_p1": True, "initialize_only": True})]
    with pytest.raises(SystemExit):
        module.main(["--campaign-root", str(tmp_path), "--output-dir", str(tmp_path / "legacy")])


def test_chart_selection_accepts_a_direct_transport_object() -> None:
    module = _runner_module()
    transport = object()

    assert module._select_chart({1.0: transport}, 1.0) is transport
    with pytest.raises(module.P1CanaryError, match="lacks beta"):
        module._select_chart({}, 1.0)


def test_seed_namespaces_are_disjoint_by_role_arm_and_attempt() -> None:
    module = _runner_module()
    first = Path("/tmp/phase9b-p1-attempt-a")
    second = Path("/tmp/phase9b-p1-attempt-b")
    first_values = {
        value
        for arm in ("factor", "strict")
        for value in module._seed_map(first, arm).values()
    }
    second_values = {
        value
        for arm in ("factor", "strict")
        for value in module._seed_map(second, arm).values()
    }

    assert len(first_values) == 20
    assert len(second_values) == 20
    assert first_values.isdisjoint(second_values)


def test_budget_forecast_fails_before_an_under_budget_minimum_schedule() -> None:
    module = _runner_module()

    module._validate_budget_forecast(100.0)
    module._validate_budget_forecast(1423.0)
    with pytest.raises(module.P1CanaryError, match="does not fit"):
        module._validate_budget_forecast(module.ARM_CAP_SECONDS / 6.0 + 1.0)


def _phase0_closeout_fixture(module, tmp_path: Path):
    common = {
        "target_signature": module.TARGET_SIGNATURE,
        "policy_id": module.POLICY_ID,
        "claim_boundary": "phase9b_m4_p0_executable_readiness_only",
        "source_hashes": dict(module._readiness_source_hashes()),
    }
    runtime = {
        **common,
        "status": "PASS_PHASE9B_M4_P0_RUNTIME_DIAGNOSTIC",
        "gpu": {
            "placement_policy_id": module.GPU_PLACEMENT_POLICY_ID,
            "selection": {"selected": {
                "uuid": "GPU-synthetic-0", "pci_bus_id": "0000:00:00.0",
                "name": "synthetic", "memory_total_mib": 32768.0,
            }},
        },
        "timing": {
            "p1_required_chunks_per_arm": 6,
            "sequential_chunk_forecast_seconds": 100.0,
            "p1_forecast_seconds_per_arm": {"factor": 900.0, "strict": 900.0},
            "p1_complete_schedule_forecast_seconds": 1900.0,
        },
    }
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    closeout = {
        **common,
        "source_hashes": dict(common["source_hashes"]),
        "status": "P1_LAUNCHABLE_PENDING_FRESH_ATTEMPT",
        "runtime_diagnostic": {"path": str(runtime_path), "sha256": module._sha256(runtime_path)},
        "budget": {"material_cap_seconds": module.MATERIAL_CAP_SECONDS, "arm_cap_seconds": module.ARM_CAP_SECONDS},
    }
    closeout_path = tmp_path / "closeout.json"
    closeout_path.write_text(json.dumps(closeout), encoding="utf-8")
    return closeout_path, closeout, runtime_path, runtime


def test_phase0_closeout_binds_runtime_sources_and_forecast(tmp_path: Path) -> None:
    module = _runner_module()
    closeout_path, _, _, _ = _phase0_closeout_fixture(module, tmp_path)

    receipt = module._verify_phase0_closeout(closeout_path, 100.0)

    assert receipt["complete_schedule_forecast_seconds"] == 1900.0
    with pytest.raises(module.P1CanaryError, match="measured forecast"):
        module._verify_phase0_closeout(closeout_path, 1.0)


def test_forecast_cannot_transfer_to_a_different_gpu_class():
    module = _runner_module()
    measured = {
        "uuid": "GPU-synthetic-0",
        "pci_bus_id": "0000:00:00.0",
        "name": "synthetic",
        "memory_total_mib": 32768.0,
    }
    module._verify_forecast_hardware({"selected": dict(measured)}, measured)
    with pytest.raises(module.P1CanaryError, match="remeasure"):
        module._verify_forecast_hardware({"selected": {**measured, "name": "different"}}, measured)
    with pytest.raises(module.P1CanaryError, match="remeasure"):
        module._verify_forecast_hardware({"selected": {**measured, "uuid": "GPU-synthetic-1"}}, measured)
    with pytest.raises(module.P1CanaryError, match="remeasure"):
        module._verify_forecast_hardware({"selected": {**measured, "pci_bus_id": "0000:01:00.0"}}, measured)
    with pytest.raises(module.P1CanaryError, match="remeasure"):
        module._verify_forecast_hardware({"selected": {}}, {})


@pytest.mark.parametrize(
    "case, message",
    (
        ("source_only", "not passing"),
        ("stale_closeout", "stale controller"),
        ("stale_runtime", "stale diagnostic"),
        ("runtime_failure", "not passing"),
        ("checksum", "checksum mismatch"),
        ("short_schedule", "complete P1 schedule"),
        ("over_arm_cap", "arm cap"),
        ("missing_arm_cost", "omits arm costs"),
        ("nonfinite", "positive and finite"),
        ("missing_gpu_uuid", "physical GPU identity missing"),
        ("nonfinite_capacity", "physical GPU identity missing"),
    ),
)
def test_phase0_closeout_rejects_invalid_evidence(
    tmp_path: Path, case: str, message: str
) -> None:
    module = _runner_module()
    closeout_path, closeout, runtime_path, runtime = _phase0_closeout_fixture(module, tmp_path)
    if case == "source_only":
        closeout["status"] = "PASS_PHASE9B_P0_SOURCE_PREFLIGHT"
    elif case == "stale_closeout":
        closeout["source_hashes"]["controller"] = "stale"
    elif case == "stale_runtime":
        runtime["source_hashes"]["diagnostic"] = "stale"
    elif case == "runtime_failure":
        runtime["status"] = "FAIL_PHASE9B_M4_P0_READINESS"
    elif case == "short_schedule":
        runtime["timing"]["p1_required_chunks_per_arm"] = 1
    elif case == "over_arm_cap":
        runtime["timing"]["p1_forecast_seconds_per_arm"]["strict"] = module.ARM_CAP_SECONDS + 1.0
    elif case == "missing_arm_cost":
        runtime["timing"]["p1_complete_schedule_forecast_seconds"] = 900.0
    elif case == "nonfinite":
        runtime["timing"]["sequential_chunk_forecast_seconds"] = float("nan")
    elif case == "missing_gpu_uuid":
        runtime["gpu"]["selection"]["selected"].pop("uuid")
    elif case == "nonfinite_capacity":
        runtime["gpu"]["selection"]["selected"]["memory_total_mib"] = float("nan")
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    closeout["runtime_diagnostic"]["sha256"] = (
        "stale" if case == "checksum" else module._sha256(runtime_path)
    )
    closeout_path.write_text(json.dumps(closeout), encoding="utf-8")

    with pytest.raises(module.P1CanaryError, match=message):
        module._verify_phase0_closeout(closeout_path, 100.0)


def test_missing_phase0_closeout_stops_before_source_load_or_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _runner_module()

    def unexpected_call(*args, **kwargs):
        pytest.fail("a missing Phase 0 closeout must stop before imports or budget allocation")

    monkeypatch.setattr(module, "_verify_source_inputs", unexpected_call)
    monkeypatch.setattr(module, "_open_campaign_ledger", unexpected_call)
    monkeypatch.setattr(module, "_load_phase9a_runner", unexpected_call)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    output = tmp_path / "attempt"

    assert module.main(
        ["--output-dir", str(output), "--phase0-closeout", str(tmp_path / "missing.json"),
         "--sequential-chunk-forecast-seconds", "100"]
    ) == 2
    payload = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert "Phase 0 closeout required" in payload["error"]
    assert payload["allocator_telemetry"]["status"] == "not_collected_on_failure"


def test_campaign_allocation_reuses_the_remaining_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _runner_module()
    monkeypatch.setattr(module, "CAMPAIGN_LEDGER_PATH", tmp_path / "campaign.json")
    ledger = module._open_campaign_ledger(tmp_path / "first", minimum_remaining_seconds=3000.0)
    ledger.settle_arm(attempt_id="first", arm="readiness_diagnostic", measured_seconds=500.0, status="failed")
    ledger.finish_attempt(attempt_id="first", status="failed")

    with pytest.raises(module.P1CanaryError, match="remaining campaign budget"):
        module._open_campaign_ledger(tmp_path / "second", minimum_remaining_seconds=3000.0)
    assert ledger.remaining_seconds() == pytest.approx(2867.39)
    assert not any(attempt["attempt_id"] == "second" for attempt in ledger.read()["attempts"])


def test_source_inputs_reject_historical_p1_audit_after_budget_amendment() -> None:
    module = _runner_module()

    with pytest.raises(module.P1CanaryError, match="p1_audit_.*_hash_mismatch"):
        module._verify_source_inputs()


def test_diagnostics_transpose_shared_controller_samples_and_compute_mcse() -> None:
    tf = pytest.importorskip("tensorflow")
    module = _runner_module()
    shared_samples = tf.cast(tf.reshape(tf.range(6 * 4 * 4), (6, 4, 4)), tf.float64)

    summary = module._diagnostic_summary(tf, shared_samples)

    assert summary["status"] == "COMPUTED"
    assert summary["input_layout"] == "[draw, chain, parameter]"
    assert summary["diagnostic_layout"] == "[chain, draw, parameter]"
    assert summary["input_shape"] == (6, 4, 4)
    assert summary["diagnostic_shape"] == (4, 6, 4)
    assert summary["mcse"]["status"] == "COMPUTED"


def test_failure_payload_records_incomplete_runtime_provenance(tmp_path: Path) -> None:
    module = _runner_module()
    module._ACTIVE_CONTEXT.clear()
    module._ACTIVE_CONTEXT.update(
        {
            "output": tmp_path,
            "started": 0.0,
            "started_at_utc": "2026-09-06T00:00:00+00:00",
            "seed_namespaces": {"factor": {"warmup": (1, 2)}},
            "memory_policy": {"status": "not_initialized"},
        }
    )

    payload = module._failure_payload(tmp_path, RuntimeError("controlled failure"))

    assert payload["status"] == "FAIL_PHASE9B_P1_CANARY"
    assert payload["started_at_utc"] == "2026-09-06T00:00:00+00:00"
    assert payload["allocator_telemetry"]["status"] == "not_collected_on_failure"
    assert payload["seed_namespaces"]["factor"]["warmup"] == (1, 2)


@pytest.mark.parametrize("fail_arm", (False, True))
def test_terminal_writer_and_failed_tuning_charge_campaign_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_arm: bool
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    tf = pytest.importorskip("tensorflow")
    tf.constant(0.0)
    from bayesfilter.inference import tempered_target_tf
    from bayesfilter.runtime import gpu_memory_policy

    module = _runner_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "CAMPAIGN_LEDGER_PATH", tmp_path / "campaign.json")
    monkeypatch.setattr(module, "_verify_source_inputs", lambda: {})
    monkeypatch.setattr(
        module,
        "_verify_phase0_closeout",
        lambda *args: {
            "complete_schedule_forecast_seconds": 1.0,
            "measured_gpu": {
                "uuid": "GPU-synthetic-0",
                "pci_bus_id": "synthetic",
                "name": "synthetic",
                "memory_total_mib": 32768.0,
            },
        },
    )
    monkeypatch.setattr(
        module,
        "_select_phase9b_gpu",
        lambda: {
            "schema": "synthetic_gpu_selection",
            "policy_id": module.GPU_PLACEMENT_POLICY_ID,
            "reason": "synthetic_cpu_test",
            "selected": {
                "uuid": "GPU-synthetic-0",
                "pci_bus_id": "synthetic",
                "name": "synthetic",
                "memory_total_mib": 32768.0,
            },
        },
    )
    monkeypatch.setattr(module, "_load_phase9a_runner", lambda: None)
    monkeypatch.setattr(module, "_arm_profile", lambda *args: None)
    monkeypatch.setattr(module, "_git_payload", lambda: {"status": "synthetic_cpu_test"})
    monkeypatch.setattr(module, "_campaign_plan_hash", lambda: "synthetic_cpu_plan")
    for name in ("PLAN", "P0_MANIFEST", "P1_AUDIT_MANIFEST"):
        monkeypatch.setattr(module, name, tmp_path / f"{name}.json")
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-synthetic-0")
    monkeypatch.setattr(
        gpu_memory_policy, "configure_tensorflow_gpu_memory_growth",
        lambda *args, **kwargs: {"status": "synthetic_cpu_test_no_gpu"},
    )
    monkeypatch.setattr(tf.config, "list_logical_devices", lambda kind: (SimpleNamespace(name="synthetic_device"),))
    monkeypatch.setattr(
        tf.config.experimental,
        "get_memory_info",
        lambda device: {"current": 0, "peak": 0},
    )
    monkeypatch.setattr(
        module,
        "check_runtime_headroom",
        lambda selection, allocator_peak: {
            "status": "synthetic_cpu_test",
            "allocator_peak_bytes": allocator_peak,
            "selected": selection["selected"],
        },
    )
    bridge = SimpleNamespace(
        target_signature=module.TARGET_SIGNATURE,
        prior_center=(0.0, 0.0, 0.0, 0.0),
        value_score_status=lambda state, beta: (
            tf.zeros((2,), tf.float64), tf.zeros((2, 4), tf.float64),
            {"bridge_valid": tf.ones((2,), tf.bool)},
        ),
    )
    monkeypatch.setattr(tempered_target_tf, "make_q20_tempered_bridge", lambda *args, **kwargs: bridge)

    def run_arm(**kwargs):
        if fail_arm:
            raise RuntimeError("synthetic pre-controller tuning failure")
        return {"sequential": {"passed": True}}

    monkeypatch.setattr(module, "_run_arm", run_arm)
    output = tmp_path / "attempt"
    assert module.main(
        ["--output-dir", str(output), "--phase0-closeout", "synthetic.json",
         "--sequential-chunk-forecast-seconds", "100"]
    ) == (2 if fail_arm else 0)
    ledger = json.loads(module.CAMPAIGN_LEDGER_PATH.read_text(encoding="utf-8"))
    attempt = ledger["attempts"][-1]
    assert attempt["consumed_seconds"] > 0.0
    assert ledger["consumed_seconds"] == pytest.approx(
        module.HISTORICAL_CONSUMED_SECONDS + attempt["consumed_seconds"]
    )
    if fail_arm:
        assert (output / "failure.json").is_file()
    else:
        assert not (output / "failure.json").exists()
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        assert result["status"] == "PASS_PHASE9B_P1_CANARY"
        assert manifest["manifest_hash"] == module._sha256(output / "result.json")
