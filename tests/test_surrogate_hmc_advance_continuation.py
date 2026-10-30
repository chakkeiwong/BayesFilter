"""CPU-only handoff/budget regressions; workers are stubs, not HMC evidence."""
import hashlib
import json
import subprocess
import time
from types import SimpleNamespace

import pytest

from scripts import run_surrogate_hmc_five_model_master as master


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(master, "ROOT", tmp_path)
    monkeypatch.setattr(master, "FOUR", tmp_path/"four")
    monkeypatch.setattr(master, "RANGE", tmp_path/"range")
    source = tmp_path/"kernel.py"
    source.write_text("fixture scientific source")
    root = tmp_path/"advance"
    rows = []
    for model in master.MODELS:
        data = (master.RANGE/"data/1704-t20.json" if model == "range_bearing"
                else master.FOUR/"inputs"/f"{model}-d26100831/dataset.json")
        data.parent.mkdir(parents=True)
        data.write_text(model)
        mixed = model == "sir_d18"
        scope = dict(model=model, particles=4032, horizon=master.HORIZONS[model], tf32=False,
            dtype="float64" if mixed else "float32", pairwise_dtype="float32",
            precision_policy="fp64_recursion_fp32_pairs" if mixed else "uniform",
            prior=dict(id=master.LITERATURE_PRIOR), source_sha256={"kernel.py": master.digest(source)},
            data_sha256=master.digest(data))
        selection = root/model/"filter-tuning"/model/"filter-selection.json"
        record = dict(status="FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC", scope=scope,
                      selected=dict(valid=True, controls={"reset_steps": 5}))
        master.write(selection.parent/"result.json", record)
        record["selection_hash"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        master.write(selection, record)
        directory = root/model/"mechanics"/model
        health = dict(finite=True, veto=False, force_fallback_count=0, divergences=0)
        master.write(directory/"pilot.json", dict(status="PASS_HMC_MECHANICS_ONLY",
            scope=scope, finite=True, force_role="selected_controls", force_fallback_count=0,
            divergences=0, cold_run_health=health, warmed_run_health=health))
        master.write(directory/"endpoint-audit.json", dict(status="PASS_ENDPOINT_MECHANICS",
            valid=True, fixed_input_replay_equal=True, value=-10.))
        master.write(directory/"result.json", dict(status="PASS_HMC_MECHANICS_ONLY",
            scope=scope, command=["worker", "--selection", str(selection)]))
        rows.append(dict(model=model, **{
            "filter-tuning": dict(status="FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC"),
            "mechanics": dict(status="PASS_HMC_MECHANICS_ONLY")}))
    master.write(root/"advance.json", dict(models=rows))
    return root


def test_actual_layout_supports_sir_mixed_precision(prepared):
    result = master.advance_prepared_selections([prepared], ("ksc", "sir_d18"), master.LITERATURE_PRIOR)
    assert set(result) == {"ksc", "sir_d18"}
    assert result["sir_d18"]["scope"]["dtype"] == "float64"
    assert result["sir_d18"]["scope"]["pairwise_dtype"] == "float32"
    assert all(not r["posterior_sampling_complete"] for r in result.values())
    with pytest.raises(ValueError, match="duplicate"):
        master.advance_prepared_selections([prepared, prepared], ("ksc",), master.LITERATURE_PRIOR)


@pytest.mark.parametrize("damage", ["source", "data", "prior", "precision", "selection_checksum",
    "tuning", "endpoint", "pilot", "health", "selection_binding", "missing_model"])
def test_inference_cannot_consume_stale_or_failed_preparation(prepared, damage):
    selected = prepared/"ksc/filter-tuning/ksc/filter-selection.json"
    mechanics = prepared/"ksc/mechanics/ksc"
    if damage == "source":
        (master.ROOT/"kernel.py").write_text("changed")
    elif damage == "data":
        (master.FOUR/"inputs/ksc-d26100831/dataset.json").write_text("changed")
    elif damage in ("prior", "precision", "selection_checksum"):
        row = master.read(selected)
        row.pop("selection_hash")
        if damage == "prior":
            row["scope"]["prior"]["id"] = master.BENCHMARK_PRIOR
        else:
            row["scope"]["dtype"] = "float64"
        row["selection_hash"] = ("incorrect" if damage == "selection_checksum" else
                                 hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest())
        master.write(selected, row)
    else:
        path, field, value = {
            "tuning": (selected.parent/"result.json", "selected", {"valid": False}),
            "endpoint": (mechanics/"endpoint-audit.json", "fixed_input_replay_equal", False),
            "pilot": (mechanics/"pilot.json", "finite", False),
            "health": (mechanics/"pilot.json", "warmed_run_health", {}),
            "selection_binding": (mechanics/"result.json", "command", ["worker"]),
            "missing_model": (prepared/"advance.json", "models", []),
        }[damage]
        master.write(path, dict(master.read(path), **{field: value}))
    with pytest.raises(ValueError):
        master.advance_prepared_selections([prepared], ("ksc",), master.LITERATURE_PRIOR)


def arguments(prepared, tmp_path):
    ledger = tmp_path/"campaign/budget.json"
    master.write(ledger, dict(authorized_gpu_wall_seconds=1000., max_attempts=2,
        attempts=[dict(status="PASS", wall_seconds=10.)], used_seconds=10., remaining_seconds=990.))
    return SimpleNamespace(priors=master.LITERATURE_PRIOR, budget_seconds=500.,
        per_model_timeout=300., advance_root=[prepared], campaign_ledger=ledger,
        output=tmp_path/"continuation", model="ksc", gpu="GPU-test-uuid")


@pytest.mark.parametrize("outcome", ["VERIFIED", "HEALTH_VETO", "timeout", "bad_exit", "launch_error"])
def test_continuation_only_runs_inference_and_charges_deadlined_worker(prepared, tmp_path, monkeypatch, outcome):
    args = arguments(prepared, tmp_path)
    calls = []
    def worker(command, **kwargs):
        calls.append(command)
        assert command[command.index("--phase")+1] == "inference"
        assert command[command.index("--priors")+1] == master.LITERATURE_PRIOR
        assert command[command.index("--selection")+1].endswith("filter-selection.json")
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-test-uuid"
        assert kwargs["env"]["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
        assert 0 < float(kwargs["env"]["BAYESFILTER_WORKER_WALL_DEADLINE"])-time.monotonic() <= 240.
        budget = master.read(args.campaign_ledger)
        assert budget["attempts"][-1]["status"] == "RUNNING"
        assert budget["attempts"][-1]["reserved_seconds"] == kwargs["timeout"] == 300.
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if outcome == "launch_error":
            raise OSError("worker launch failed")
        output = master.Path(command[command.index("--output")+1])
        master.write(output/"hmc.json", dict(status="VERIFIED" if outcome == "bad_exit" else outcome))
        return SimpleNamespace(returncode=1 if outcome == "bad_exit" else 0)
    monkeypatch.setattr(master.subprocess, "run", worker)
    monkeypatch.setattr(master.subprocess, "check_output", lambda *a, **k: "fixture-head\n")
    if outcome == "launch_error":
        with pytest.raises(OSError, match="worker launch failed"):
            master.continue_advance(args)
    else:
        assert master.continue_advance(args) == (0 if outcome == "VERIFIED" else 2)
    assert len(calls) == 1
    result = master.read(args.output/"master.json")
    expected = {"timeout": "STAGE_TIME_BUDGET_STOP", "bad_exit": "WORKER_EXIT_ERROR",
                "launch_error": "STAGE_ERROR"}.get(outcome, outcome)
    assert result["models"]["ksc"]["inference"]["status"] == expected
    budget = master.read(args.campaign_ledger)
    assert len(budget["attempts"]) == 2
    assert budget["used_seconds"] == 10. + budget["attempts"][-1]["wall_seconds"]
    assert budget["remaining_seconds"] == 1000. - budget["used_seconds"]
    assert master.read(args.output/"next-phase-plan.json")["automatic_continuation"] is False
    assert master.read(args.campaign_ledger.parent/"next-phase-plan.json")["part_of_master"]


@pytest.mark.parametrize("reason", ["attempts", "time", "running"])
def test_no_worker_launch_when_cumulative_campaign_is_unavailable(prepared, tmp_path, monkeypatch, reason):
    args = arguments(prepared, tmp_path)
    ledger = master.read(args.campaign_ledger)
    if reason == "attempts":
        ledger["max_attempts"] = 1
    elif reason == "time":
        ledger["authorized_gpu_wall_seconds"] = 10.
    else:
        ledger["attempts"][0]["status"] = "RUNNING"
    master.write(args.campaign_ledger, ledger)
    monkeypatch.setattr(master.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    monkeypatch.setattr(master.subprocess, "check_output", lambda *a, **k: "fixture-head\n")
    if reason == "running":
        with pytest.raises(ValueError, match="worker is running"):
            master.continue_advance(args)
    else:
        assert master.continue_advance(args) == 2
    assert len(master.read(args.campaign_ledger)["attempts"]) == 1


def test_all_models_continue_after_a_model_health_veto(prepared, tmp_path, monkeypatch):
    args = arguments(prepared, tmp_path)
    args.model = None
    master.write(args.campaign_ledger, dict(master.read(args.campaign_ledger), max_attempts=8))
    calls = []
    def worker(command, **kwargs):
        model = command[command.index("--model")+1]
        calls.append(model)
        entries = master.read(args.campaign_ledger)["attempts"]
        assert entries[-1]["status"] == "RUNNING"
        assert all(row["status"] != "RUNNING" for row in entries[:-1])
        output = master.Path(command[command.index("--output")+1])
        master.write(output/"hmc.json", dict(status="HEALTH_VETO" if model == "ksc" else "VERIFIED"))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(master.subprocess, "run", worker)
    monkeypatch.setattr(master.subprocess, "check_output", lambda *a, **k: "fixture-head\n")
    assert master.continue_advance(args) == 2
    assert calls == ["ksc", "range_bearing", "lgssm", "predator_prey", "sir_d18"]
    final = master.read(args.output/"master.json")
    assert final["status"] == "INCOMPLETE_SEE_MODEL_STATUS"
    assert final["models"]["ksc"]["inference"]["status"] == "HEALTH_VETO"
    assert final["models"]["sir_d18"]["inference"]["status"] == "VERIFIED"
    ledger = master.read(args.campaign_ledger)
    assert len(ledger["attempts"]) == 6
    assert ledger["used_seconds"] == sum(r["wall_seconds"] for r in ledger["attempts"])
