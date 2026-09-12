"""CPU-only synthetic contract tests; no q=20 or GPU-readiness evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_PATH = ROOT / "docs/benchmarks/diagnose_ssl_lstm_q20_phase9b_executable_readiness_2026_09_06.py"
P1_PATH = ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_p1_sequential_canary_2026_09_05.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _arm_result():
    return {
        "timing": {
            "profile_seconds": 1.0,
            "bridge_seconds": 1.0,
            "chart_seconds": 1.0,
            "tuning_seconds": 1.0,
            "first_compiled_call_seconds": 10.0,
            "steady_state_chunk_seconds": 2.0,
            "serialization_seconds": 1.0,
            "cleanup_seconds": 1.0,
        }
    }


def test_complete_forecast_counts_setup_archives_reserve_and_distinct_grace() -> None:
    module = _load(DIAGNOSTIC_PATH, "phase9b_readiness_forecast_test")
    results = {arm: _arm_result() for arm in ("factor", "strict")}

    forecast = module._complete_schedule_forecast(results, startup_seconds=3.0, chunk_results=500)

    assert forecast["p1_forecast_seconds_per_arm"] == {"factor": 43.0, "strict": 43.0}
    assert forecast["p1_complete_schedule_forecast_seconds"] == 89.0
    assert forecast["sequential_chunk_forecast_seconds"] == 4.0


@pytest.mark.parametrize("case", ("missing", "nonfinite", "negative", "wrong_chunk"))
def test_incomplete_timing_cannot_issue_a_forecast(case: str) -> None:
    module = _load(DIAGNOSTIC_PATH, "phase9b_readiness_invalid_timing_test")
    results = {arm: _arm_result() for arm in ("factor", "strict")}
    if case == "missing":
        del results["strict"]["timing"]["first_compiled_call_seconds"]
    elif case == "nonfinite":
        results["strict"]["timing"]["steady_state_chunk_seconds"] = float("nan")
    elif case == "negative":
        results["strict"]["timing"]["cleanup_seconds"] = -1.0

    with pytest.raises(module.ReadinessDiagnosticError):
        module._complete_schedule_forecast(
            results, startup_seconds=3.0, chunk_results=5 if case == "wrong_chunk" else 500
        )


def _synthetic_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load(DIAGNOSTIC_PATH, "phase9b_readiness_runtime_test")
    p1_runner = _load(P1_PATH, "phase9b_readiness_p1_test")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "_git_payload", lambda: {"status": "synthetic_cpu_test"})
    monkeypatch.setattr(p1_runner, "CAMPAIGN_LEDGER_PATH", tmp_path / "campaign.json")
    monkeypatch.setattr(p1_runner, "_verify_source_inputs", lambda: {"status": "synthetic_cpu_test"})
    monkeypatch.setattr(p1_runner, "_load_phase9a_runner", lambda: None)
    monkeypatch.setattr(module, "_load_module", lambda *args: p1_runner)
    monkeypatch.setattr(module, "_run_arm", lambda **kwargs: _arm_result())
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-synthetic-cpu-test")
    monkeypatch.setattr(module, "_select_phase9b_gpu", lambda: {
        "policy_id": module.GPU_PLACEMENT_POLICY_ID,
        "selected": {"uuid": "GPU-synthetic-cpu-test"}, "reason": "synthetic_cpu_test",
    })
    monkeypatch.setattr(module, "check_runtime_headroom", lambda *args: {"status": "synthetic_cpu_test"})
    fake_tf = SimpleNamespace(
        __version__="synthetic_cpu_test",
        config=SimpleNamespace(
            experimental=SimpleNamespace(
                enable_tensor_float_32_execution=lambda enabled: None,
                tensor_float_32_execution_enabled=lambda: True,
                get_memory_info=lambda device: {"current": 0, "peak": 0},
            ),
            set_soft_device_placement=lambda enabled: None,
            list_physical_devices=lambda kind: (SimpleNamespace(name="synthetic_physical"),),
            list_logical_devices=lambda kind: (SimpleNamespace(name="synthetic_logical"),),
        ),
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setitem(sys.modules, "tensorflow_probability", SimpleNamespace(__version__="synthetic_cpu_test"))
    monkeypatch.setitem(
        sys.modules,
        "bayesfilter.runtime.gpu_memory_policy",
        SimpleNamespace(configure_tensorflow_gpu_memory_growth=lambda *args, **kwargs: {"status": "synthetic_cpu_test"}),
    )
    return module, p1_runner


@pytest.mark.parametrize("failure", (None, "strict_arm", "manifest_write"))
def test_diagnostic_charges_the_whole_attempt_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    module, p1_runner = _synthetic_runtime(tmp_path, monkeypatch)
    if failure == "strict_arm":
        def run_arm(**kwargs):
            if kwargs["arm_name"] == "strict":
                raise RuntimeError("synthetic strict-arm failure")
            return _arm_result()
        monkeypatch.setattr(module, "_run_arm", run_arm)
    elif failure == "manifest_write":
        original_write = module._write_json

        def write_json(path, payload):
            if path.name == "run_manifest.json":
                raise RuntimeError("synthetic manifest failure")
            original_write(path, payload)
        monkeypatch.setattr(module, "_write_json", write_json)
    output = tmp_path / "attempt"

    assert module.main(["--output-dir", str(output), "--max-seconds", "100"]) == (
        0 if failure is None else 2
    )
    ledger = json.loads(p1_runner.CAMPAIGN_LEDGER_PATH.read_text(encoding="utf-8"))
    attempt = ledger["attempts"][-1]
    assert len(attempt["arms"]) == 1
    assert attempt["arms"][0]["arm"] == "readiness_diagnostic"
    assert attempt["consumed_seconds"] > 0.0
    assert ledger["reserved_seconds"] == 0.0
    assert ledger["consumed_seconds"] == pytest.approx(
        p1_runner.HISTORICAL_CONSUMED_SECONDS + attempt["consumed_seconds"]
    )
    assert ledger["total_budget_seconds"] == p1_runner.HISTORICAL_MATERIAL_CAP_SECONDS
    if failure is None:
        receipt = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        assert receipt["source_hashes"]["controller"]
        assert receipt["timing"]["p1_required_chunks_per_arm"] == 6
        assert (
            receipt["runtime_provenance"]["trust_basis"]
            == module.MANAGED_SESSION_TRUST_BASIS
        )
        run_start = json.loads((output / "run_start.json").read_text(encoding="utf-8"))
        assert run_start["runtime_provenance"]["python"]["executable"]
    else:
        receipt = json.loads((output / "failure.json").read_text(encoding="utf-8"))
        assert (
            receipt["runtime_provenance"]["trust_basis"]
            == module.MANAGED_SESSION_TRUST_BASIS
        )


def test_diagnostic_cannot_mint_an_additional_campaign_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, p1_runner = _synthetic_runtime(tmp_path, monkeypatch)

    def unexpected_arm(**kwargs):
        pytest.fail("an over-budget diagnostic must not execute an arm")

    monkeypatch.setattr(module, "_run_arm", unexpected_arm)
    output = tmp_path / "over-budget"

    assert module.main(["--output-dir", str(output), "--max-seconds", "5200"]) == 2
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert "remaining campaign budget" in failure["error"]
    ledger = json.loads(p1_runner.CAMPAIGN_LEDGER_PATH.read_text(encoding="utf-8"))
    assert ledger["consumed_seconds"] == p1_runner.HISTORICAL_CONSUMED_SECONDS
    assert all(attempt["status"] == "historical_failed" for attempt in ledger["attempts"])


def test_arm_timings_and_real_archive_feed_the_complete_forecast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tf = pytest.importorskip("tensorflow")
    from bayesfilter.inference import neutra_hmc

    module = _load(DIAGNOSTIC_PATH, "phase9b_readiness_arm_test")
    p1_runner = _load(P1_PATH, "phase9b_readiness_arm_p1_test")
    monkeypatch.setattr(p1_runner, "ROOT", tmp_path)
    profile = SimpleNamespace(initial_state_bank=tf.zeros((4, 4), tf.float64))
    chart = SimpleNamespace(forward_batch=lambda values: values)
    handoff = SimpleNamespace(transformed_adapter=object(), step_size=0.1, num_leapfrog_steps=1)
    source_module = SimpleNamespace(
        _build_fresh_chart=lambda *args: ({1.0: chart}, [], []),
        _tune_scope=lambda *args, **kwargs: {"_live_handoff": handoff},
    )
    monkeypatch.setattr(p1_runner, "_arm_profile", lambda *args: profile)
    import bayesfilter.inference.tempered_target_tf as bridge_module

    monkeypatch.setattr(
        bridge_module, "make_q20_tempered_bridge",
        lambda *args, **kwargs: SimpleNamespace(target_signature=p1_runner.TARGET_SIGNATURE),
    )
    initial_states = []

    def build_program(**kwargs):
        def program(state, seed):
            initial_states.append(state)
            samples = state[tf.newaxis, :, :] + tf.random.stateless_normal(
                (kwargs["num_results"], 4, 4), seed=seed, dtype=tf.float64
            )
            return samples, {"target_log_prob": -tf.reduce_sum(tf.square(samples), axis=-1)}

        program.input_signature = (
            tf.TensorSpec((4, 4), tf.float64), tf.TensorSpec((2,), tf.int32)
        )
        program.experimental_get_tracing_count = lambda: 1
        return program

    monkeypatch.setattr(neutra_hmc, "_build_batched_hmc_program", build_program)
    monkeypatch.setattr(module, "_allocator_info", lambda framework: {"current": 0, "peak": 0})
    monkeypatch.setattr(module, "_xla_receipt", lambda *args: {"status": "synthetic_cpu_test_no_xla"})
    result = module._run_arm(
        tf=tf, p1_runner=p1_runner, source_module=source_module, output=tmp_path,
        arm_name="factor", chunk_results=500, started=time.monotonic(), max_seconds=300.0,
    )
    forecast = module._complete_schedule_forecast(
        {"factor": result, "strict": result}, startup_seconds=0.0, chunk_results=500
    )

    assert result["timing"]["first_compiled_call_seconds"] > 0.0
    assert result["timing"]["steady_state_chunk_seconds"] > 0.0
    assert result["timing"]["serialization_seconds"] > 0.0
    assert forecast["p1_complete_schedule_forecast_seconds"] > 0.0
    assert result["movement"]["first_compiled_call"]["per_chain_moved"] == (
        True,
        True,
        True,
        True,
    )
    assert result["movement"]["steady_state_call"]["per_chain_moved"] == (
        True,
        True,
        True,
        True,
    )
    assert len(result["sample_archives"]) == 2
    for receipt in result["sample_archives"]:
        archive = json.loads((tmp_path / receipt["path"]).read_text(encoding="utf-8"))
        assert archive["warmup_excluded_from_posterior"] is True
        assert len(archive["latent_samples"]) == 500
    first_archive = json.loads((tmp_path / result["sample_archives"][0]["path"]).read_text(encoding="utf-8"))
    assert initial_states[1].numpy().tolist() == first_archive["latent_samples"][-1]


@pytest.mark.parametrize(
    ("stationary_call", "message"),
    (
        (1, "first compiled controller call had a stationary chain"),
        (2, "steady-state controller call had a stationary chain"),
    ),
)
def test_arm_rejects_a_stationary_controller_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stationary_call: int,
    message: str,
) -> None:
    tf = pytest.importorskip("tensorflow")
    from bayesfilter.inference import neutra_hmc

    module = _load(DIAGNOSTIC_PATH, "phase9b_readiness_stationary_arm_test")
    p1_runner = _load(P1_PATH, "phase9b_readiness_stationary_p1_test")
    monkeypatch.setattr(p1_runner, "ROOT", tmp_path)
    profile = SimpleNamespace(initial_state_bank=tf.zeros((4, 4), tf.float64))
    chart = SimpleNamespace(forward_batch=lambda values: values)
    handoff = SimpleNamespace(transformed_adapter=object(), step_size=0.1, num_leapfrog_steps=1)
    source_module = SimpleNamespace(
        _build_fresh_chart=lambda *args: ({1.0: chart}, [], []),
        _tune_scope=lambda *args, **kwargs: {"_live_handoff": handoff},
    )
    monkeypatch.setattr(p1_runner, "_arm_profile", lambda *args: profile)
    import bayesfilter.inference.tempered_target_tf as bridge_module

    monkeypatch.setattr(
        bridge_module, "make_q20_tempered_bridge",
        lambda *args, **kwargs: SimpleNamespace(target_signature=p1_runner.TARGET_SIGNATURE),
    )
    call_count = 0

    def build_program(**kwargs):
        def program(state, seed):
            nonlocal call_count
            del seed
            call_count += 1
            state_value = state if call_count == stationary_call else state + tf.ones_like(state)
            samples = tf.broadcast_to(
                state_value[tf.newaxis, :, :],
                (kwargs["num_results"], 4, 4),
            )
            return samples, {"target_log_prob": -tf.reduce_sum(tf.square(samples), axis=-1)}

        program.input_signature = (
            tf.TensorSpec((4, 4), tf.float64), tf.TensorSpec((2,), tf.int32)
        )
        program.experimental_get_tracing_count = lambda: 1
        return program

    monkeypatch.setattr(neutra_hmc, "_build_batched_hmc_program", build_program)
    monkeypatch.setattr(module, "_allocator_info", lambda framework: {"current": 0, "peak": 0})
    monkeypatch.setattr(module, "_xla_receipt", lambda *args: {"status": "synthetic_cpu_test_no_xla"})

    with pytest.raises(module.ReadinessDiagnosticError, match=message):
        module._run_arm(
            tf=tf, p1_runner=p1_runner, source_module=source_module, output=tmp_path,
            arm_name="factor", chunk_results=500, started=time.monotonic(), max_seconds=300.0,
        )
