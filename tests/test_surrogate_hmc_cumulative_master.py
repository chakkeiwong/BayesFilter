"""CPU-only accounting and call-chain checks; no M13 scientific promotion."""
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter import inference
from bayesfilter.inference import hmc_sequential_verification as sequential
from bayesfilter.inference import hmc_tensorflow_tuning as stage
from bayesfilter.inference.surrogate_hmc_program import file_hash, write_json
from bayesfilter.testing import surrogate_hmc_m13_cumulative as cumulative


@pytest.fixture(scope="module")
def gaussian_tuning(tmp_path_factory):
    from bayesfilter.inference.neural_force_hmc import PositionOnlyForceEvaluation, TargetPotentialEvaluation
    from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import bind_components
    from bayesfilter.testing.surrogate_hmc_m13_public_tuning import tuning_config
    variance = math.log(2.)**2
    @tf.function(input_signature=[tf.TensorSpec([None, 2], tf.float64)], jit_compile=True)
    def force(x):
        return PositionOnlyForceEvaluation(x/variance, tf.ones(tf.shape(x)[:-1], tf.bool))
    @tf.function(input_signature=[tf.TensorSpec([None, 2], tf.float64)], jit_compile=True)
    def endpoint(x):
        valid = tf.ones(tf.shape(x)[:-1], tf.bool)
        return TargetPotentialEvaluation(tf.reduce_sum(x*x, -1)/(2*variance),
            dict(value_valid=valid, force_valid=valid, numerical_failure=~valid, outside_declared_support=~valid))
    scope = "Gaussian-cumulative-accounting-mechanics-only"
    binding = bind_components(force, endpoint, scope)
    adapter = SimpleNamespace(target_scope=scope, adapter_signature=lambda: scope+":"+binding.binding_hash)
    cfg = tuning_config(scope)
    cfg = replace(cfg, nomination=replace(cfg.nomination, mass_window_results=(16, 32),
        step_adaptation_results=64, verification_results=64), warmup_results=2000)
    tuning = inference.tune_hmc_kernel(adapter=adapter, initial_position=(0., 0.), config=cfg,
        parameter_scales=(math.log(2.),)*2, output_dir=tmp_path_factory.mktemp("cumulative")/"tuning",
        runner_binding=binding)
    assert tuning.passed
    return tuning, binding


@pytest.fixture(autouse=True)
def unbudgeted_cpu_mechanics(monkeypatch):
    monkeypatch.setattr(sequential, "runtime_budget_guard", lambda: None)


def load(tuning, binding, resume_report=None):
    return cumulative.load_history(tuning.artifact_manifest_path,
        binding_hash=binding.binding_hash, resume_report=resume_report)


def no_runner(**kwargs):
    pytest.fail("a terminal saved history must not construct a sampler")


def test_passing_tuner_samples_finish_without_a_second_chain(tmp_path, monkeypatch, gaussian_tuning):
    tuning, binding = gaussian_tuning
    history = load(tuning, binding)
    assert history.count == cumulative.read(tuning.artifact_manifest_path)["sequence"]["counts"]["verification"]
    assert history.warmup >= 2000
    monkeypatch.setattr(inference, "build_retained_bound_hmc_archive_runner_from_tuning_result", no_runner)
    result = cumulative.consume_chunks(tuning, binding, tmp_path/"result")
    assert result["status"] == cumulative.COMPLETE
    assert result["draws_per_chain"] == history.count
    assert result["new_hmc_transitions"] == 0
    assert result["independent_validation_required"] is False
    assert result["posterior_reference_qualified"] is False


def test_low_ess_is_a_precision_warning_not_another_phase():
    metrics = dict(draws_per_chain=9000, rhat=dict(maximum=[1.0074, 1.0055]),
                   bulk_and_tail_ess_at_least_400=False)
    result = cumulative.decision(metrics, minimum=2000)
    assert result["sampling_qualified"] and result["warnings"]
    assert not cumulative.decision(metrics, minimum=2000, health_passed=False)["sampling_qualified"]
    for bad in (1.054, float("nan"), float("inf")):
        assert not cumulative.decision(dict(metrics, rhat=dict(maximum=[bad, 1.])), minimum=2000)["sampling_qualified"]


def test_real_continuation_uses_total_cap_and_terminal_partial_look(tmp_path, monkeypatch, gaussian_tuning):
    tuning, binding = gaussian_tuning
    initial = load(tuning, binding).count
    original_diagnostics = cumulative.diagnostics
    def failed_mixing(samples, accepts):
        result = original_diagnostics(samples, accepts)
        result["rhat"]["maximum"] = [1.1, 1.1]
        return result
    monkeypatch.setattr(cumulative, "diagnostics", failed_mixing)
    result = cumulative.consume_chunks(tuning, binding, tmp_path/"run", max_results=initial+500)
    assert result["status"] == "M13_POSTERIOR_SAMPLE_CAP_INCONCLUSIVE"
    assert result["draws_per_chain"] == initial+500
    assert result["new_draws_per_chain"] == 500 and len(result["chunks"]) == 2
    assert result["diagnostics"]["draws_per_chain"] == initial+500
    assert [row["draws_per_chain"] for row in result["assessments"]] == [initial, initial+500]
    # Every saved continuation must be loaded before assessing or selecting a prefix.
    replay = load(tuning, binding, tmp_path/"run/report.json")
    assert replay.count == initial+500 and replay.warmup == result["warmup_excluded_per_chain"]
    monkeypatch.setattr(inference, "build_retained_bound_hmc_archive_runner_from_tuning_result", no_runner)
    inspected = cumulative.run_history(replay, tmp_path/"inspect", maximum=initial+500)
    assert not inspected["sampling_qualified"] and inspected["draws_per_chain"] == initial+500
    assert inspected["new_hmc_transitions"] == 0


def test_budget_stop_assesses_every_saved_draw(tmp_path, monkeypatch, gaussian_tuning):
    tuning, binding = gaussian_tuning
    original = cumulative.diagnostics
    def not_yet(samples, accepts):
        result = original(samples, accepts)
        result["rhat"]["maximum"] = [1.05, 1.05]
        return result
    monkeypatch.setattr(cumulative, "diagnostics", not_yet)
    permits = iter((True, False))
    monkeypatch.setattr(sequential, "runtime_budget_guard", lambda: lambda *args: next(permits))
    result = cumulative.consume_chunks(tuning, binding, tmp_path/"budget")
    assert result["status"] == "M13_POSTERIOR_BUDGET_EXHAUSTED"
    assert result["new_draws_per_chain"] == 250
    assert result["diagnostics"]["draws_per_chain"] == result["draws_per_chain"]


@pytest.mark.parametrize("corruption", ["seed", "state", "binding", "checksum", "count"])
def test_replay_rejects_corrupted_continuation(tmp_path, gaussian_tuning, corruption):
    tuning, binding = gaussian_tuning
    runner = inference.build_retained_bound_hmc_archive_runner_from_tuning_result(tuning_result=tuning, runner_binding=binding)
    result = runner.run(inference.BoundRetainedHMCArchiveConfig(num_results=250, seed=cumulative.SEED,
        output_dir=tmp_path/"chunk", budget_provenance="CPU mechanics only"))
    path = Path(result["archive_manifest_path"])
    saved = cumulative.read(path)
    if corruption == "seed":
        saved["config"]["seed"][1] += 10
    elif corruption == "binding":
        saved["binding_hash"] = "wrong-target"
    elif corruption == "state":
        saved["initial_state"] = stage._write_tensor(path.parent/"changed.tfbin", tf.zeros([4, 2], tf.float64))
    write_json(path, saved)
    row = dict(manifest=str(path), sha256=file_hash(path), health_passed=True,
               no_force_fallback=True, checks=saved["checks"])
    if corruption == "checksum":
        row["sha256"] = "wrong-checksum"
    report = dict(tuning_manifest=tuning.artifact_manifest_path, binding_hash=binding.binding_hash,
        warmup_excluded=True, seed=list(cumulative.SEED), chunks=[row], draws_per_chain=250)
    if corruption == "count":
        report["draws_per_chain"] = 500
    write_json(tmp_path/"old.json", report)
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        load(tuning, binding, tmp_path/"old.json")


def test_bad_execution_cannot_be_reclassified(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    write_json(tmp_path/"run/report.json", dict(source_unchanged=True,
        execution=dict(exit_code=1, status="FAILED", accounting_complete=True)))
    with pytest.raises(ValueError, match="invalid saved master execution"):
        cumulative.reclassify_saved_run(tmp_path/"run", tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_cli_reclassify_never_launches_worker(tmp_path, monkeypatch):
    from bayesfilter.inference import surrogate_hmc_deadline_extension as master
    called = []
    monkeypatch.setattr(master, "worker", lambda *_: pytest.fail("no GPU worker in archive-only mode"))
    def reclassify(source, output):
        called.append((source, output))
        return dict(status="M13_POSTERIOR_MIXING_INCONCLUSIVE", draws_per_chain=12750,
                    diagnostics=dict(rhat=dict(maximum=[1.054, 1.009])))
    monkeypatch.setattr(cumulative, "reclassify_saved_run", reclassify)
    assert master.main(["reclassify", "--input", str(tmp_path/"in"), "--output", str(tmp_path/"out")]) == 0
    assert called == [(tmp_path/"in", tmp_path/"out")]



def test_legacy_recovery_stage_reuses_passing_prefix_without_sampling(tmp_path, monkeypatch, gaussian_tuning):
    from bayesfilter.testing import surrogate_hmc_m13_retained as legacy
    tuning, binding = gaussian_tuning
    scope = tuning.config.nomination.target_scope
    adapter = SimpleNamespace(target_scope=scope, adapter_signature=lambda: scope+":"+binding.binding_hash)
    reference = dict(reference_sha256="fixture-reference", observation_checksum="fixture-data")
    monkeypatch.setattr(legacy, "checked_tuning_report", lambda _: ({}, tuning.artifact_manifest_path))
    monkeypatch.setattr(legacy, "load_reference", lambda _: (reference, None, None))
    monkeypatch.setattr(legacy, "selected_components", lambda: (
        adapter, binding, None, SimpleNamespace(observation_checksum="fixture-data"), None, None))
    def forbidden(*args, **kwargs):
        pytest.fail("a passing cumulative prefix must not start a second chain")
    monkeypatch.setattr(legacy, "consume_chunks", forbidden)
    monkeypatch.setattr(inference, "build_retained_bound_hmc_archive_runner_from_tuning_result", forbidden)
    write_json(tmp_path/"report.json", {})
    write_json(tmp_path/"coverage.json", dict(qualified=True))
    write_json(tmp_path/"baseline.json", dict(health_passed=True, qualified=True, reference_sha256="fixture-reference"))
    result = legacy.retained_stage(tmp_path, tmp_path/"posterior", "fixture-reference",
        tmp_path/"coverage.json", tmp_path/"baseline.json")
    assert result["sampling_qualified"] and result["new_hmc_transitions"] == 0
    assert not result["posterior_reference_qualified"]
    assert result["draws_per_chain"] == load(tuning, binding).count


@pytest.mark.parametrize("passed", [True, False])
def test_legacy_cli_preserves_sampling_verdict_without_reporting_execution_failure(tmp_path, monkeypatch, passed):
    import importlib.util
    import json
    from bayesfilter.runtime import gpu_memory_policy
    from bayesfilter.testing import surrogate_hmc_m13_retained as legacy
    script = Path(__file__).resolve().parents[1]/"scripts/run_surrogate_hmc_m13_retained.py"
    spec = importlib.util.spec_from_file_location("legacy_retained_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(gpu_memory_policy, "configure_tensorflow_gpu_memory_growth", lambda _: {"mock_cpu_test": True})
    status = cumulative.COMPLETE if passed else "M13_POSTERIOR_MIXING_INCONCLUSIVE"
    monkeypatch.setattr(legacy, "retained_stage", lambda *a: dict(status=status, sampling_qualified=passed))
    args = SimpleNamespace(tuning_root=tmp_path, output=tmp_path/"output", reference=None, coverage=None, baseline_report=None)
    assert cli.worker(args) == 0
    report = json.loads((args.output/"report.json").read_text())
    assert report["status"] == status and report["sampling_qualified"] is passed
