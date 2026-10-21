"""CPU-only consumer mechanics; these tests make no M13 posterior claim."""
import dataclasses
import json
import math
from types import SimpleNamespace

import tensorflow as tf

from bayesfilter.inference.neural_force_hmc import PositionOnlyForceEvaluation, TargetPotentialEvaluation
from bayesfilter.testing import surrogate_hmc_m13_public_tuning as consumer
from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import bind_components


def test_prospective_scope_and_metric_protection():
    config = consumer.tuning_config("M13-unit-test")
    assert config.nomination.mass_window_results == (128, 256)
    assert config.nomination.max_leapfrog_steps == 4
    assert config.nomination.use_xla
    assert config.warmup_results == 500
    assert config.verification_results == 1000
    assert config.fresh_seed != config.nomination.seed
    assert len(set(config.initial_chain_positions)) == 4
    assert consumer.metric_guard_diagnostic(config)["passed"]


def test_consumer_uses_public_authority_and_fresh_archive(tmp_path, monkeypatch):
    """Exercise the consumer -> public tuner -> native kernel call chain."""
    variance = math.log(2.)**2

    @tf.function(input_signature=[tf.TensorSpec([None, 2], tf.float64)],
                 jit_compile=True, autograph=False)
    def force(x):
        return PositionOnlyForceEvaluation(x / variance, tf.ones(tf.shape(x)[:-1], tf.bool))

    @tf.function(input_signature=[tf.TensorSpec([None, 2], tf.float64)],
                 jit_compile=True, autograph=False)
    def endpoint(x):
        valid = tf.ones(tf.shape(x)[:-1], tf.bool)
        return TargetPotentialEvaluation(tf.reduce_sum(x*x, -1)/(2*variance),
            dict(value_valid=valid, force_valid=valid, numerical_failure=~valid,
                 outside_declared_support=~valid))

    scope = "M13-consumer-Gaussian-mechanics-only"
    binding = bind_components(force, endpoint, scope)
    adapter = SimpleNamespace(target_scope=scope, adapter_signature=lambda: scope+":"+binding.binding_hash)
    fixture = SimpleNamespace(observation_checksum="diagnostic", observations=tf.zeros([2, 2]),
                              model_definition=lambda: {"role": "Gaussian consumer test only"})
    selection_path = tmp_path / "selection.json"
    selection = {"multiplier": .1, "controls": {}}
    selection_path.write_text(json.dumps(selection))
    monkeypatch.setattr(consumer, "selected_components", lambda: (
        adapter, binding, SimpleNamespace(manifest={"diagnostic": True}), fixture, selection, selection_path))
    config = consumer.tuning_config(scope)
    config = dataclasses.replace(config, nomination=dataclasses.replace(config.nomination,
        mass_window_results=(16, 32), step_adaptation_results=64, verification_results=64),
        warmup_results=100)
    monkeypatch.setattr(consumer, "tuning_config", lambda _: config)
    monkeypatch.setattr(tf.config.experimental, "tensor_float_32_execution_enabled", lambda: True)
    result = consumer.public_tuning_stage(tmp_path / "result" / "report.json")
    # A stochastic nomination or R-hat rejection is valid; no rejection may
    # issue posterior qualification or silently bypass public verification.
    assert result["posterior_qualified"] is False
    assert result["exact_score_admission"] is False
    assert result["status"] in ("M13_PUBLIC_TUNING_VERIFIED", "M13_PUBLIC_TUNING_REPAIR_REQUIRED")
    nomination = json.loads((tmp_path / "result/tuning/nomination/manifest.json").read_text())
    assert nomination["runner_binding"] == json.loads(json.dumps(binding.payload()))
    if "manifest" in result.get("tuning", {}):
        manifest = json.loads(open(result["tuning"]["manifest"]).read())
        assert manifest["artifact_authority"] is True
        assert manifest["config"]["verification_results"] == 1000
        assert result["qualified"] is manifest["handoff_eligible"]
        assert result["tuning"]["fresh_force_fallback_counts"] == {"warmup": 0, "verification": 0}


def test_reviewed_orchestration_drift_does_not_admit_failed_or_changed_worker(tmp_path):
    import pytest
    from bayesfilter.inference.surrogate_hmc_program import file_hash
    parent = dict(stages={"m13-materiality-repair": dict(execution="COMPLETED",
        result=dict(qualified=True))})
    parent_path = tmp_path / "report.json"
    parent_path.write_text(json.dumps(parent))
    selection_path = tmp_path / "m13-materiality-repair/selection.json"
    selection_path.parent.mkdir()
    selection_path.write_text("{}")
    with pytest.raises(FileNotFoundError):
        consumer.checked_control_run(tmp_path)
    review = dict(parent_report_sha256=file_hash(parent_path),
        selection_sha256=file_hash(selection_path), worker_source_hashes_unchanged=True,
        original_snapshot_verified=True, repair_dispatch_ast_unchanged=True)
    review_path = tmp_path / "source-change-review.json"
    review_path.write_text(json.dumps(review))
    consumer.checked_control_run(tmp_path)
    selection_path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="reconciliation"):
        consumer.checked_control_run(tmp_path)
    parent["stages"]["m13-materiality-repair"]["execution"] = "FAILED"
    parent_path.write_text(json.dumps(parent))
    with pytest.raises(ValueError, match="worker"):
        consumer.checked_control_run(tmp_path)
