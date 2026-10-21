"""Verify retained-consumer authority, continuation and warmup exclusion."""
import dataclasses
import json
import math
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference import tune_hmc_kernel
from bayesfilter.inference.neural_force_hmc import PositionOnlyForceEvaluation, TargetPotentialEvaluation
from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import bind_components
from bayesfilter.testing.surrogate_hmc_m13_public_tuning import tuning_config
from bayesfilter.testing.surrogate_hmc_m13_posterior_assessment import METHOD, LOOKS
from bayesfilter.testing.surrogate_hmc_m13_retained import consume_chunks, checked_tuning_report


def test_retained_consumer_calls_verified_public_runner_and_excludes_warmup(tmp_path, monkeypatch):
    variance = math.log(2.)**2
    @tf.function(input_signature=[tf.TensorSpec([None, 2], tf.float64)], jit_compile=True)
    def force(x):
        return PositionOnlyForceEvaluation(x/variance, tf.ones(tf.shape(x)[:-1], tf.bool))
    @tf.function(input_signature=[tf.TensorSpec([None, 2], tf.float64)], jit_compile=True)
    def endpoint(x):
        valid = tf.ones(tf.shape(x)[:-1], tf.bool)
        return TargetPotentialEvaluation(tf.reduce_sum(x*x, -1)/(2*variance),
            dict(value_valid=valid, force_valid=valid, numerical_failure=~valid,
                 outside_declared_support=~valid))
    scope = "Gaussian-retained-consumer-mechanics-only"
    binding = bind_components(force, endpoint, scope)
    adapter = SimpleNamespace(target_scope=scope, adapter_signature=lambda: scope+":"+binding.binding_hash)
    config = tuning_config(scope)
    config = dataclasses.replace(config, nomination=dataclasses.replace(config.nomination,
        mass_window_results=(16, 32), step_adaptation_results=64, verification_results=64),
        warmup_results=2000)
    tuning = tune_hmc_kernel(adapter=adapter, initial_position=(0., 0.), config=config,
        parameter_scales=(math.log(2.), math.log(2.)),
        output_dir=tmp_path/"tuning", runner_binding=binding)
    assert tuning.passed, "The Gaussian public-tuner fixture must itself pass fresh verification."
    reference = dict(reference_sha256="analytic-Gaussian-test-only",
        mean=[0., 0.], sd=[math.log(2.)]*2, cuts=[[-.467516, 0., .467516]]*2,
        empirical_allowance=[0.]*10)
    calibration = dict(method=METHOD, qualified=True, looks=list(LOOKS),
        reference_sha256=reference["reference_sha256"])
    # The numerical archive is real. The comparison hook inspects exact retained
    # shape and returns at the first permitted look; it grants no scientific claim.
    from bayesfilter.testing import surrogate_hmc_m13_retained as consumer
    counts = []
    def compare(points, ref, *, calibration):
        counts.append(points.shape.as_list())
        return dict(status="EQUIVALENT", rhat=[1., 1.], diagnostic_test_only=True)
    monkeypatch.setattr(consumer, "assess", compare)
    result = consume_chunks(tuning, binding, tmp_path/"retained", reference, calibration)
    assert counts == [[4, 1000, 2]]
    assert len(result["chunks"]) == 4
    assert result["draws_per_chain"] == 1000
    for index, row in enumerate(result["chunks"]):
        archive = json.loads(open(row["manifest"]).read())
        assert archive["config"]["num_results"] == 250
        assert archive["health_passed"]
        assert archive["chunk_graph_trace_count"] == 1
        if index:
            assert archive["continuation_manifest"] == result["chunks"][index-1]["manifest"]


def test_retained_rejects_unverified_public_result_before_sampling(tmp_path):
    payload = dict(source_unchanged=True, stages={"m13-public-tuning":
        dict(execution="COMPLETED", result=dict(qualified=False))})
    (tmp_path/"report.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="verified public tuning"):
        checked_tuning_report(tmp_path)


def test_retained_rejects_mismatched_calibration_before_runner(tmp_path):
    with pytest.raises(ValueError, match="uncertainty qualification"):
        consume_chunks(None, None, tmp_path/"retained",
            dict(reference_sha256="reference"), dict(qualified=True, method=METHOD,
            looks=list(LOOKS), reference_sha256="wrong"))
    assert not (tmp_path/"retained").exists()
