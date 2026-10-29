"""Diagnostic tests: exact input preservation, canonical wiring, and budget."""
import json
from types import MappingProxyType, SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_range_bearing_target_tf as target_module
from bayesfilter.testing import surrogate_hmc_m13_correction_localize as probe


def baseline():
    a = tf.constant([[.13, 0., 0., 0.], [0., .27, 0., 0.],
                     [0., 0., .39, 0.], [0., 0., 0., .51]], tf.float64)
    model = target_module.make_range_bearing_model(tf.cast(a, tf.float32),
        .13*tf.eye(4), .27*tf.eye(2))
    inputs = dict(initial_states=tf.ones([3, 4]), initial_covariances=.39*tf.eye(4, batch_shape=[3]),
                  noises=tf.zeros([2, 3, 4]), observations=.51*tf.ones([2, 2]))
    base = target_module.M13LikelihoodTarget(model, MappingProxyType(inputs),
        MappingProxyType(dict(reset_design=tf.ones([3, 4]), reset_ridge=1e-5)),
        MappingProxyType(dict(dtype="float32", tf32=True)), "test")
    return base, a


def test_fp64_reference_lifts_every_realized_input_and_coefficient():
    base, a = baseline()
    lifted = probe.arithmetic_target(base, a, dtype=tf.float64, tf32=False)
    for key, value in base.inputs.items():
        tf.debugging.assert_equal(lifted.inputs[key], tf.cast(value, tf.float64))
    tf.debugging.assert_equal(lifted.controls["reset_design"], tf.cast(base.controls["reset_design"], tf.float64))
    for key in ("process_covariance", "observation_covariance"):
        tf.debugging.assert_equal(getattr(lifted.model, key), tf.cast(getattr(base.model, key), tf.float64))
    actual = lifted.model.transition_mean_fn(tf.zeros([4, 2], tf.float64), tf.eye(4, dtype=tf.float64))
    tf.debugging.assert_equal(actual, tf.transpose(tf.cast(tf.cast(a, tf.float32), tf.float64)))
    assert bool(tf.reduce_any(actual != tf.transpose(a)))
    assert lifted.manifest["base_identity"] == base.identity
    assert lifted.manifest["diagnostic_role"] == "exact_lift_of_saved_FP32_inputs"


@pytest.mark.parametrize("key", ["design_partition", "input_sha256", "source_sha256", "settings"])
def test_saved_binding_rejects_changed_design_inputs_source_or_controls(key):
    fields = ("fixture", "horizon", "particle_count", "dtype", "tf32", "jit_compile",
              "settings", "design_partition", "seeds", "input_sha256", "source_sha256")
    saved = {k: "original" for k in fields}
    probe.verify_saved_binding(SimpleNamespace(manifest=saved), saved)
    with pytest.raises(ValueError, match=key):
        probe.verify_saved_binding(SimpleNamespace(manifest=saved | {key: "changed"}), saved)


def test_lifted_consumer_calls_the_general_canonical_implementation(monkeypatch):
    base, a = baseline()
    target = probe.arithmetic_target(base, a, dtype=tf.float64, tf32=False)
    calls = []
    def canonical(model, theta, directions, **kwargs):
        calls.append((model, theta.dtype, directions.shape, kwargs))
        return tf.zeros([2], tf.float64), tf.zeros([2, 2], tf.float64), dict(
            value_valid=tf.constant([True, True]), force_valid=tf.constant([True, True]))
    monkeypatch.setattr(target_module, "canonical_batch_fused_value_score", canonical)
    target.evaluate(tf.zeros([2, 2], tf.float64), filter_moments=True, all_correction_diagnostics=True)
    assert len(calls) == 1 and calls[0][0] is target.model and calls[0][1] == tf.float64
    assert calls[0][3]["return_filter_moments"] and calls[0][3]["return_all_correction_diagnostics"]


def test_master_worker_wiring_and_remaining_control_reservation(tmp_path, monkeypatch):
    from bayesfilter.inference import surrogate_hmc_closure_program as program
    from bayesfilter.testing import surrogate_hmc_closure_diagnostics as worker
    jobs = program.jobs("m13-correction-localize")
    assert [j["name"] for j in jobs] == ["review", "m13-correction-localize"]
    assert jobs[1]["plan"] == probe.PLAN and jobs[1]["device"] == "gpu"
    assert sum(j["cpu"] for j in jobs) <= 352.445851
    assert "tests/test_surrogate_hmc_m13_correction_localize.py" in jobs[0]["command"]
    with pytest.raises(ValueError, match="separate bounded pass"):
        program.jobs("m13-correction-localize,m13-controls")
    seen = []
    def stage(output):
        seen.append(output)
        return dict(status="WIRING_ONLY", qualified=False)
    monkeypatch.setattr(probe, "correction_localization_stage", stage)
    output = tmp_path/"report.json"
    worker.main(["--stage", jobs[1]["name"], "--device", "cpu", "--output", str(output)])
    assert seen == [output]
    assert json.loads(output.read_text())["gpu_intentionally_hidden"]
