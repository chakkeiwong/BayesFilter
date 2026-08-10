from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference.hmc import (
    FullChainHMCConfig,
    _target_status_telemetry_diagnostics,
    _trace_fn_for_config,
)


class _TelemetryAdapter:
    def target_status_telemetry(self, state):
        batch = tf.shape(state)[0]
        return {
            "status_code": tf.zeros([batch], tf.int32),
            "valid_pre_regularized_score": tf.ones([batch], tf.bool),
            "floor_count_value": tf.zeros([batch], tf.int32),
            "min_innovation_eigenvalue": tf.ones([batch], tf.float64),
            "innovation_condition_estimate": tf.ones([batch], tf.float64),
            "diagnostic_limits": "metadata must not enter TensorArray",
        }


def _kernel_results():
    accepted = SimpleNamespace(target_log_prob=tf.zeros([2], tf.float64))
    proposed = SimpleNamespace(target_log_prob=tf.zeros([2], tf.float64))
    return SimpleNamespace(
        is_accepted=tf.ones([2], tf.bool),
        log_accept_ratio=tf.zeros([2], tf.float64),
        accepted_results=accepted,
        proposed_results=proposed,
    )


def test_target_status_trace_filters_non_tensor_metadata():
    config = FullChainHMCConfig(
        num_results=1,
        num_burnin_steps=1,
        step_size=0.1,
        num_leapfrog_steps=1,
        seed=(1, 2),
        trace_policy="standard",
        target_status_trace_policy="per_chain_step",
        chain_execution_mode="eager",
    )
    trace = _trace_fn_for_config(config, adapter=_TelemetryAdapter())(
        tf.zeros([2, 3], tf.float64), _kernel_results()
    )
    telemetry = trace["target_status_telemetry"]
    assert set(telemetry) == {
        "status_code",
        "valid_pre_regularized_score",
        "floor_count_value",
        "min_innovation_eigenvalue",
        "innovation_condition_estimate",
    }
    assert all(isinstance(value, tf.Tensor) for value in telemetry.values())


def test_target_status_trace_accepts_core_only_target_schema():
    class CoreOnly(_TelemetryAdapter):
        def target_status_telemetry(self, state):
            payload = super().target_status_telemetry(state)
            payload.pop("min_innovation_eigenvalue")
            payload.pop("innovation_condition_estimate")
            return payload

    config = FullChainHMCConfig(
        num_results=1,
        num_burnin_steps=1,
        step_size=0.1,
        num_leapfrog_steps=1,
        seed=(1, 2),
        trace_policy="standard",
        target_status_trace_policy="per_chain_step",
        chain_execution_mode="eager",
    )
    telemetry = _trace_fn_for_config(config, adapter=CoreOnly())(
        tf.zeros([2, 3], tf.float64), _kernel_results()
    )["target_status_telemetry"]
    assert set(telemetry) == {
        "status_code",
        "valid_pre_regularized_score",
        "floor_count_value",
    }


def test_target_status_diagnostics_accepts_core_only_target_schema():
    telemetry = {
        "status_code": tf.zeros([2, 3], tf.int32),
        "valid_pre_regularized_score": tf.ones([2, 3], tf.bool),
        "floor_count_value": tf.zeros([2, 3], tf.int32),
    }
    summary = _target_status_telemetry_diagnostics(telemetry)
    assert int(summary["trace_entry_count"].numpy()) == 6
    assert bool(summary["all_status_valid"].numpy()) is True
    assert "min_min_innovation_eigenvalue" not in summary
    assert "max_innovation_condition_estimate" not in summary


def test_target_status_diagnostics_rejects_partial_conditioning_schema():
    telemetry = {
        "status_code": tf.zeros([2, 3], tf.int32),
        "valid_pre_regularized_score": tf.ones([2, 3], tf.bool),
        "floor_count_value": tf.zeros([2, 3], tf.int32),
        "min_innovation_eigenvalue": tf.ones([2, 3], tf.float64),
    }
    with pytest.raises(ValueError, match="both present or both absent"):
        _target_status_telemetry_diagnostics(telemetry)
