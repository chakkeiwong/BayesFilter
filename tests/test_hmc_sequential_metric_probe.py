"""Discarded local metric probes: intermediate health and configuration wiring."""
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference import hmc_warmup as warmup
from bayesfilter.inference.hmc_configuration import HMCKernelTuningConfig, _public_loop_config, _phase7_windowed_stage_config
from bayesfilter.inference.hmc_mass_adaptation import _windowed_mass_stage_internal_config
from bayesfilter.inference.hmc_tuning_artifacts import _validate_reasonable_epsilon_payload
from tests.test_hmc_warmup import _GaussianAdapter


def _find(**kwargs):
    return warmup.find_reasonable_epsilon(adapter=_GaussianAdapter(tf.eye(2, dtype=tf.float64)),
        current_state=tf.constant([.4, -.2], tf.float64), initial_step_size=.5,
        seed=(20260920, 62), num_leapfrog_steps=3, momentum_probe_count=2, **kwargs)


def test_longer_probe_config_reaches_operational_warmup():
    public = HMCKernelTuningConfig(metric_probe_num_results=16)
    loop = _public_loop_config(public)
    stage = _phase7_windowed_stage_config(loop, attempt_index=0)
    internal = _windowed_mass_stage_internal_config(metric_probe_num_results=stage.metric_probe_num_results)
    for value in (public, loop, stage, internal):
        assert value.metric_probe_num_results == 16
        assert value.payload()["metric_probe_num_results"] == 16


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "16"])
def test_invalid_probe_count_rejected(value):
    with pytest.raises(ValueError, match="metric_probe_num_results"):
        HMCKernelTuningConfig(metric_probe_num_results=value)


def test_longer_probe_rejects_irrelevant_fixed_identity_configuration():
    with pytest.raises(ValueError, match="windowed"):
        HMCKernelTuningConfig(mass_policy="fixed_identity", metric_probe_num_results=16)


def test_one_step_branch_has_unchanged_results_and_never_calls_sequential_runner(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("one-transition branch called a sequential probe")
    monkeypatch.setattr(warmup, "_run_reasonable_epsilon_probe_chain", forbidden)
    assert _find().payload() == _find(probe_num_results=1).payload()
    assert "probe_num_results" not in _find().attempts[-1].payload()


@pytest.mark.parametrize("field", ["proposed_state", "proposed_target", "proposed_score", "momentum", "energy",
                                  "retained_state", "retained_target", "retained_score"])
def test_intermediate_nonfinite_cannot_be_hidden_by_finite_last_transition(monkeypatch, field):
    original = warmup._run_reasonable_epsilon_probe_chain

    def corrupt(**kwargs):
        states, trace = original(**kwargs)
        def middle(x):
            return tf.concat([x[:1], tf.fill(tf.shape(x[1:2]), tf.constant(float("nan"), x.dtype)), x[2:]], axis=0)
        if field == "proposed_state":
            trace = trace._replace(proposed_state=middle(trace.proposed_state))
        elif field == "retained_state":
            states = middle(states)
        else:
            retained = field.startswith("retained")
            name = "accepted_results" if retained else "proposed_results"
            result = getattr(trace, name)
            if field.endswith("target"):
                result = result._replace(target_log_prob=middle(result.target_log_prob))
            elif field.endswith("score"):
                result = result._replace(grads_target_log_prob=[middle(result.grads_target_log_prob[0])])
            elif field == "momentum":
                result = result._replace(final_momentum=[middle(result.final_momentum[0])])
            else:
                result = result._replace(log_acceptance_correction=middle(result.log_acceptance_correction))
            trace = trace._replace(**{name: result})
        return states, trace

    monkeypatch.setattr(warmup, "_run_reasonable_epsilon_probe_chain", corrupt)
    if field.startswith("retained"):
        with pytest.raises(warmup._ReasonableEpsilonSharedInvalidity, match="accepted or retained state"):
            _find(probe_num_results=3, max_attempts=1)
    else:
        result = _find(probe_num_results=3, max_attempts=1)
        assert not result.passed
        assert not result.attempts[0].finite
        assert result.attempts[0].probe_num_results == 3


def test_accepted_state_mismatch_is_fatal(monkeypatch):
    original = warmup._run_reasonable_epsilon_probe_chain
    def inconsistent(**kwargs):
        states, trace = original(**kwargs)
        return states + tf.constant(.01, tf.float64), trace
    monkeypatch.setattr(warmup, "_run_reasonable_epsilon_probe_chain", inconsistent)
    with pytest.raises(warmup._ReasonableEpsilonSharedInvalidity, match="accepted-state consistency"):
        _find(probe_num_results=3, max_attempts=1)


def test_sequential_probe_unclassified_execution_error_is_fatal(monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("shared execution failed")
    monkeypatch.setattr(warmup, "_run_reasonable_epsilon_probe_chain", broken)
    with pytest.raises(warmup._ReasonableEpsilonSharedInvalidity, match="HMC proposal execution failed"):
        _find(probe_num_results=3, max_attempts=1)


def test_sequential_probe_finite_control_and_durable_probe_design():
    result = _find(probe_num_results=3)
    assert result.passed
    payload = json.loads(json.dumps(result.payload()))
    assert all(row["probe_num_results"] == 3 for row in payload["attempts"])
    assert _validate_reasonable_epsilon_payload(payload, name="probe") == result.selected_step_size
    earlier = dict(payload["attempts"][0], probe_num_results=4)
    payload["attempts"].insert(0, earlier)
    with pytest.raises(ValueError, match="changed between attempts"):
        _validate_reasonable_epsilon_payload(payload, name="probe")


def test_retained_status_failure_precedes_invalid_proposal(monkeypatch):
    original = warmup._run_reasonable_epsilon_probe_chain
    calls = []

    def status(adapter, state, **kwargs):
        calls.append(state)
        return len(calls) > 1  # Initial state valid, first retained row invalid.

    def bad_proposal(**kwargs):
        states, trace = original(**kwargs)
        return states, trace._replace(proposed_state=tf.fill(tf.shape(states), tf.constant(float("nan"), tf.float64)))

    monkeypatch.setattr(warmup, "_target_status_failed", status)
    monkeypatch.setattr(warmup, "_run_reasonable_epsilon_probe_chain", bad_proposal)
    with pytest.raises(warmup._ReasonableEpsilonSharedInvalidity, match="retained target status"):
        _find(probe_num_results=3, max_attempts=1, target_status_trace_policy="per_chain_step")


@pytest.mark.parametrize("shared_failure", [False, "retained", "malformed_telemetry"])
def test_live_metric_boundary_uses_longer_probe_and_preserves_fatal_failures(monkeypatch, shared_failure):
    from bayesfilter.inference.hmc_coordinates import WarmupTrajectoryPolicy
    from bayesfilter.inference.hmc_tuning import WindowedMassAdaptationConfig
    from tests.test_hmc_warmup import _transform
    original = warmup.find_reasonable_epsilon
    calls = []
    def probe(**kwargs):
        calls.append(kwargs)
        if shared_failure:
            if shared_failure == "malformed_telemetry":
                raise ValueError("injected malformed target telemetry")
            raise warmup._ReasonableEpsilonSharedInvalidity("injected retained-state failure")
        return original(**kwargs)
    monkeypatch.setattr(warmup, "find_reasonable_epsilon", probe)
    kwargs = dict(adapter=_GaussianAdapter(tf.eye(2, dtype=tf.float64)),
        initial_transform=_transform(tf.eye(2, dtype=tf.float64)),
        initial_canonical_theta=tf.constant([.4, -.3], tf.float64), initial_step_size=.001,
        initial_step_size_upper_bound=.001, initial_step_qualification_source="test_fixed_ceiling",
        trajectory_policy=WarmupTrajectoryPolicy(3, 16),
        config=WindowedMassAdaptationConfig(warmup_steps=112, initial_buffer=16,
            final_buffer=32, first_window_size=64, min_window_samples=32,
            metric_evidence_policy="finite_window", metric_probe_num_results=3),
        target_accept_prob=.70, seed=(20260920, 51), target_scope="hmc_warmup_gaussian",
        chain_execution_mode="tf_function")
    if shared_failure:
        with pytest.raises(warmup._ReasonableEpsilonSharedInvalidity, match="injected"):
            warmup.run_operational_windowed_warmup(**kwargs)
    else:
        result = warmup.run_operational_windowed_warmup(**kwargs)
        assert result.operational_metric_update_count == 1
        attempt = result.windows[1].next_reasonable_epsilon.attempts[-1]
        assert attempt.probe_num_results == 3 and attempt.usable
        from bayesfilter.inference.hmc_tuning_artifacts import _validate_operational_warmup_payload
        _validate_operational_warmup_payload(json.loads(json.dumps(result.public_payload())))
    assert len(calls) == 1 and calls[0]["probe_num_results"] == 3
