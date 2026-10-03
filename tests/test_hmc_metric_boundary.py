"""Metric-boundary and failed-window regression diagnostics; CPU reference tests."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import hmc_warmup


def _finite_trace():
    return {
        "is_accepted": np.ones(4, dtype=bool),
        "log_accept_ratio": np.zeros(4),
        "target_log_prob": np.full(4, -1.),
        "step_size": np.full(4, .1),
        "proposed_step_size": np.full(4, .1),
        "consumed_step_size": np.full(4, .1),
    }


def _validate(trace):
    return hmc_warmup._validate_operational_window_trace(
        trace=trace, expected_draw_count=4, step_size_upper_bound=.2,
        target_status_trace_policy="none")


@pytest.mark.parametrize("field", ["log_accept_ratio", "target_log_prob", "step_size",
                                   "proposed_step_size", "consumed_step_size"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_failed_window_names_nonfinite_component_and_first_index(field, invalid):
    trace = _finite_trace()
    assert _validate(trace)["epsilon_end"] == .1
    trace[field][[1, 3]] = invalid
    with pytest.raises(ValueError, match=field + ": count=2, first_index=1"):
        _validate(trace)


def test_failed_window_preserves_every_nonfinite_component():
    trace = _finite_trace()
    trace["log_accept_ratio"][2] = -np.inf
    trace["proposed_step_size"][1] = np.inf
    with pytest.raises(ValueError) as caught:
        _validate(trace)
    message = str(caught.value)
    assert "log_accept_ratio: count=1, first_index=2" in message
    assert "proposed_step_size: count=1, first_index=1" in message


@pytest.mark.parametrize("log_average", [35., 70., 1000.])
def test_metric_boundary_clips_before_exponentiation(log_average):
    assert hmc_warmup._bounded_metric_boundary_step(
        log_average, step_size_upper_bound=.0004412119158251195) == .0004412119158251195


@pytest.mark.parametrize("step", [.00001, .02, .3])
def test_metric_boundary_unsaturated_average_is_exactly_unchanged(step):
    log_average = tf.math.log(tf.constant(step, tf.float64))
    assert hmc_warmup._bounded_metric_boundary_step(
        log_average, step_size_upper_bound=.5) == float(tf.exp(log_average).numpy())


@pytest.mark.parametrize("log_average", [np.nan, np.inf, -np.inf, -1000.])
def test_metric_boundary_rejects_invalid_internal_average(log_average):
    with pytest.raises(ValueError, match="metric-boundary"):
        hmc_warmup._bounded_metric_boundary_step(log_average, step_size_upper_bound=.5)


@pytest.mark.parametrize("invalid_average", [False, True])
def test_live_metric_probe_starts_bounded_but_can_qualify_larger_step(monkeypatch, invalid_average):
    from tests.test_hmc_warmup import _GaussianAdapter, _transform
    from bayesfilter.inference.hmc_coordinates import WarmupTrajectoryPolicy
    from bayesfilter.inference.hmc_tuning import WindowedMassAdaptationConfig

    observed = []
    find = hmc_warmup.find_reasonable_epsilon

    def capture(**kwargs):
        result = find(**kwargs)
        observed.append((kwargs, result))
        return result

    monkeypatch.setattr(hmc_warmup, "find_reasonable_epsilon", capture)
    if invalid_average:
        def invalid_nomination(*args, **kwargs):
            raise ValueError("metric-boundary log average must be a finite scalar")
        monkeypatch.setattr(hmc_warmup, "_bounded_metric_boundary_step", invalid_nomination)
    result = hmc_warmup.run_operational_windowed_warmup(
        adapter=_GaussianAdapter(np.eye(2)), initial_transform=_transform(np.eye(2)),
        initial_canonical_theta=np.array([.4, -.3]), initial_step_size=.001,
        initial_step_size_upper_bound=.001, initial_step_qualification_source="test_fixed_ceiling",
        trajectory_policy=WarmupTrajectoryPolicy(3, 16),
        config=WindowedMassAdaptationConfig(warmup_steps=112, initial_buffer=16,
            final_buffer=32, first_window_size=64, min_window_samples=32,
            metric_evidence_policy="finite_window"),
        target_accept_prob=.70, seed=(20260920, 51), target_scope="hmc_warmup_gaussian",
        chain_execution_mode="tf_function")
    if invalid_average:
        assert observed == []
        assert result.operational_metric_update_count == 0
        decision = result.windows[1].metric_decision
        assert decision.outcome == "candidate_metric_rejected"
        assert decision.report["candidate_rejection_stage"] == "reasonable_epsilon"
        assert decision.report["incumbent_metric_retained"] is True
        return
    assert len(observed) == 1  # The supplied initial ceiling skips only the initial probe.
    kwargs, probe = observed[0]
    assert kwargs["initial_step_size"] == .001
    assert kwargs["num_leapfrog_steps"] == 3
    assert kwargs["momentum_probe_count"] == 4
    assert kwargs["seed"] == hmc_warmup._seed((20260920, 51), 1, lane=3)
    assert probe.passed and probe.selected_step_size > .001
    assert result.operational_metric_update_count == 1
    update = result.windows[1]
    assert update.metric_decision.report["candidate_step_nomination"]["unbounded_log_average"] > np.log(.001)
    assert update.next_reasonable_epsilon.payload() == probe.payload()


def _probe_payload():
    import json
    attempt = hmc_warmup.ReasonableEpsilonAttempt(
        .1, .7, True, (1, 2), num_leapfrog_steps=3,
        probe_seeds=((1, 2), (1, 3)), minimum_acceptance_probability=.6,
        maximum_acceptance_probability=.8)
    return json.loads(json.dumps(hmc_warmup.ReasonableEpsilonResult("passed", .1, (attempt,)).payload()))


def test_reasonable_step_reader_accepts_current_and_historical_writer_shapes():
    from bayesfilter.inference.hmc_tuning_artifacts import _validate_reasonable_epsilon_payload
    payload = _probe_payload()
    assert _validate_reasonable_epsilon_payload(payload, name="test") == .1
    payload.pop("qualification_source")
    for key in ("num_leapfrog_steps", "probe_count", "probe_seeds",
                "minimum_acceptance_probability", "maximum_acceptance_probability"):
        payload["attempts"][0].pop(key)
    assert _validate_reasonable_epsilon_payload(payload, name="test") == .1


@pytest.mark.parametrize("corruption", ["count", "duplicate_seed", "first_seed", "L", "minimum",
                                       "missing_field", "unknown_field", "qualification"])
def test_reasonable_step_reader_rejects_corrupt_probe_metadata(corruption):
    from bayesfilter.inference.hmc_tuning_artifacts import _validate_reasonable_epsilon_payload
    payload = _probe_payload()
    attempt = payload["attempts"][0]
    if corruption == "count":
        attempt["probe_count"] = 3
    elif corruption == "duplicate_seed":
        attempt["probe_seeds"][1] = [1, 2]
    elif corruption == "first_seed":
        attempt["probe_seeds"][0] = [5, 6]
    elif corruption == "L":
        attempt["num_leapfrog_steps"] = 0
    elif corruption == "minimum":
        attempt["minimum_acceptance_probability"] = .9
    elif corruption == "missing_field":
        attempt.pop("probe_count")
    elif corruption == "unknown_field":
        attempt["unknown"] = True
    else:
        payload["qualification_source"] = "fabricated"
    with pytest.raises(ValueError):
        _validate_reasonable_epsilon_payload(payload, name="test")


@pytest.mark.parametrize("change", ["L", "seeds", "format"])
def test_reasonable_step_reader_keeps_one_fixed_probe_design(change):
    import copy
    from bayesfilter.inference.hmc_tuning_artifacts import _validate_reasonable_epsilon_payload
    payload = _probe_payload()
    earlier = copy.deepcopy(payload["attempts"][0])
    earlier["step_size"] = .05
    payload["attempts"].insert(0, earlier)
    assert _validate_reasonable_epsilon_payload(payload, name="test") == .1
    if change == "L":
        earlier["num_leapfrog_steps"] = 5
    elif change == "seeds":
        earlier["probe_seeds"][1] = [9, 10]
    else:
        for key in ("num_leapfrog_steps", "probe_count", "probe_seeds",
                    "minimum_acceptance_probability", "maximum_acceptance_probability"):
            earlier.pop(key)
    with pytest.raises(ValueError, match="changed between attempts|formats cannot be mixed"):
        _validate_reasonable_epsilon_payload(payload, name="test")
