"""Regression fixtures for bootstrap decisions; no posterior claims."""
from dataclasses import replace
import pytest
import tensorflow as tf

from bayesfilter.inference import hmc_bootstrap as bootstrap
from bayesfilter.inference.hmc_mass_adaptation import (
    _active_bootstrap_handoff_kernel_payload, _bootstrap_preflight_passed,
)
from tests.test_hmc_kernel_tuning_bootstrap import (
    _ToyGaussianAdapter, _config, _fake_result, _geometry,
)


@pytest.mark.parametrize("probability,binary,decision,direction", [
    (0.6846376853336343, 0.8125, "passed", None),
    (.70, .25, "passed", None),
    (.40, .9375, "repair", "reduce_epsilon_recompute_l"),
    (.90, .1875, "repair", "increase_epsilon_recompute_l"),
    (.40, .6875, "repair", "reduce_epsilon_recompute_l"),
    (.90, .6875, "repair", "increase_epsilon_recompute_l"),
])
def test_public_bootstrap_uses_probability_in_both_directions(probability, binary, decision, direction):
    result = bootstrap.run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(), geometry=_geometry(),
        config=_config(screen_num_results=16, max_repairs=0),
        run_full_chain=lambda *args: _fake_result(acceptance=probability, binary_rate=binary, count=16),
    )
    row = result.rounds[0]
    assert row.classification == decision
    assert row.repair_action == direction
    assert row.diagnostics["binary_acceptance_rate"] == binary
    assert row.diagnostics["mean_acceptance_probability"] == pytest.approx(probability)
    assert row.diagnostics["acceptance_probability_count"] == 16
    assert row.diagnostics["divergence_count"] is None


def test_reported_three_round_sequence_stops_before_unsafe_fourth_call():
    probabilities = [.9793739706652833, .8932222868851928, .6846376853336343]
    binary = [.875, .9375, .8125]
    calls = []

    def run(adapter, initial, cfg):
        i = len(calls)
        assert i < 3, "unsafe fourth screen must not be nominated"
        calls.append((cfg.step_size, cfg.num_leapfrog_steps))
        return _fake_result(acceptance=probabilities[i], binary_rate=binary[i], count=16)

    geometry = replace(_geometry(), initial_step_size=.28867513459481275,
                       target_trajectory_length=1.5)
    result = bootstrap.run_hmc_bootstrap_screen(adapter=_ToyGaussianAdapter(), geometry=geometry,
        config=_config(screen_num_results=16), run_full_chain=run)
    assert [row[1] for row in calls] == [6, 3, 3]
    assert result.rounds[2].unclamped_num_leapfrog_steps == 2
    assert result.rounds[2].clamp_direction == "min"
    assert result.selected_kernel_payload["step_size"] == 1.154700538379251
    assert result.selected_round_index == 2


def test_rejected_proposals_contribute_and_burnin_is_not_recorded():
    raw = _fake_result(acceptance=.7, binary_rate=.8125, count=16)
    probabilities = [0.9] * 13 + [0.05] * 3
    raw = replace(raw, trace={**raw.trace, "log_accept_ratio": tf.math.log(tf.constant(probabilities, tf.float64))})
    result = bootstrap.run_hmc_bootstrap_screen(adapter=_ToyGaussianAdapter(), geometry=_geometry(),
        config=_config(screen_num_results=16, screen_num_burnin_steps=4, max_repairs=0),
        run_full_chain=lambda *args: raw)
    assert result.rounds[0].diagnostics["mean_acceptance_probability"] == pytest.approx(sum(probabilities)/16)
    assert result.passed


@pytest.mark.parametrize("case", ["missing", "empty", "nan", "inf", "minus_inf", "shape", "burnin_included"])
def test_missing_or_malformed_probability_cannot_fall_back_to_binary(case):
    raw = _fake_result(acceptance=.7, count=16)
    trace = dict(raw.trace)
    if case == "missing":
        trace.pop("log_accept_ratio")
    else:
        values = {"empty": [], "nan": [float("nan")] * 16, "inf": [float("inf")] * 16,
                  "minus_inf": [-float("inf")] * 16, "shape": [[-.3]] * 16,
                  "burnin_included": [-.3] * 20}[case]
        trace["log_accept_ratio"] = tf.constant(values, tf.float64)
    raw = replace(raw, trace=trace)
    result = bootstrap.run_hmc_bootstrap_screen(adapter=_ToyGaussianAdapter(), geometry=_geometry(),
        config=_config(screen_num_results=16), run_full_chain=lambda *args: raw)
    assert not result.passed
    assert len(result.rounds) == 1
    assert "screen_acceptance_missing_or_nonfinite" in result.rounds[0].hard_vetoes
    assert result.rounds[0].diagnostics["mean_acceptance_probability"] is None


class _DeclaredDomainGaussian(_ToyGaussianAdapter):
    def classify_target_exception(self, error):
        return isinstance(error, tf.errors.InvalidArgumentError) and "fixture proposal domain" in str(error)

    def log_prob_and_grad(self, theta):
        with tf.control_dependencies([tf.debugging.assert_less(
                tf.reduce_max(tf.abs(theta)), tf.constant(100., tf.float64),
                message="fixture proposal domain")]):
            return super().log_prob_and_grad(tf.identity(theta))


def _proposal_error(**changes):
    error = tf.errors.InvalidArgumentError(None, None, "fixture proposal domain")
    error.failure_record = {
        "schema": "bayesfilter.traced_hmc_first_failure.v1",
        "evaluation_phase": "trajectory", "failure_location": "target_callback",
        "failed_leapfrog_substep": 3, "pre_transition_state": [0., 0.],
        "target_state": [float("inf"), 0.], "original_error": "fixture proposal domain",
        **changes,
    }
    return error


def test_domain_failure_brackets_smaller_trials_and_preserves_original():
    adapter = _DeclaredDomainGaussian()
    geometry = _geometry(adapter=adapter)
    error = _proposal_error()
    calls = []

    def run(adapter, initial, cfg):
        calls.append(cfg.step_size)
        if len(calls) == 2:
            raise error
        return _fake_result(acceptance=.9 if len(calls) < 4 else .7)

    result = bootstrap.run_hmc_bootstrap_screen(adapter=adapter, geometry=geometry,
        config=_config(), run_full_chain=run)
    assert result.passed and _bootstrap_preflight_passed(result)
    assert calls[1] == 2 * calls[0]
    assert calls[0] < calls[2] < calls[3] < calls[1]
    failed = result.rounds[1]
    assert failed.diagnostics["first_failure"] == error.failure_record
    assert failed.diagnostics["mean_acceptance_probability"] is None
    assert "screen_hmc_error" in failed.hard_vetoes
    assert _active_bootstrap_handoff_kernel_payload(geometry=geometry, bootstrap=result)["step_size"] == calls[-1]


def test_domain_retry_exhaustion_never_hands_off_geometry_or_failed_step():
    adapter = _DeclaredDomainGaussian()
    geometry = _geometry(adapter=adapter)
    calls = []

    def run(adapter, initial, cfg):
        calls.append(cfg.step_size)
        raise _proposal_error()

    result = bootstrap.run_hmc_bootstrap_screen(adapter=adapter, geometry=geometry,
        config=_config(max_repairs=2), run_full_chain=run)
    assert calls == [geometry.initial_step_size / 2**i for i in range(3)]
    assert result.final_status == "repair_budget_exhausted"
    assert not _bootstrap_preflight_passed(result)
    assert result.selected_kernel_payload is None
    with pytest.raises(ValueError, match="hard veto"):
        _active_bootstrap_handoff_kernel_payload(geometry=geometry, bootstrap=result)


def test_unrepresentable_smaller_step_preserves_first_failure(monkeypatch):
    adapter = _DeclaredDomainGaussian()
    error = _proposal_error()

    def run(*args):
        raise error

    def unavailable(*args, **kwargs):
        raise ValueError("no representable smaller trial")

    monkeypatch.setattr(bootstrap, "_bootstrap_smaller_trial_step", unavailable)
    result = bootstrap.run_hmc_bootstrap_screen(adapter=adapter, geometry=_geometry(adapter=adapter),
        config=_config(), run_full_chain=run)
    assert result.final_status == "repair_step_unavailable"
    assert result.selected_kernel_payload is None
    assert result.rounds[0].diagnostics["first_failure"] == error.failure_record
    assert result.rounds[0].diagnostics["repair_failure"] == "no representable smaller trial"
    assert not _bootstrap_preflight_passed(result)


@pytest.mark.parametrize("case", ["undeclared", "runtime", "device", "classifier_error", "classifier_nonbool",
                                  "initial", "trace", "missing", "bad_prestate"])
def test_unrelated_or_unattributed_errors_remain_terminal(case):
    adapter = _DeclaredDomainGaussian()
    error = _proposal_error()
    if case == "undeclared":
        error = tf.errors.InvalidArgumentError(None, None, "incompatible matrix shapes")
    elif case == "runtime":
        error = RuntimeError("programming error")
    elif case == "device":
        error = tf.errors.ResourceExhaustedError(None, None, "device exhausted")
    elif case == "classifier_error":
        def broken(error):
            raise ValueError("classifier failed")
        adapter.classify_target_exception = broken
    elif case == "classifier_nonbool":
        adapter.classify_target_exception = lambda error: "yes"
    elif case == "initial":
        error = _proposal_error(evaluation_phase="bootstrap", failed_leapfrog_substep=0)
    elif case == "trace":
        error = _proposal_error(evaluation_phase="trace", failure_location="trace")
    elif case == "missing":
        del error.failure_record
    elif case == "bad_prestate":
        error = _proposal_error(pre_transition_state=[None, 0.])
    calls = []
    def run(*args):
        calls.append(1)
        raise error
    result = bootstrap.run_hmc_bootstrap_screen(adapter=adapter, geometry=_geometry(adapter=adapter),
        config=_config(), run_full_chain=run)
    assert result.final_status == "hard_veto"
    assert len(calls) == 1
    assert not _bootstrap_preflight_passed(result)
    assert result.rounds[0].diagnostics["error_type"] == type(error).__name__


def test_real_traced_domain_failure_builds_fresh_runner_before_retry():
    adapter = _DeclaredDomainGaussian()
    geometry = replace(_geometry(adapter=adapter), initial_step_size=16., target_trajectory_length=1.5)
    result = bootstrap.run_hmc_bootstrap_screen(adapter=adapter, geometry=geometry,
        config=_config(chain_execution_mode="tf_function", acceptance_role="warmup_startup_only",
                       max_repairs=5, screen_num_results=16))
    assert result.passed and _bootstrap_preflight_passed(result)
    first = result.rounds[0].diagnostics
    assert first["proposal_domain_retry_eligible"] is True
    assert first["first_failure"]["failure_location"] == "target_callback"
    assert first["first_failure"]["diagnostic_target_replayed"] is False
    assert result.bootstrap_runner_route["reusable_runner_build_count"] > 1
    assert result.selected_kernel_payload["step_size"] < 16.
    assert result.selected_round.diagnostics["mean_acceptance_probability"] >= .55


def test_probability_selected_bootstrap_reaches_public_mass_preparation(monkeypatch):
    from bayesfilter.inference import HMCKernelTuningConfig
    from bayesfilter.inference.hmc_preparation import prepare_operational_windowed_mass_handoff
    from bayesfilter.inference.hmc_kernel_tuning import _bootstrap_public_summary
    calls = []

    def factory(adapter, initial, cfg):
        class Runner:
            def run(self, **kwargs):
                calls.append(kwargs)
                return _fake_result(acceptance=.6846376853336343,
                                    binary_rate=.8125, count=cfg.num_results)
        return Runner()

    def reached(**kwargs):
        selected = kwargs["bootstrap"]
        assert selected.passed and _bootstrap_preflight_passed(selected)
        assert len(calls) == 1
        assert _bootstrap_public_summary(selected, xla_requested=False)["last_acceptance_relation_to_band"] == "inside"
        raise RuntimeError("verified probability handoff reached mass stage")

    monkeypatch.setattr(bootstrap, "build_reusable_full_chain_tfp_hmc_runner", factory)
    monkeypatch.setattr("bayesfilter.inference.hmc_mass_adaptation.run_hmc_windowed_mass_stage", reached)
    with pytest.raises(RuntimeError, match="verified probability handoff reached mass stage"):
        prepare_operational_windowed_mass_handoff(adapter=_ToyGaussianAdapter(), initial_position=[0., 0.],
            config=HMCKernelTuningConfig.smoke(use_xla=False, target_scope="kernel_bootstrap_toy_gaussian"))
