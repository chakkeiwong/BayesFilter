"""Real TF/TFP preparation and adversarial discarded-window recovery checks.

CPU reference tests; small schedules test mechanisms, not posterior adequacy.
"""
import base64
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference import hmc_warmup as warmup
from bayesfilter.inference.hmc_configuration import HMCKernelTuningConfig, _public_loop_config, _phase7_windowed_stage_config
from bayesfilter.inference.hmc_mass_adaptation import _windowed_mass_stage_internal_config
from bayesfilter.inference.hmc_coordinates import WarmupTrajectoryPolicy
from bayesfilter.inference.hmc_tuning import WindowedMassAdaptationConfig
from bayesfilter.inference.hmc_preparation_recovery import RejectedPreparationProposal
from bayesfilter.inference.hmc_tuning_artifacts import _validate_operational_warmup_payload
from bayesfilter.testing.inference_validation.targets import ValidationTarget
from tests.test_hmc_warmup import _transform


def _kwargs(target="gaussian", *, restarts=2, mode="eager", late=False):
    adapter = ValidationTarget(target, jit_compile=False)
    dim = adapter.parameter_dim
    return dict(adapter=adapter, initial_transform=_transform(np.eye(dim)),
        initial_canonical_theta=tf.fill([dim], tf.constant(.2, tf.float64)),
        initial_step_size=.2, initial_step_size_upper_bound=.2,
        initial_step_qualification_source="fixture_finite_ceiling",
        trajectory_policy=WarmupTrajectoryPolicy(3, 25),
        config=WindowedMassAdaptationConfig(warmup_steps=112 if late else 64,
            initial_buffer=16, first_window_size=64 if late else 32, final_buffer=32 if late else 16,
            min_window_samples=16, metric_evidence_policy="finite_window", preparation_max_restarts=restarts),
        target_accept_prob=.7, seed=(2026092080, 91), target_scope="inference_validation",
        chain_execution_mode=mode, target_status_trace_policy="none", jit_compile=False)


def _inject(monkeypatch, *, corruption="proposal", call_index=1, every=False):
    original = tfp.mcmc.sample_chain
    calls = []
    def sample(**kwargs):
        result = original(**kwargs)
        if (not hasattr(result, "all_states") or not isinstance(result.trace, dict)
                or "recovery_proposed_state" not in result.trace):
            return result
        calls.append(kwargs)
        if len(calls) != call_index and not every:
            return result
        states, trace = result.all_states, dict(result.trace)
        # Change only the last transition. No later draw depends on the
        # deliberately invalid proposed endpoint; retained state stays put.
        states = tf.concat([states[:-1], states[-2:-1]], axis=0)
        def last(value, replacement):
            return tf.concat([value[:-1], tf.cast(replacement, value.dtype)[None]], axis=0)
        trace["is_accepted"] = last(trace["is_accepted"], False)
        trace["log_accept_ratio"] = last(trace["log_accept_ratio"], -np.inf)
        trace["target_log_prob"] = last(trace["target_log_prob"], trace["target_log_prob"][-2])
        trace["recovery_proposed_state"] = last(trace["recovery_proposed_state"], tf.fill(tf.shape(states[-1]), tf.constant(np.inf, tf.float64)))
        trace["recovery_proposal_finite"] = last(trace["recovery_proposal_finite"], False)
        if corruption == "accepted":
            trace["is_accepted"] = last(trace["is_accepted"], True)
        elif corruption == "retained":
            states = last(states, tf.fill(tf.shape(states[-1]), tf.constant(np.nan, tf.float64)))
        elif corruption == "score":
            trace["recovery_retained_score_finite"] = last(trace["recovery_retained_score_finite"], False)
        elif corruption == "step":
            trace["consumed_step_size"] = last(trace["consumed_step_size"], 0.)
        elif corruption == "unclassified":
            trace["log_accept_ratio"] = last(trace["log_accept_ratio"], np.nan)
        elif corruption == "telemetry":
            trace["recovery_proposal_finite"] = tf.cast(trace["recovery_proposal_finite"], tf.int32)
        elif corruption == "execution":
            raise RuntimeError("unclassified backend failure")
        return result._replace(all_states=states, trace=trace)
    monkeypatch.setattr(tfp.mcmc, "sample_chain", sample)
    return calls


def test_public_recovery_config_reaches_actual_warmup():
    public=HMCKernelTuningConfig(preparation_max_restarts=3)
    loop=_public_loop_config(public)
    stage=_phase7_windowed_stage_config(loop, attempt_index=0)
    internal=_windowed_mass_stage_internal_config(preparation_max_restarts=stage.preparation_max_restarts)
    for config in (public, loop, stage, internal):
        assert config.preparation_max_restarts == config.payload()["preparation_max_restarts"] == 3


@pytest.mark.parametrize("value", [-1, True, .5, "3"])
def test_recovery_rejects_invalid_caps(value):
    with pytest.raises(ValueError, match="preparation_max_restarts"):
        HMCKernelTuningConfig(preparation_max_restarts=value)


def test_recovery_rejects_fixed_identity():
    with pytest.raises(ValueError, match="windowed"):
        HMCKernelTuningConfig(mass_policy="fixed_identity", preparation_max_restarts=1)


def test_default_calls_original_attempt_unchanged():
    kwargs=_kwargs(restarts=0, mode="tf_function")
    original=warmup._run_operational_windowed_warmup_attempt(**kwargs)
    actual=warmup.run_operational_windowed_warmup(**kwargs)
    assert actual.preparation_recovery is None
    for a,b in zip(original.windows, actual.windows):
        np.testing.assert_array_equal(a.adaptation_latent_states,b.adaptation_latent_states)
        np.testing.assert_array_equal(a.log_accept_ratio,b.log_accept_ratio)
    assert original.final_kernel_state.transform.signature == actual.final_kernel_state.transform.signature
    assert original.final_kernel_state.epsilon == actual.final_kernel_state.epsilon


@pytest.mark.parametrize("target", ["gaussian", "rotated_gaussian", "beta_binomial", "lgssm_location", "banana", "funnel_noncentered"])
def test_real_multimodel_recovery_preserves_failed_draws_and_resets(monkeypatch,target):
    _inject(monkeypatch)
    events=[]
    result=warmup.run_operational_windowed_warmup(**_kwargs(target), recovery_callback=lambda e,p: events.append((e,p)))
    summary=result.preparation_recovery
    assert summary["restart_count"] == 1
    assert summary["discarded_transition_count"] == 16
    assert summary["successful_transition_count"] == 64
    assert len(result.discarded_attempts) == 1
    assert sum(w.window.length for w in result.windows) == 64
    repair=summary["repairs"][0]
    assert repair["probe"]["selected_step_size"] <= repair["step_ceiling"] == repair["failed_step"] / 2
    assert all(a["step_size"] <= repair["step_ceiling"] for a in repair["probe"]["attempts"])
    assert result.seed_root != _kwargs(target)["seed"]
    assert result.windows[0].dual_averaging_generation == 0
    archived=next(p for e,p in events if e=="attempt_discarded")
    saved=tf.io.parse_tensor(base64.b64decode(archived["tensors"]["trace_log_accept_ratio"]["tensor_base64"]),out_type=tf.float64)
    assert bool(tf.math.is_inf(saved[-1]))
    assert all(np.isfinite(w.log_accept_ratio).all() for w in result.windows)
    _validate_operational_warmup_payload(json.loads(json.dumps(result.public_payload())))


@pytest.mark.parametrize("corruption", ["accepted","retained","score","step","unclassified","telemetry","execution"])
def test_shared_retained_and_unclassified_failures_do_not_restart(monkeypatch,corruption):
    _inject(monkeypatch,corruption=corruption)
    events=[]
    with pytest.raises((ValueError,RuntimeError)) as error:
        warmup.run_operational_windowed_warmup(**_kwargs(),recovery_callback=lambda e,p:events.append(e))
    assert not isinstance(error.value,RejectedPreparationProposal)
    assert events.count("attempt_start") == 1
    assert "restart_probe" not in events


def test_exhaustion_preserves_every_failed_attempt_and_work(monkeypatch):
    _inject(monkeypatch,every=True)
    with pytest.raises(RejectedPreparationProposal) as caught:
        warmup.run_operational_windowed_warmup(**_kwargs(restarts=1))
    assert len(caught.value.discarded_attempts)==2
    summary=caught.value.recovery_summary
    assert summary["status"]=="restart_cap_exhausted"
    assert summary["discarded_transition_count"]==32


def test_recovery_after_metric_change_keeps_only_valid_checkpoint(monkeypatch):
    _inject(monkeypatch,call_index=3)
    result=warmup.run_operational_windowed_warmup(**_kwargs(late=True))
    failed=result.discarded_attempts[0]
    assert failed.completed_windows[1].metric_decision.update_applied
    assert failed.checkpoint.transform.signature==failed.completed_windows[1].next_coordinate_signature
    assert result.initial_coordinate_signature==failed.checkpoint.transform.signature
    np.testing.assert_array_equal(failed.checkpoint.canonical_theta,failed.completed_windows[1].final_canonical_theta)
    _validate_operational_warmup_payload(json.loads(json.dumps(result.public_payload())))


def test_contracted_probe_accepts_high_acceptance_without_expanding():
    kwargs=_kwargs()
    result=warmup.find_reasonable_epsilon(adapter=kwargs["adapter"], current_state=kwargs["initial_canonical_theta"],
        initial_step_size=.01, seed=(191,291), num_leapfrog_steps=3, momentum_probe_count=4,
        probe_num_results=2, preparation_step_ceiling=.01)
    assert result.passed and len(result.attempts)==1 and result.selected_step_size==.01
    assert result.attempts[-1].mean_acceptance_probability > .75


def test_timeout_after_failure_preserves_discarded_attempt(monkeypatch):
    _inject(monkeypatch)
    attempt=[]
    def event(name,payload):
        if name=="attempt_start":
            attempt.append(payload["attempt_index"])
    def stop(boundary,windows):
        if attempt==[0,1] and boundary=="before_first_window":
            return {"stop_source":"test_budget","stop_reason":"time_cap"}
    result=warmup.run_operational_windowed_warmup(**_kwargs(),recovery_callback=event,boundary_callback=stop)
    assert isinstance(result,warmup.OperationalWindowedWarmupCloseout)
    assert result.boundary_payload["preparation_recovery"]["discarded_transition_count"]==16
    assert len(result.discarded_attempts)==1


def test_graph_recovery_executes_with_stable_runner_signature(monkeypatch):
    # Inject at the host classification boundary after the real graph executes.
    from bayesfilter.inference import hmc_preparation_recovery as recovery
    original=recovery.rejected_proposal_failure
    seen=[]
    def reject_once(**kwargs):
        seen.append(kwargs)
        if len(seen)!=1:
            return original(**kwargs)
        trace=dict(kwargs["trace"])
        states=kwargs["latent_draws"]
        states=tf.concat([states[:-1],states[-2:-1]],axis=0)
        trace["is_accepted"]=tf.concat([trace["is_accepted"][:-1],[False]],axis=0)
        trace["log_accept_ratio"]=tf.concat([trace["log_accept_ratio"][:-1],tf.constant([-np.inf],tf.float64)],axis=0)
        trace["recovery_proposal_finite"]=tf.concat([trace["recovery_proposal_finite"][:-1],[False]],axis=0)
        return original(**dict(kwargs,trace=trace,latent_draws=states))
    monkeypatch.setattr(recovery,"rejected_proposal_failure",reject_once)
    result=warmup.run_operational_windowed_warmup(**_kwargs(mode="tf_function"))
    assert result.preparation_recovery["restart_count"]==1
    assert all(w.runner_trace_count == 1 for w in result.windows)


@pytest.mark.parametrize("field", ["restart_count","successful_seed","discarded_transition_count","transition_work_limit"])
def test_recovery_reader_rejects_corrupt_accounting(monkeypatch,field):
    _inject(monkeypatch)
    result=warmup.run_operational_windowed_warmup(**_kwargs())
    payload=json.loads(json.dumps(result.public_payload()))
    payload["preparation_recovery"][field]=[0,0] if field=="successful_seed" else -1
    with pytest.raises(ValueError,match="recovery|preparation"):
        _validate_operational_warmup_payload(payload)


@pytest.mark.parametrize("delta", [-1, 1])
def test_recovery_reader_reconstructs_probe_work(monkeypatch, delta):
    _inject(monkeypatch)
    result = warmup.run_operational_windowed_warmup(**_kwargs())
    payload = json.loads(json.dumps(result.public_payload()))
    _validate_operational_warmup_payload(payload)
    payload["preparation_recovery"]["charged_probe_transition_bound"] += delta
    with pytest.raises(ValueError, match="probe.*work|work.*probe"):
        _validate_operational_warmup_payload(payload)
