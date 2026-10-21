"""CPU-only regression for fixed nomination caps and rare energy excursions."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import tensorflow as tf
from bayesfilter.inference import hmc_tensorflow_tuning as native
from bayesfilter.testing.surrogate_hmc_m13_public_tuning import tuning_config


def config(**changes):
    original=tuning_config("energy-guard-mechanics").nomination
    return replace(original, verification_results=16, verification_repair_rounds=2,
        acceptance_policy=native.FourChainMeanBandAcceptancePolicy(
            overall_band=(.5,.85),per_chain_band=(.05,1.)), **changes)


@pytest.mark.parametrize("field", ["max_step_size","energy_error_max"])
@pytest.mark.parametrize("value", [0.,-1.,float("nan"),float("inf")])
def test_guard_configuration_rejects_invalid_values(field,value):
    with pytest.raises(ValueError):
        config(**{field:value})


def test_roundtrip_and_legacy_absent_guards():
    cfg=config(max_step_size=.5,energy_error_max=1000.)
    assert native.TensorFlowHMCKernelTuningConfig.from_payload(cfg.payload())==cfg
    legacy=config().payload()
    del legacy["max_step_size"],legacy["energy_error_max"]
    restored=native.TensorFlowHMCKernelTuningConfig.from_payload(legacy)
    assert restored.max_step_size is None and restored.energy_error_max is None


def chunk(step, *, energy=.35, extreme=None):
    delta=tf.fill([16,4],tf.constant(energy,tf.float64))
    if extreme is not None:
        delta=tf.tensor_scatter_nd_update(delta,[[0,0]],tf.constant([extreme],tf.float64))
    state=tf.ones([4,2],tf.float64)
    status=tf.concat([tf.ones([16,4,1],tf.int32),tf.zeros([16,4,10],tf.int32)],-1)
    return native._TensorChunk(tf.broadcast_to(state,[16,4,2]),-delta,
        tf.zeros([16,4],tf.bool),tf.zeros([16,4],tf.bool),delta,
        tf.ones([16,4],tf.bool),state,tf.constant(step,tf.float64),status)


def test_healthy_guard_is_noop_and_rare_energy_overrides_good_acceptance():
    cfg=config(energy_error_max=1000.)
    healthy=chunk(.5)
    decision=cfg.acceptance_policy.evaluate(healthy.log_accept_ratio,divergence_count=0)
    checked=native._guard_candidate_energy(decision,healthy.delta_h,cfg)
    for got,expected in zip(checked,decision):
        tf.debugging.assert_equal(got,expected)
    rare=chunk(.5,extreme=1201.)
    nominal=cfg.acceptance_policy.evaluate(rare.log_accept_ratio,divergence_count=0)
    assert bool(nominal.passed)
    guarded=native._guard_candidate_energy(nominal,rare.delta_h,cfg)
    assert not bool(guarded.passed) and int(guarded.repair_direction)==-1
    assert bool(guarded.log_acceptance_defined)
    nonfinite=native._guard_candidate_energy(nominal,chunk(.5,extreme=float("nan")).delta_h,cfg)
    assert not bool(nonfinite.passed) and not bool(nonfinite.log_acceptance_defined)


def test_candidate_repairs_energy_with_smaller_step_after_adaptation_cap(monkeypatch):
    """Exercise the actual candidate consumer, including the bounded repair loop."""
    calls=[]
    monkeypatch.setattr(native,"_adaptive_kernel",lambda **kw:SimpleNamespace(adaptive=True))
    monkeypatch.setattr(native,"_fixed_kernel",lambda **kw:SimpleNamespace(adaptive=False,step=float(kw["step_size"])))
    def run(**kw):
        if kw["adaptive"]:
            return chunk(2.)
        step=kw["kernel"].step
        calls.append(step)
        return chunk(step,extreme=1201. if len(calls)==1 else None)
    monkeypatch.setattr(native,"_run_tensor_steps",run)
    result,decision,trace=native._candidate_verification(binding=None,adapter=None,
        initial_state=tf.zeros([4,2],tf.float64),initial_step=tf.constant(.2,tf.float64),
        leapfrog_steps=1,config=config(max_step_size=.5,energy_error_max=1000.),
        seed=tf.constant([14,15],tf.int32))
    assert calls==[.5,.25]
    assert bool(decision.passed) and float(result.final_step_size)==.25
    assert tuple(trace.shape)==(3*16,4,14)


def test_high_acceptance_repair_cannot_escape_step_ceiling(monkeypatch):
    calls=[]
    monkeypatch.setattr(native,"_adaptive_kernel",lambda **kw:None)
    monkeypatch.setattr(native,"_fixed_kernel",lambda **kw:float(kw["step_size"]))
    def run(**kw):
        if kw["adaptive"]:
            return chunk(2.)
        calls.append(kw["kernel"])
        return chunk(kw["kernel"],energy=.001)
    monkeypatch.setattr(native,"_run_tensor_steps",run)
    result,decision,_=native._candidate_verification(binding=None,adapter=None,
        initial_state=tf.zeros([4,2],tf.float64),initial_step=tf.constant(.2,tf.float64),
        leapfrog_steps=1,config=config(max_step_size=.5,energy_error_max=1000.),
        seed=tf.constant([14,15],tf.int32))
    assert calls==[.5,.5,.5]
    assert not bool(decision.passed) and float(result.final_step_size)==.5
