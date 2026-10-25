"""CPU diagnostic tests for model-coordinate stopping and device continuation."""
from types import SimpleNamespace
import dataclasses
import numpy as np
import pytest
import tensorflow as tf
from bayesfilter.inference import hmc_tensorflow_verified as verified
from bayesfilter.inference import hmc_tensorflow_resume as resume
from bayesfilter.inference import neutra_hmc
from bayesfilter.inference.neural_force_hmc import (
    FrozenPositionOnlyForce, FrozenTargetPotential, bind_neural_force_hmc_tuning_runner)


def binding():
    return bind_neural_force_hmc_tuning_runner(
        force=FrozenPositionOnlyForce(lambda x:x, 'test-force'),
        target=FrozenTargetPotential(lambda x:tf.reduce_sum(x*x, -1), 'test-target'),
        target_scope='mapped-test',
        model_parameter_transform=lambda q:tf.reduce_sum(q, -1, keepdims=True),
        model_parameter_identity='sum_of_two_parameters_v1')


def counterexample():
    rng=np.random.default_rng(39)
    x=rng.normal(size=(4000,4))
    e=rng.normal(size=(4000,4))
    correlation=np.array([.98,.98,-.98,-.98])[None,:]
    return tf.constant(np.stack([x,correlation*x+np.sqrt(1-correlation**2)*e],-1),tf.float64)


def affine():
    return SimpleNamespace(latent_to_position=lambda x:x,position_to_latent=lambda x:x)


def test_mapped_quantity_detects_correlation_failure_hidden_by_marginal_rhat():
    states=counterexample()
    assert float(tf.reduce_max(verified._diagnostic_rhat(states)['maximum'])) < 1.01
    mapped=verified._diagnostic_samples(states,affine(),binding())
    assert mapped.shape==(4000,4,3)
    np.testing.assert_array_equal(mapped[:,:,:2],states)
    assert float(verified._diagnostic_rhat(mapped)['maximum'][-1]) > 1.1
    assert bool(tf.reduce_all(tf.math.is_inf(verified._diagnostic_rhat(
        tf.fill([4,4,3],tf.constant(float('nan'),tf.float64)))['maximum'])))


def test_binding_identity_records_map_and_rejects_partial_configuration():
    b=binding()
    assert 'sum_of_two_parameters_v1' in b.payload()['diagnostic_identity']
    assert dataclasses.replace(b,diagnostic_identity='different').binding_hash != b.binding_hash
    plain=dataclasses.replace(b,diagnostic_transform=None,diagnostic_identity=None)
    assert 'diagnostic_identity' not in plain.payload()
    with pytest.raises(ValueError): dataclasses.replace(b,diagnostic_identity=None)


@pytest.mark.parametrize('route',['fresh','resume'])
def test_consumers_use_same_model_diagnostics(monkeypatch,tmp_path,route):
    monkeypatch.setattr(verified,'_affine',lambda _:affine())
    cfg=SimpleNamespace(initial_chain_positions=((0.,0.),)*4)
    def controller(**kwargs):
        mapped=kwargs['transform'](counterexample())
        assert mapped.shape[-1]==3
        assert float(kwargs['diagnose'](mapped)['maximum'][-1]) > 1.1
        return {'chunks':[]}
    if route=='fresh':
        monkeypatch.setattr(neutra_hmc,'run_sequential_neutra_hmc',controller)
        verified._fresh_sequence(cfg,None,binding(),tmp_path)
    else:
        monkeypatch.setattr(resume,'run_sequential_neutra_hmc',controller)
        resume.replay_then_continue(config=cfg,nomination=None,binding=binding(),
                                   source_root=tmp_path,records=[])


def test_resume_propagates_configured_numerical_device(monkeypatch,tmp_path):
    monkeypatch.setattr(verified,'_affine',lambda _:affine())
    monkeypatch.setattr(verified,'_chunk_graph',lambda *args:'graph')
    called=[]
    def invoke(graph,state,seed,*,numerical_device):
        called.append((graph,numerical_device))
        return SimpleNamespace(final_state=state)
    monkeypatch.setattr(resume.stage,'_call_numerical_graph',invoke)
    def controller(**kwargs):
        kwargs['run_chunk'](tf.zeros([4,2],tf.float64),tf.constant([1,2]),250)
        return {'chunks':[]}
    monkeypatch.setattr(resume,'run_sequential_neutra_hmc',controller)
    resume.replay_then_continue(config=SimpleNamespace(initial_chain_positions=((0.,0.),)*4),
        nomination=SimpleNamespace(config=SimpleNamespace(numerical_device='/GPU:0')),
        binding=binding(),source_root=tmp_path,records=[],output_root=tmp_path)
    assert called==[('graph','/GPU:0')]


def test_terminal_check_uses_model_parameters(monkeypatch):
    monkeypatch.setattr(verified,'_affine',lambda _:affine())
    monkeypatch.setattr(verified,'_health',lambda *args:{'finite':True})
    cfg=SimpleNamespace(initial_chain_positions=((0.,0.),)*4,energy_error_max=None)
    checks,rhat=verified._fresh_checks(cfg,SimpleNamespace(heuristic_screen_passed=True),
        SimpleNamespace(final_state=tf.zeros([4,2],tf.float64)),
        SimpleNamespace(states=counterexample()),binding())
    assert not bool(checks['fresh_rhat'])
    assert len(rhat['maximum'])==3
