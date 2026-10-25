"""CPU-only wiring and campaign restart tests."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf
from bayesfilter.testing.structural_ledh_models_tf import StructuralModelSpec,make_structural_data,make_structural_design
from bayesfilter.testing.structural_ledh_target import StructuralTarget,ready


@pytest.mark.parametrize('name',['model_a','model_b'])
def test_native_target_prior_and_analytical_score(name):
    spec=StructuralModelSpec(name)
    dataset=ready(make_structural_data(spec,2,61,tf.float64));dataset['model']=name
    design=ready(make_structural_design(16,2,62,tf.float64))
    target=StructuralTarget(dataset,design,dict(flow_steps=8,reset_steps=5,correction_steps=1),
        batch_size=4,jit_compile=False,dtype=tf.float64)
    q=tf.stack([spec.truth(tf.float64)+i*.01 for i in range(4)])
    v,s,valid=target.value_score(q)
    assert bool(tf.reduce_all(valid))
    ll,ls,_=target.likelihood(q);prior,ps=target.log_prior(q)
    np.testing.assert_allclose(v,ll+prior,atol=1e-10)
    np.testing.assert_allclose(s,ls+ps,atol=1e-10)
    for j in range(spec.parameter_count):
        d=tf.one_hot(j,spec.parameter_count,dtype=tf.float64)[None]*2e-5
        fd=(target.value_score(q+d)[0]-target.value_score(q-d)[0])/4e-5
        np.testing.assert_allclose(s[:,j],fd,atol=4e-6,rtol=4e-5)
    mean,scale=spec.prior_parameters(tf.float64)
    expected=-.5*tf.reduce_sum(((q-mean)/scale)**2+np.log(2*np.pi)+2*tf.math.log(scale),-1)
    np.testing.assert_allclose(prior,expected,atol=1e-12)


def master_module():
    path=Path(__file__).resolve().parents[2]/'scripts/run_structural_ledh_ab.py'
    spec=importlib.util.spec_from_file_location('structural_master',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_budget_persists_and_resume_does_not_replenish(tmp_path):
    module=master_module();root=tmp_path/'campaign'
    c=module.Campaign(root,48)
    c.state['attempts'].append(dict(key='failed',gpu=True,charged_seconds=123))
    c.persist()
    resumed=module.Campaign(root)
    assert resumed.remaining(True)==48*3600-123
    assert resumed.remaining(False)==7200
    with pytest.raises(ValueError): module.Campaign(tmp_path/'too_big',49)


def test_hmc_delegates_to_public_sequential_controller():
    import ast,inspect
    module=master_module()
    tree=ast.parse(inspect.getsource(module.hmc))
    calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    assert 'tune_hmc_kernel' in calls
    assert 'VerifiedTensorFlowHMCConfig' in calls
    assert 'sample_chain' not in calls


def test_reviewed_migration_preserves_budget_and_invalidates_stale_runs(tmp_path,monkeypatch):
    module=master_module();root=tmp_path/'campaign'
    monkeypatch.setattr(module,'source_identity',lambda:{'code':'old'})
    c=module.Campaign(root,48)
    c.state['attempts']=[dict(key='pilot',kind='probe',gpu=True,charged_seconds=123,finished='done'),
        dict(key='prepare',kind='prepare',gpu=False,charged_seconds=12,finished='done')]
    c.state['selection']={'old':'stale.json'};c.persist()
    monkeypatch.setattr(module,'source_identity',lambda:{'code':'new'})
    with pytest.raises(ValueError): module.Campaign(root)
    note=tmp_path/'review.md';note.write_text('Reviewed local algebraic repair')
    resumed=module.Campaign(root,migration_note=note)
    assert resumed.remaining(True)==48*3600-123
    assert resumed.state['selection']=={}
    assert resumed.state['attempts'][0]['superseded_by_source_migration']
    assert not resumed.state['attempts'][1].get('superseded_by_source_migration')


def test_crashed_attempt_charged_conservatively(tmp_path):
    module=master_module();c=module.Campaign(tmp_path/'campaign',48)
    c.state['attempts']=[dict(key='pilot',kind='probe',gpu=True,charged_seconds=5,
        output=str(tmp_path/'missing'),started=module.now(),allowance_seconds=100)]
    c.recover()
    assert c.remaining(True)==48*3600-100
    assert c.state['attempts'][0]['exit_code']==-1


def test_scoped_reset_migration_preserves_a_budget_and_old_evidence(tmp_path,monkeypatch):
    module=master_module();root=tmp_path/'campaign'
    source='bayesfilter/highdim/covariance_proposal_score_batch_tf.py'
    monkeypatch.setattr(module,'source_identity',lambda:{source:'old'})
    c=module.Campaign(root,48)
    oldsel=root/'selections/b.json';module.save(oldsel,dict(controls={'flow_steps':8}))
    frozen=root/'old-training/frozen-map.json';module.save(frozen,dict(old=True))
    module.save(frozen.with_name('checkpoint.json'),dict(job=dict(width=18,rate=.001,seed=17)))
    a=dict(result=dict(posterior_sampling_complete=True,manifest='original-a'),output='old-a')
    c.state.update(selection={'model_b-T50':str(oldsel)},frozen_map=str(frozen),
        hmc={'model_a':a,'model_b':{'result':{'status':'failed'}}})
    c.state['attempts']=[dict(key='old',kind='hmc',gpu=True,charged_seconds=321,finished='done')]
    c.persist();oldbytes=oldsel.read_bytes();notepath=tmp_path/'note.md';notepath.write_text('Reviewed isolated reset repair')
    monkeypatch.setattr(module,'source_identity',lambda:{source:'new'})
    repaired=module.Campaign(root,migration_note=notepath,repair_model_b_reset=True)
    assert repaired.state['hmc']=={'model_a':a}
    assert repaired.remaining(True)==48*3600-321
    assert oldsel.read_bytes()==oldbytes
    assert not repaired.state['attempts'][0].get('superseded_by_source_migration')
    assert repaired.state['reset_repair']['prior_frozen_map']==str(frozen)
    assert module.read(repaired.state['selection']['model_b-T50'])['reset_precision_policy']=='conditional_fp64'
    # Ordinary resume keeps the scoped recovery path, without repeating migration.
    resumed=module.Campaign(root)
    assert resumed.state['reset_repair']==repaired.state['reset_repair']


def test_reset_migration_rejects_unrelated_numerical_changes(tmp_path,monkeypatch):
    module=master_module();root=tmp_path/'campaign'
    monkeypatch.setattr(module,'source_identity',lambda:{'unrelated.py':'old'})
    module.Campaign(root,48);before=(root/'state.json').read_bytes()
    monkeypatch.setattr(module,'source_identity',lambda:{'unrelated.py':'new'})
    note=tmp_path/'note.md';note.write_text('Not enough to permit unrelated changes')
    with pytest.raises(ValueError,match='unrelated'):
        module.Campaign(root,migration_note=note,repair_model_b_reset=True)
    assert (root/'state.json').read_bytes()==before
