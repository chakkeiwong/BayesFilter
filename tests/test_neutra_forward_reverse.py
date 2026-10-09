"""Tiny CPU reference tests; no scientific fit or accelerator-readiness claim."""
import json
import os
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH','true')
import pytest
import tensorflow as tf

from bayesfilter.testing.neutra_generic_targets import ExactTargetEvaluator
from bayesfilter.testing.neutra_scientific_campaign import flow_config,save_tensor,write
from bayesfilter.testing.neutra_forward_reverse import load_teacher,run_fit,trainer_for,train_block,prepare_teacher

SPEC={'kind':'gaussian','mean':[0.,0.],'covariance':[[1.,0.],[0.,1.]]}
PROFILE={'width':4,'batch':4,'forward_updates':[1,1],'forward_rates':[.001,.0003],
         'rkl_rates':[.0001,.0003],'rkl_rungs':[1,2],'gradient_clip':1000.,'jit_compile':False}


def fixture_teacher(path):
    # Gaussian samples emulate a perfect approximate teacher only for mechanics.
    # This synthetic fixture is never a campaign scientific artifact.
    path.mkdir();e=ExactTargetEvaluator(SPEC,jit_compile=False);hashes={}
    for label,salt in (('training',1),('validation',10)):
        for i in range(2):
            for suffix,data in (('points',e.sample(16,tf.constant([131,salt+i]))),
                                ('logweights',tf.zeros([16],tf.float64))):
                name=f'{label}-{i}-{suffix}.tensor';hashes[name]=save_tensor(path/name,data)
    name='map-reference.tensor';hashes[name]=save_tensor(path/name,e.sample(64,tf.constant([131,20])))
    write(path/'teacher.json',{'status':'teacher_passed','target_signature':e.target.signature,
        'profile':{'replications':2},'teacher_data_kind':'approximate_weighted_native_populations',
        'role':'synthetic_unit_test_fixture_no_quality_claim','artifact_sha256':hashes})
    return e


def test_study_default_and_explicit_iaf_control_are_distinct():
    assert flow_config(2,11).kind=='naf_dsf'
    assert flow_config(2,11).naf_conditioner=='author_cmade'
    assert flow_config(2,11,kind='iaf').mask_policy=='hoffman_block_masks_v1'


def test_native_cpu_populations_trace_once_and_preserve_independent_streams(tmp_path):
    profile=dict(particles=32,stages=2,mutation_steps=1,step_size=.05,
        mode_starts=4,mode_iterations=40,replications=2,reference_rows=128,jit_compile=False)
    result=prepare_teacher(SPEC,profile,701,tmp_path)
    assert result['native_graph_traced_before_cpu_threads']
    assert len(result['population_seeds'])==4
    assert len({tuple(k) for k in result['population_seeds']})==4
    assert result['screens']['validation']['replications']==2
    assert result['teacher_data_kind']=='approximate_weighted_native_populations'


def test_native_banks_are_separate_and_changed_target_or_bytes_rejected(tmp_path):
    path=tmp_path/'teacher';e=fixture_teacher(path)
    banks,reference,_=load_teacher(path,e.target.signature)
    assert tuple(banks['training'][0].shape)==(32,2) and tuple(reference.shape)==(64,2)
    assert float(tf.reduce_sum(tf.exp(banks['training'][1])))==pytest.approx(1.)
    assert not bool(tf.reduce_all(banks['training'][0]==banks['validation'][0]))
    with pytest.raises(ValueError,match='target'):load_teacher(path,'wrong')
    (path/'training-0-points.tensor').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):load_teacher(path,e.target.signature)


def test_weighted_and_reverse_objectives_and_chunk_rng_are_wired():
    e=ExactTargetEvaluator(SPEC,jit_compile=False)
    a=trainer_for(e.target,113,PROFILE,.001);b=trainer_for(e.target,113,PROFILE,.001)
    assert a.forward_weight==1 and a.reverse_weight==0
    reverse=trainer_for(e.target,113,PROFILE,.001,reverse=True)
    assert reverse.forward_weight==0 and reverse.reverse_weight==1
    bank=(e.sample(16,tf.constant([531,1])),tf.math.log(tf.range(1,17,dtype=tf.float64)))
    train_block(a,bank,3,[41,1]);train_block(b,bank,1,[41,1]);train_block(b,bank,2,[41,1],1)
    for x,y in zip(a.variables,b.variables):tf.debugging.assert_equal(x,y)


def test_actual_forward_reverse_call_chain_freezes_all_endpoints_and_probes(tmp_path):
    fixture_teacher(tmp_path/'teacher');output=tmp_path/'fit';output.mkdir()
    result=run_fit(SPEC,{**PROFILE,'teacher_output':str(tmp_path/'teacher')},191,output)
    assert result['status']=='fit_complete' and len(result['branches'])==4
    assert result['forward']['warm_start']['feature_z_role']=='explanatory_only_for_forward_warm_start'
    assert all(b['criterion']==result['criterion'] for b in result['branches'])
    assert all(b['parent_hash']==result['forward']['transport_hash'] for b in result['branches'])
    assert len(list(output.glob('*-checkpoint.json')))==5
    for endpoint in [result['forward'],*[b['endpoint'] for b in result['branches']]]:
        probe=json.loads(Path(endpoint['probe_path']).read_text())
        assert probe['complete'] and probe['finite'] and probe['valid_rows']==1000
