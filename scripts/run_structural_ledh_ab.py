#!/usr/bin/env python3
"""Resumable structural LEDH/surrogate-HMC campaign; standard-library supervisor.

Workers use TensorFlow, canonical NeuTraTransportTrainer and tune_hmc_kernel's
shared sequential HMC. Independent NumPy references run in a CPU worker only.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
GPU='GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a'
SCOPES=[('model_a',2),('model_a',10),('model_a',50),
        ('model_b',2),('model_b',3),('model_b',10),('model_b',50)]
PARTITIONS=('calibration','validation','final')
CONTROLS=[dict(flow_steps=f,reset_steps=r,reset_epsilon=1.,correction_steps=1)
          for f,r in ((16,5),(8,5),(16,10))]
PLAN='docs/plans/structural-ledh-model-ab-master-2026-10-23.md'


def ready(x):
    if hasattr(x,'numpy'): return x.numpy().tolist()
    if isinstance(x,dict): return {str(k):ready(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [ready(v) for v in x]
    return x


def save(path,x):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(ready(x),indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def read(path): return json.loads(Path(path).read_text())
def hash_file(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def scope_key(model,horizon): return f'{model}-T{horizon}'


def source_identity():
    paths=[ROOT/'scripts/run_structural_ledh_ab.py']
    for folder in ('bayesfilter/highdim','bayesfilter/inference','bayesfilter/testing','bayesfilter/runtime'):
        paths.extend(sorted((ROOT/folder).glob('*.py')))
    return {str(p.relative_to(ROOT)):hash_file(p) for p in paths}


def configure(gpu):
    os.environ['CUDA_VISIBLE_DEVICES']=GPU if gpu else '-1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
    os.environ.setdefault('BAYESFILTER_PRELOAD_CUSTOM_OP','0')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    import tensorflow as tf
    if gpu:
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    else:
        memory=dict(mode='CPU_reference_GPU_intentionally_hidden')
    tf.config.experimental.enable_tensor_float_32_execution(False)
    return tf,ready(dict(gpu=gpu,memory_policy=memory,tf_version=tf.__version__,
                        dtype='float32',tf32=False,jit_compile=gpu,device=GPU if gpu else 'CPU'))


def prepared_paths(root,model,horizon,partition):
    key=scope_key(model,horizon)
    return root/'prepared'/f'{key}-{partition}.json',root/'prepared'/f'{key}-{partition}-design.json'


def target_for(tf,job,partition='final'):
    from bayesfilter.testing.structural_ledh_target import StructuralTarget
    data,design=prepared_paths(Path(job['root']),job['model'],job['horizon'],partition)
    if job['kind'] in ('training','hmc'):
        selection=read(job['selection'])
        if (selection['controls']!=job['controls'] or selection['model']!=job['model'] or
            selection['horizon']!=job['horizon'] or selection['data_sha256']!=hash_file(data) or
            selection['design_sha256']!=hash_file(design) or selection['source_identity']!=source_identity() or
            selection.get('reset_precision_policy','uniform')!=job.get('reset_precision_policy','uniform')):
            raise ValueError('Missing/stale scope-specific offline tuning identity')
    return StructuralTarget(read(data),read(design),job['controls'],batch_size=4,
                            reset_precision_policy=job.get('reset_precision_policy','uniform'))


def prepare(tf,job,out):
    from bayesfilter.testing.structural_ledh_models_tf import (
        StructuralModelSpec,make_structural_data,make_structural_design)
    root=Path(job['root']); receipts=[]
    for model,horizon in SCOPES:
        spec=StructuralModelSpec(model)
        for part,partition in enumerate(PARTITIONS):
            seed=26102300+(0 if model=='model_a' else 100)+part*1000+horizon
            data=ready(make_structural_data(spec,horizon,seed))
            data.update(model=model,horizon=horizon,seed=seed,partition=partition)
            data_path,design_path=prepared_paths(root,model,horizon,partition)
            save(data_path,data)
            save(design_path,make_structural_design(4032,horizon,seed+10000))
            mean,scale=spec.prior_parameters(); truth=spec.truth()
            save(data_path.with_name(data_path.stem+'-probes.json'),
                 tf.stack([truth,truth+.2*scale,truth-.2*scale,mean]))
            receipts.append(dict(data=str(data_path),data_sha256=hash_file(data_path),
                design=str(design_path),design_sha256=hash_file(design_path),seed=seed))
    return dict(status='PREPARED',receipts=receipts,particles=4032,external_generation='CPU',
                generation_note='Small stateless draws, no parallel replay generation required')


def references(tf,job,out):
    from bayesfilter.testing.structural_reference_diagnostics import (
        kalman_a,quadrature_value_score,replicated_reference,gaussian_value_score)
    root=Path(job['root']); rows=[]
    for model,horizon in SCOPES:
        for partition in PARTITIONS:
            data_path,_=prepared_paths(root,model,horizon,partition)
            data=read(data_path); probes=read(data_path.with_name(data_path.stem+'-probes.json'))
            estimates=[]; heuristics=[]
            for index,q in enumerate(probes):
                if model=='model_a':
                    h=1e-4; score=[]
                    for j in range(len(q)):
                        plus,minus=list(q),list(q); plus[j]+=h; minus[j]-=h
                        score.append((kalman_a(plus,data['observations'])-kalman_a(minus,data['observations']))/(2*h))
                    estimate=dict(mean=[kalman_a(q,data['observations']),*score],
                        mcse=[0.]*(len(q)+1),refinement=[0.]*(len(q)+1),kind='exact_Kalman')
                elif horizon<=3:
                    low,high=[quadrature_value_score(q,data['observations'],order) for order in (24,32)]
                    a,b=[low['value'],*low['score']],[high['value'],*high['score']]
                    estimate=dict(mean=b,mcse=[0.]*4,refinement=[abs(x-y) for x,y in zip(a,b)],
                                  kind='refined_path_quadrature',orders=[24,32])
                else:
                    low=replicated_reference(q,data['observations'],16384,list(range(3100,3108)))
                    high=replicated_reference(q,data['observations'],65536,list(range(4100,4108)))
                    estimate=dict(mean=high['mean'],mcse=high['mcse'],
                        refinement=[abs(x-y) for x,y in zip(high['mean'],low['mean'])],
                        kind='replicated_Fisher_bootstrap',low=low,high=high,finite_particle_bias_resolved=False)
                estimates.append(dict(q=q,**estimate))
                heuristics.append(dict(q=q,region=('truth','positive_local','negative_local','prior_center')[index],
                    ekf=gaussian_value_score(model,q,data['observations'],'ekf'),
                    cubature=gaussian_value_score(model,q,data['observations'],'cubature'),
                    bootstrap=(estimate.get('low') if horizon>3 and model=='model_b' else None)))
                save(out/'progress.json',dict(model=model,horizon=horizon,partition=partition,probe=index))
            row=dict(model=model,horizon=horizon,partition=partition,estimates=estimates,heuristics=heuristics,
                     data_sha256=hash_file(data_path))
            path=root/'references'/f'{scope_key(model,horizon)}-{partition}.json'
            save(path,row); rows.append(str(path))
    return dict(status='REFERENCES_COMPUTED',references=rows,
                uncertainty='PF Monte Carlo errors measured; particle bias requires interpretation')


def probe(tf,job,out):
    target=target_for(tf,job,job.get('partition','calibration'))
    data_path,_=prepared_paths(Path(job['root']),job['model'],job['horizon'],job.get('partition','calibration'))
    q=tf.constant(read(data_path.with_name(data_path.stem+'-probes.json')),tf.float32)
    rows=[]
    for repeat in range(2):
        start=time.monotonic(); value,score,valid=target.likelihood(q)
        row=ready(dict(value=value,score=score,valid=valid)); row['seconds']=time.monotonic()-start
        rows.append(row); save(out/'progress.json',dict(repeat=repeat,**row))
    return dict(status='PROBE_VALID' if all(rows[-1]['valid']) else 'CANDIDATE_INVALID',
                rows=rows,allocator=tf.config.experimental.get_memory_info('GPU:0'),
                target_signature=target.target_signature,batch_size=4,particles=4032,chunk=2016,
                controls=job['controls'],native_batch=True,samplewise_loop=False,scientific_promotion=False)


def reset_precision_check(tf,job,out):
    """Production call-chain check; does not certify posterior accuracy."""
    import numpy as np
    target=target_for(tf,job);root=Path(job['root'])
    prior=root/'attempts/model_b-reset-precision-01'
    reference=read(prior/'original64.json')['rows']
    original=read(prior/'original32.json')['rows']
    rows=[];passed=True
    for ref,old in zip(reference,original):
        with tf.device('/CPU:0'):
            q=tf.constant(ref['q'],tf.float32)
        result=target.program(q,target.observations,target.design['initial'],target.design['innovations'])
        value,score=result['value'].numpy(),result['score'].numpy()
        rv,rs=np.asarray(ref['value']),np.asarray(ref['score'])
        value_error=np.abs(value-rv)/np.maximum(1.,np.abs(rv))
        score_error=np.linalg.norm(score-rs,axis=1)/np.maximum(1.,np.linalg.norm(rs,axis=1))
        valid=bool(np.all(result['valid'].numpy())) and bool(np.all(result['steps_completed'].numpy()==50))
        healthy_equal=(ref['group']!='healthy' or
            (np.array_equal(value,np.asarray(old['value'])) and np.array_equal(score,np.asarray(old['score']))))
        posterior,gradient,good=target.value_score(q)
        with tf.device('/CPU:0'):
            pv,ps=target.log_prior(q)
        wiring=(np.allclose(posterior.numpy(),value+pv.numpy(),rtol=1e-6,atol=1e-5) and
            np.allclose(gradient.numpy(),score+ps.numpy(),rtol=1e-6,atol=1e-5) and bool(np.all(good.numpy())))
        passed &= valid and healthy_equal and wiring and bool(np.all(value_error<=.01)) and bool(np.all(score_error<=.01))
        rows.append(ready(dict(group=ref['group'],value=value.tolist(),score=score.tolist(),valid=valid,
            relative_value_error=value_error.tolist(),relative_score_error=score_error.tolist(),
            score_component_error=(score-rs).tolist(),healthy_bitwise_equal=healthy_equal,posterior_wiring=bool(wiring),
            fallback_count=np.sum(result['reset_precision_fallback'].numpy(),axis=1).tolist(),
            max_cov_error=np.max(result['trace'].numpy()[:,:,6],axis=1).tolist())))
    return dict(status='RESET_PRECISION_VALID' if passed else 'RESET_PRECISION_INVALID',rows=rows,
        target_signature=target.target_signature,reset_precision_policy=job['reset_precision_policy'],
        reference_files={name:hash_file(prior/name) for name in ('original32.json','original64.json')},
        scientific_default_promoted=False)


def map_checks(tf,transport):
    @tf.function(input_signature=[],jit_compile=True,autograph=False)
    def check():
        z=tf.random.stateless_normal([8,transport.parameter_dim],[261026,1],dtype=tf.float64)
        q,ld=transport.forward_and_logdet(z)
        inverse_error=tf.reduce_max(tf.abs(transport.inverse_theta_to_z_batch(q)-z))
        h=tf.constant(1e-5,tf.float64);columns=[]
        for j in range(transport.parameter_dim):
            step=h*tf.one_hot(j,transport.parameter_dim,dtype=tf.float64)
            columns.append((transport.forward_batch(z+step)-transport.forward_batch(z-step))/(2*h))
        jac=tf.stack(columns,-1)
        return inverse_error,jac,ld
    inverse_error,jac,ld=check()
    # LogMatrixDeterminant is not supported by this XLA GPU build. The
    # independent finite-difference comparison is reporting, outside the map.
    with tf.device('/CPU:0'):
        logdet_error=float(tf.reduce_max(tf.abs(tf.linalg.slogdet(jac)[1]-ld)))
        inverse_error=float(inverse_error)
    return dict(inverse_error=inverse_error,logdet_error=logdet_error,
                valid=inverse_error<1e-7 and logdet_error<1e-6)


def integration(tf,job,out):
    """XLA runtime wiring smoke only; no sampling/convergence conclusion."""
    from bayesfilter.inference.neural_force_hmc import NeuralForceHMCConfig,neural_force_hmc_transition
    from bayesfilter.inference.neutra_transport import NeuTraTransport,NeuTraTransportConfig
    target=target_for(tf,job,'calibration');payload=None; checks=None
    if target.spec.name=='model_b':
        with tf.device('/CPU:0'):
            transport=NeuTraTransport(NeuTraTransportConfig.hoffman_author_iaf(3,
                conditional_scale_cap=2.,seed=(261023,34),dtype='float64'),trainable=False)
        checks=map_checks(tf,transport)
        payload=transport.frozen_payload(target_signature=target.target_signature,
            training_state_hash=hashlib.sha256(b'integration_smoke').hexdigest())
    parts=target.hmc_components(payload)
    with tf.device('/CPU:0'):
        q=tf.repeat(target.spec.truth(tf.float64)[None],4,axis=0)
        if parts['transport'] is not None:q=parts['transport'].inverse_theta_to_z_batch(q)
    endpoint=parts['target'].function(q); force=parts['force'].function(q)
    config=NeuralForceHMCConfig(step_size=.005,num_leapfrog_steps=2,inverse_mass_diagonal=(1.,)*target.p)
    @tf.function(input_signature=[tf.TensorSpec([4,target.p],tf.float64)],jit_compile=True,autograph=False)
    def transition(x):
        result=neural_force_hmc_transition(x,parts['target'].function(x).potential,
            parts['force'],parts['target'],config,(261023,34))
        valid=(tf.reduce_all(tf.math.is_finite(result.position)) &
               tf.reduce_all(result.trace.finite_status) & ~tf.reduce_any(result.trace.force_fallback))
        return result,valid
    result,health=transition(q)
    valid=bool(health) and all(ready(endpoint.status['value_valid'])) and all(ready(force.force_valid))
    if checks:valid &= checks['valid']
    from bayesfilter.inference import TensorFlowHMCKernelTuningConfig,FourChainMeanBandAcceptancePolicy
    from bayesfilter.inference.hmc_tensorflow_tuning import _run_tensorflow_hmc_tuning
    smoke=TensorFlowHMCKernelTuningConfig(parameter_dimension=target.p,evidence_role='candidate',
        mass_window_results=(8,),step_adaptation_results=8,verification_results=8,
        max_leapfrog_steps=1,initial_step_size=.005,budget_provenance='Integration mechanics only',
        initial_step_size_provenance='Small smoke step',geometry_provenance='Own-target tiny smoke',
        target_scope=parts['scope'],target_accept_prob=.7,
        acceptance_policy=FourChainMeanBandAcceptancePolicy(overall_band=(0.,1.),per_chain_band=(0.,1.)),
        verification_repair_rounds=0,step_repair_factor=1.5,mass_shrinkage=.1,covariance_jitter=0.,
        eigenvalue_floor=64*sys.float_info.epsilon,max_condition_number=sys.float_info.epsilon**-.5,
        seed=(261023,35),use_xla=True,energy_error_max=None,numerical_device='/GPU:0')
    with tf.device('/CPU:0'):
        starts=q+tf.constant([[-.05],[-.02],[.02],[.05]],tf.float64)
        # The typed adaptive campaign calls this nomination stage directly.
        # The ordinary public config selects a different candidate-set controller.
        tuned=_run_tensorflow_hmc_tuning(adapter=parts['adapter'],initial_position=tuple(map(tuple,ready(starts))),
            config=smoke,parameter_scales=(1.,)*target.p,runner_binding=parts['binding'],
            output_dir=out/'campaign-nomination-smoke')
    valid &= bool(tuned.heuristic_screen_passed)
    from bayesfilter.inference.hmc_tensorflow_verified import _diagnostic_samples, _affine
    with tf.device('/CPU:0'):
        diagnostic_positions=_diagnostic_samples(tf.repeat(q[None],4,axis=0),_affine(tuned),parts['binding'])
        expected_diagnostics=target.p*(3 if parts['transport'] is not None else 2)
        valid &= int(diagnostic_positions.shape[-1])==expected_diagnostics
        valid &= bool(tf.reduce_all(tf.math.is_finite(diagnostic_positions)))
    return dict(status='INTEGRATION_VALID' if valid else 'INTEGRATION_FAILED',
        endpoint=ready(endpoint._asdict()),force=ready(force._asdict()),map_checks=checks,
        trace=ready(result.trace._asdict()),campaign_nomination_passed=bool(tuned.heuristic_screen_passed),
        campaign_nomination_manifest=tuned.artifact_manifest_path,
        convergence_coordinate_count=int(diagnostic_positions.shape[-1]),
        convergence_identity=parts['binding'].diagnostic_identity,posterior_promoted=False)


def worker(job_path):
    job=read(job_path); out=Path(job['output']); start=time.monotonic()
    tf,environment=configure(job['gpu']); save(out/'environment.json',environment)
    try:
        result=WORKERS[job['kind']](tf,job,out)
        result.update(wall_seconds=time.monotonic()-start,environment=environment)
        save(out/'result.json',result)
    except Exception as exc:
        save(out/'error.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
            wall_seconds=time.monotonic()-start))
        raise


def training_builder(tf,target,job):
    """CPU initialization arithmetic, GPU resources and compiled batch updates."""
    from bayesfilter.inference.neutra_transport import (
        NeuTraTransport,NeuTraTransportConfig,NeuTraOptimizerConfig,NeuTraTransportTrainer)
    def gpu_resource(next_creator,**kwargs):
        if callable(kwargs.get('initial_value')):
            with tf.device('/CPU:0'):
                kwargs['initial_value']=kwargs['initial_value']()
        with tf.device('/GPU:0'):
            return next_creator(**kwargs)
    with tf.device('/CPU:0'),tf.variable_creator_scope(gpu_resource):
        center,scale=target.spec.prior_parameters()
        config=replace(NeuTraTransportConfig.hoffman_author_iaf(3,
            conditional_scale_cap=2.,seed=(261023,job['seed']),dtype='float32'),
            hidden_layers=(job['width'],)*2,affine_center=tuple(ready(center)),
            affine_scale=tuple(ready(.5*scale)))
        optimizer=NeuTraOptimizerConfig(batch_size=8,shard_size=4,estimator='standard',
            learning_rate=job['rate'],beta1=.9,beta2=.999,epsilon=1e-7,
            gradient_clip_norm=1.,jit_compile=True,target_dtype='float32')
        trainer=NeuTraTransportTrainer(NeuTraTransport(config),target.value_score,optimizer,
                                       target_signature=target.target_signature)
    variables=(*trainer.variables,*trainer.optimizer.variables)
    devices=[v.handle.device for v in variables]
    if not all('/device:GPU:0' in device for device in devices):
        raise RuntimeError('NeuTra training requires GPU model and optimizer resource variables')
    return trainer,devices


def training_setup(tf,job,out):
    """Two actual native-batch updates; mechanics only, never map promotion."""
    target=target_for(tf,job,'calibration')
    trainer,devices=training_builder(tf,target,dict(width=18,rate=1e-3,seed=17))
    before=trainer.transport.parameter_state();rows=[]
    for index in range(2):
        with tf.device('/CPU:0'):
            z=tf.random.stateless_normal([8,3],[17,index+100],dtype=tf.float32)
        start=time.monotonic()
        with tf.device('/GPU:0'):
            result=trainer.train_step(z)
        rows.append(dict(valid=bool(result['valid']),loss=ready(result['loss']),
            gradient_norm=ready(result['gradient_norm']),iteration=ready(result['iteration']),
            device=result['loss'].device,seconds=time.monotonic()-start))
    changed=before!=trainer.transport.parameter_state()
    compiled=bool(trainer.train_step.get_concrete_function().function_def.attr['_XlaMustCompile'].b)
    valid=changed and compiled and all(r['valid'] and '/device:GPU:0' in r['device'] for r in rows)
    return dict(status='TRAINING_SETUP_VALID' if valid else 'TRAINING_SETUP_FAILED',
        updates=rows,parameters_changed=changed,resource_devices=devices,jit_compile=compiled,
        batch_size=8,native_shard_size=4,samplewise_loop=False,posterior_promoted=False)


def training(tf,job,out):
    target=target_for(tf,job)
    trainer,devices=training_builder(tf,target,job)
    accepted=attempts=consecutive=0; history=[]
    if job.get('warm_start'):
        if job.get('resume'):raise ValueError('Choose parameter warm start or exact resume')
        from bayesfilter.inference.neutra_transport import _hash
        previous=read(job['warm_start']);saved=previous['trainer']
        if saved['checkpoint_hash']!=_hash({k:v for k,v in saved.items() if k!='checkpoint_hash'}):
            raise ValueError('Warm-start checkpoint checksum mismatch')
        if _hash(saved['transport_config'])!=_hash(trainer.transport.config.payload()):
            raise ValueError('Warm-start transport configuration mismatch')
        with tf.device('/CPU:0'):
            trainer.transport.restore_parameters(saved['parameters'])
        save(out/'warm-start.json',dict(source=job['warm_start'],sha256=hash_file(job['warm_start']),
            old_target=previous['target_signature'],new_target=target.target_signature,
            optimizer_restored=False,sampling_state_restored=False))
    if job.get('resume'):
        previous=read(job['resume'])
        if previous['target_signature']!=target.target_signature: raise ValueError('Stale training target')
        with tf.device('/CPU:0'):
            trainer.restore(previous['trainer'])
        accepted=previous['accepted'];attempts=previous['attempts']
        history=previous['history']
    deadline=float(os.environ['BAYESFILTER_WORKER_WALL_DEADLINE'])
    while accepted<job['updates'] and attempts<2*job['updates']+32:
        if time.monotonic()>deadline-120: break
        start=time.monotonic()
        with tf.device('/CPU:0'):
            z=tf.random.stateless_normal([8,3],[job['seed'],attempts+100+job.get('draw_offset',0)],dtype=tf.float32)
        with tf.device('/GPU:0'):
            result=trainer.train_step(z)
        attempts+=1
        valid=bool(result['valid'])
        accepted+=int(valid);consecutive=0 if valid else consecutive+1
        row=ready(dict(accepted=accepted,attempt=attempts,valid=valid,
            loss=result['loss'],gradient_norm=result['gradient_norm'],seconds=time.monotonic()-start))
        # Invalid diagnostics can be NaN. Preserve them as explicit strings, not JSON NaN.
        row={k:('nonfinite' if isinstance(v,float) and not math.isfinite(v) else v) for k,v in row.items()}
        history.append(row)
        save(out/'checkpoint.json',dict(trainer=trainer.checkpoint(),accepted=accepted,attempts=attempts,
            history=history,target_signature=target.target_signature,job=job))
        save(out/'progress.json',row)
        if consecutive>=8: break
    # Common independent heldout draws: nomination only, never posterior promotion.
    heldout=[];valid=True
    @tf.function(input_signature=[tf.TensorSpec([4,3],tf.float32)],jit_compile=True,autograph=False)
    def heldout_evaluate(z):
        q,ld=trainer.transport.forward_and_logdet(z)
        v,_,good=target.value_score(q)
        losses=-v-ld
        return losses,tf.reduce_all(good)&tf.reduce_all(tf.math.is_finite(losses))
    for group in range(8):
        with tf.device('/CPU:0'):
            z=tf.random.stateless_normal([4,3],[261025,group],dtype=tf.float32)
        with tf.device('/GPU:0'):
            losses,good=heldout_evaluate(z)
        valid &= bool(good)
        host_losses=ready(losses)
        if all(math.isfinite(x) for x in host_losses):heldout.extend(host_losses)
    with tf.device('/CPU:0'):
        converted=trainer.transport.as_dtype('float64')
    checks=map_checks(tf,converted)
    inverse_error=checks['inverse_error']
    valid &= checks['valid']
    payload=converted.frozen_payload(target_signature=target.target_signature,
        training_state_hash=trainer.checkpoint()['checkpoint_hash'])
    save(out/'frozen-map.json',payload)
    return dict(status='TRAINING_CANDIDATE_VALID' if valid and accepted>=job['updates'] else 'TRAINING_INCOMPLETE',
        accepted=accepted,attempts=attempts,heldout_losses=heldout,inverse_error=inverse_error,map_checks=checks,
        checkpoint=str(out/'checkpoint.json'),frozen_map=str(out/'frozen-map.json'),
        target_signature=target.target_signature,batch_size=8,native_shard_size=4,
        samplewise_loop=False,resource_devices=devices,numerical_device='/GPU:0',jit_compile=True,
        width=job['width'],rate=job['rate'],seed=job['seed'],
        architecture='bayesfilter_neutra_iaf_author_v1',posterior_promoted=False)


def hmc(tf,job,out):
    from bayesfilter.inference import (tune_hmc_kernel,TensorFlowHMCKernelTuningConfig,
        VerifiedTensorFlowHMCConfig,FourChainMeanBandAcceptancePolicy)
    from bayesfilter.inference.hmc_posterior_diagnostics import compute_coordinate_diagnostics
    target=target_for(tf,job)
    payload=read(job['frozen_map']) if job.get('frozen_map') else None
    parts=target.hmc_components(payload)
    mean,scale=target.spec.prior_parameters(tf.float64)
    if job.get('resume_manifest'):
        parent=read(job['resume_manifest'])
        config=replace(VerifiedTensorFlowHMCConfig.from_payload(parent['config']),resume_manifest=job['resume_manifest'])
    else:
        nomination=TensorFlowHMCKernelTuningConfig(parameter_dimension=target.p,evidence_role='candidate',
            mass_window_results=(128,256),step_adaptation_results=256,verification_results=128,
            max_leapfrog_steps=4,initial_step_size=.01,
            budget_provenance='Owner 48 GPU hour structural Model A/B master',
            initial_step_size_provenance='Small standardized warm start, adapted on this target',
            geometry_provenance='Own-target mass windows, identity shrinkage .1; no transferred covariance',
            target_scope=parts['scope'],target_accept_prob=.7,
            acceptance_policy=FourChainMeanBandAcceptancePolicy(overall_band=(.60,.80),per_chain_band=(.4,.9)),
            verification_repair_rounds=2,step_repair_factor=1.5,mass_shrinkage=.1,covariance_jitter=0.,
            eigenvalue_floor=64*sys.float_info.epsilon,max_condition_number=sys.float_info.epsilon**-.5,
            seed=(261027,11),use_xla=True,energy_error_max=None,numerical_device='/GPU:0')
        with tf.device('/CPU:0'):
            initial=tf.stack([mean+s*scale for s in (-.75,-.25,.25,.75)])
            if parts['transport'] is not None: initial=parts['transport'].inverse_theta_to_z_batch(initial)
        config=VerifiedTensorFlowHMCConfig(nomination=nomination,
            initial_chain_positions=tuple(tuple(row) for row in ready(initial)),
            initialization_provenance='Four dispersed prior-coordinate starts, transformed through frozen map when present',
            warmup_results=2000,warmup_max_results=10000,verification_results=1000,
            verification_max_results=20000,fresh_seed=(261027,12),energy_error_max=1000.,
            chunk_results=250,check_every_results=1000)
    save(out/'config.json',config.payload())
    with tf.device('/CPU:0'):
        result=tune_hmc_kernel(adapter=parts['adapter'],initial_position=config.initial_chain_positions,
            config=config,parameter_scales=(1.,)*target.p,runner_binding=parts['binding'],output_dir=out/'public-tuning')
    manifest=read(result.artifact_manifest_path)
    record=dict(status=manifest['sequence']['status'],manifest=result.artifact_manifest_path,
        counts=manifest['sequence']['counts'],warmup_ready=manifest['sequence']['warmup_ready'],
        posterior_sampling_complete=result.passed,endpoint='exact_Kalman' if target.spec.name=='model_a' else 'frozen_finite_LEDH_approximation',
        all_postwarmup_draws_are_inference=True,no_extra_validation_phase=True,
        ess_role='diagnostic',energy_error_role='diagnostic',approximation_accuracy='requires_reference_comparison')
    with tf.device('/CPU:0'):
        _posterior_report(tf,target,parts,result,manifest,out,record,compute_coordinate_diagnostics)
    return record


def _posterior_report(tf,target,parts,result,manifest,out,record,compute_coordinate_diagnostics):
    if manifest['verification']:
        root=Path(result.artifact_manifest_path).parent
        def tensor(key):
            item=manifest['verification'][key];path=root/item['path']
            if hash_file(path)!=item['sha256']: raise ValueError('Sample checksum mismatch')
            return tf.io.parse_tensor(tf.io.read_file(str(path)),tf.as_dtype(item.get('dtype','float64')))
        states=tensor('states')
        q=result.nomination.center+tf.linalg.matmul(states,result.nomination.factor,transpose_b=True)
        if parts['transport'] is not None:
            q=tf.reshape(parts['transport'].forward_batch(tf.reshape(q,[-1,target.p])),tf.shape(q))
        physical=target.spec.physical(q)
        import tensorflow_probability as tfp
        record.update(q_diagnostics=compute_coordinate_diagnostics(tf.transpose(q,[1,0,2])),
            physical_diagnostics=compute_coordinate_diagnostics(tf.transpose(physical,[1,0,2])),
            q_mean=tf.reduce_mean(q,axis=[0,1]),physical_mean=tf.reduce_mean(physical,axis=[0,1]),
            credible_interval=tfp.stats.percentile(tf.reshape(physical,[-1,target.p]),[2.5,97.5],axis=0),
            truth=target.spec.physical(target.spec.truth(tf.float64)),
            acceptance=tf.reduce_mean(tf.cast(tensor('is_accepted'),tf.float64),axis=0),
            acceptance_probability=tf.reduce_mean(tf.exp(tf.minimum(tensor('log_accept_ratio'),0.)),axis=0),
            max_abs_delta_h=tf.reduce_max(tf.abs(tensor('delta_h'))),
            force_fallback_count=tf.reduce_sum(tf.cast(tensor('force_fallback'),tf.int32)))
        tf.io.write_file(str(out/'posterior-q.tfbin'),tf.io.serialize_tensor(q))
        record['posterior_q']=str(out/'posterior-q.tfbin')


def assessment(tf,job,out):
    """Independent reference reporting only; cannot feed tuning or sampling."""
    from bayesfilter.testing.structural_reference_diagnostics import kalman_posterior_grid
    from bayesfilter.testing.structural_ledh_models_tf import StructuralModelSpec
    root=Path(job['root']); state=read(root/'state.json'); rows={}
    entry=state.get('hmc',{}).get('model_a',{}).get('result',{})
    if entry.get('posterior_q'):
        data=read(prepared_paths(root,'model_a',50,'final')[0])
        mean,scale=StructuralModelSpec('model_a').prior_parameters(tf.float64)
        references=[kalman_posterior_grid(data['observations'],ready(mean),ready(scale),n)
                    for n in (129,257)]
        low,high=references
        error=[abs(a-b) for a,b in zip(low['mean'],high['mean'])]
        sample=entry['q_diagnostics']['mean']
        tolerance=[max(.1*s,2*m+e) for s,m,e in zip(high['sd'],sample['mean_mcse'],error)]
        differences=[abs(a-b) for a,b in zip(sample['pooled_mean'],high['mean'])]
        refined=high['boundary_mass']<1e-7 and all(e<.01*s for e,s in zip(error,high['sd']))
        rows['model_a']=dict(reference=high,refinement=error,boundary_mass=high['boundary_mass'],
            mean_errors=differences,tolerance=tolerance,reference_refined=refined,
            posterior_agreement=refined and all(d<=t for d,t in zip(differences,tolerance)),
            interpretation='Mean agreement with refined exact Kalman posterior; separate from convergence')
    else:rows['model_a']=dict(status='NO_POSTERIOR_SAMPLES')
    rows['model_b']=dict(status='FINITE_LEDH_POSTERIOR_ONLY',
        posterior_accuracy='Long-horizon exact posterior agreement not established',
        reference_evidence='Per-scope likelihood shape and scores, including PF MCSE and refinement',
        short_posterior='Not sampled; short-target checks concern likelihood and score only')
    return dict(status='ASSESSMENT_COMPUTED',models=rows,scientific_default_promoted=False)


WORKERS={'prepare':prepare,'references':references,'probe':probe,'integration':integration,'reset_precision_check':reset_precision_check,
         'training_setup':training_setup,'training':training,'hmc':hmc,'assessment':assessment}


class Campaign:
    def __init__(self,root,hours=None,migration_note=None,repair_model_b_reset=False):
        self.root=Path(root).resolve(); self.path=self.root/'state.json'
        if self.path.exists():
            self.state=read(self.path)
            if self.state['source_identity']!=source_identity():
                if not migration_note or not Path(migration_note).is_file():
                    raise ValueError('Source changed; provide --migration-note pointing to reviewed repair evidence')
                if any('finished' not in a and a.get('pid') for a in self.state['attempts']):
                    raise ValueError('Recover unfinished workers before migrating numerical source')
                new=source_identity(); old=self.state['source_identity']
                changed=[p for p in set(old)|set(new) if old.get(p)!=new.get(p)]
                if repair_model_b_reset:
                    self.migrate_model_b_reset(old,new,changed,migration_note)
                    return
                self.state.setdefault('source_migrations',[]).append(dict(time=now(),changed=sorted(changed),
                    previous_source_identity=old,note=str(Path(migration_note).resolve()),
                    note_sha256=hash_file(migration_note),remaining_gpu_seconds=self.remaining(True)))
                # Keep prepared data only if its numerical generator is unchanged.
                regenerate_data='bayesfilter/testing/structural_ledh_models_tf.py' in changed
                for a in self.state['attempts']:
                    if a['kind']!='prepare' or regenerate_data:a['superseded_by_source_migration']=True
                self.state['source_identity']=new
                self.state['selection']={}
                self.state.pop('hmc',None);self.state.pop('frozen_map',None)
                self.state.pop('failed_scopes',None)
                self.persist()
        else:
            if hours is None or not 0<hours<=48: raise ValueError('Need authorized GPU budget, <=48 hours')
            self.root.mkdir(parents=True,exist_ok=False)
            self.state=dict(schema='structural_ledh_ab_campaign_v1',created=now(),
                gpu_budget_seconds=hours*3600,cpu_budget_seconds=7200,attempts=[],phase='prepared',
                source_identity=source_identity(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
                plan=PLAN,selection={},decision='RUNNING',next_action='CPU preparation')
            self.persist()

    def migrate_model_b_reset(self,old,new,changed,note):
        allowed={'scripts/run_structural_ledh_ab.py',
            'bayesfilter/highdim/covariance_proposal_score_batch_tf.py',
            'bayesfilter/testing/structural_ledh_target.py',
            'bayesfilter/inference/hmc_tensorflow_verified.py'}
        if not set(changed)<=allowed:raise ValueError('Reset repair includes unrelated numerical changes')
        if 'reset_repair' in self.state:
            if set(changed)!={'scripts/run_structural_ledh_ab.py'} or any(
                a['key'].startswith(('B-reset-train','model_b-reset-hmc')) for a in self.state['attempts']):
                raise ValueError('Further numerical changes need a new reviewed migration')
            # Reviewed diagnostic-only repair before any new-target training/sampling.
            oldpath=self.state['selection']['model_b-T50'];selection=read(oldpath)
            version=len(self.state.get('source_migrations',[]))+1
            path=self.root/'selections'/f'model_b-T50-reset-harness-v{version}.json'
            if path.exists():raise ValueError('Harness migration receipt already exists')
            selection.update(source_identity=new,predecessor_selection=oldpath,predecessor_sha256=hash_file(oldpath))
            save(path,selection)
            self.state.setdefault('source_migrations',[]).append(dict(time=now(),changed=sorted(changed),
                previous_source_identity=old,new_source_identity=new,note=str(Path(note).resolve()),
                note_sha256=hash_file(note),scope='Reviewed pretraining diagnostic harness repair',
                remaining_gpu_seconds=self.remaining(True)))
            self.state['source_identity']=new;self.state['selection']['model_b-T50']=str(path)
            self.phase('reset_repair_checks','Retry production diagnostic; numerical target unchanged')
            return
        if not self.state.get('hmc',{}).get('model_a',{}).get('result',{}).get('posterior_sampling_complete'):
            raise ValueError('Scoped repair requires preserved complete Model A evidence')
        selection_path=self.state['selection']['model_b-T50'];selection=read(selection_path)
        checkpoint=Path(self.state['frozen_map']).with_name('checkpoint.json');previous=read(checkpoint)
        path=self.root/'selections/model_b-T50-reset-fp64-v1.json'
        if path.exists():raise ValueError('Repair selection already exists')
        selection.update(source_identity=new,reset_precision_policy='conditional_fp64',
            predecessor_selection=selection_path,predecessor_sha256=hash_file(selection_path),
            selection_rule='Frozen controls retained for isolated arithmetic repair; production checks required',
            approximation_promoted=False)
        save(path,selection)
        self.state.setdefault('source_migrations',[]).append(dict(time=now(),changed=sorted(changed),
            previous_source_identity=old,new_source_identity=new,note=str(Path(note).resolve()),
            note_sha256=hash_file(note),scope='Model B reset only; A and original references preserved',
            remaining_gpu_seconds=self.remaining(True)))
        self.state['reset_repair']=dict(version=1,warm_start=str(checkpoint),
            warm_start_sha256=hash_file(checkpoint),training_job=previous['job'],
            prior_b_hmc=self.state.get('hmc',{}).pop('model_b',None),prior_frozen_map=self.state.pop('frozen_map'))
        self.state['selection']['model_b-T50']=str(path);self.state['source_identity']=new
        self.phase('reset_repair_checks','Verify production reset before fresh Model B training and HMC')

    def continue_reset_repair(self):
        repair=self.state['reset_repair']
        path=self.state['selection']['model_b-T50'];selection=read(path)
        args=dict(model='model_b',horizon=50,controls=selection['controls'],selection=path,
                  reset_precision_policy='conditional_fp64')
        self.phase('reset_repair_checks','Production endpoint precision and posterior wiring checks')
        result,_=self.task('B-reset-production-check','reset_precision_check',900,**args)
        if result['status']!='RESET_PRECISION_VALID':return self.finish('RESET_REPAIR_CHECK_FAILED')
        self.phase('reset_repair_training','Parameter-only warm start; fresh Adam and 256 accepted own-target updates')
        old=repair['training_job']
        if hash_file(repair['warm_start'])!=repair['warm_start_sha256']:
            return self.finish('WARM_START_EVIDENCE_CHANGED')
        trained,_=self.task('B-reset-train','training',5400,**args,
            width=old['width'],rate=old['rate'],seed=old['seed'],updates=256,
            draw_offset=10000,warm_start=repair['warm_start'])
        if trained['status']!='TRAINING_CANDIDATE_VALID':return self.finish('RESET_TRAINING_INCOMPLETE')
        self.state['frozen_map']=trained['frozen_map'];self.persist()
        self.phase('adaptive_hmc','Fresh Model B nomination, adaptive warmup and cumulative posterior sampling')
        hmc_args=dict(args,frozen_map=trained['frozen_map'])
        result,out=self.task('model_b-reset-hmc','hmc',36000,**hmc_args)
        if not result.get('posterior_sampling_complete') and result.get('manifest'):
            result,out=self.task('model_b-reset-hmc-continued','hmc',36000,
                resume_manifest=result['manifest'],**hmc_args)
        self.state.setdefault('hmc',{})['model_b']=dict(result=result,output=str(out) if out else None);self.persist()
        self.phase('analysis','Report retained Model A and repaired Model B separately from reference accuracy')
        assessed,_=self.task('reset-posterior-assessment','assessment',300,gpu=False)
        self.state['assessment']=assessed;self.persist();self.report()
        done=all(self.state['hmc'][m]['result'].get('posterior_sampling_complete') for m in ('model_a','model_b'))
        return self.finish('CAMPAIGN_EXECUTED_SEE_REFERENCE_LIMITS' if done else 'SAMPLING_INCOMPLETE')

    def persist(self):
        self.state['updated']=now(); save(self.path,self.state)
        save(self.root/'next-phase.json',dict(phase=self.state['phase'],next_action=self.state['next_action'],
            remaining_gpu_seconds=self.remaining(True),remaining_cpu_seconds=self.remaining(False)))

    def remaining(self,gpu):
        return self.state['gpu_budget_seconds' if gpu else 'cpu_budget_seconds']-sum(
            a.get('charged_seconds',0.) for a in self.state['attempts'] if a['gpu']==gpu)

    def phase(self,name,next_action):
        self.state.update(phase=name,next_action=next_action,decision='RUNNING'); self.persist()

    def recover(self):
        """Account interrupted attempts without launching duplicate GPU workers."""
        for attempt in self.state['attempts']:
            if 'finished' in attempt or not attempt.get('output'): continue
            pid=attempt.get('pid');out=Path(attempt['output'])
            command_file=Path(f'/proc/{pid}/cmdline') if pid else None
            def owns_worker():
                try: return str(out/'job.json').encode() in command_file.read_bytes()
                except (OSError,AttributeError): return False
            initial=datetime.fromisoformat(attempt['started']).timestamp()
            while owns_worker():
                elapsed=time.time()-initial
                attempt['charged_seconds']=max(attempt.get('charged_seconds',0),elapsed)
                self.persist()
                if elapsed>=attempt['allowance_seconds']:
                    os.killpg(pid,signal.SIGTERM)
                    time.sleep(2)
                    if owns_worker(): os.killpg(pid,signal.SIGKILL)
                    break
                time.sleep(10)
            result_path=out/'result.json'
            charged=read(result_path).get('wall_seconds',0) if result_path.exists() else attempt['allowance_seconds']
            attempt.update(charged_seconds=max(attempt.get('charged_seconds',0),charged),
                finished=now(),exit_code=0 if result_path.exists() else -1,
                recovery='Completed receipt recovered; otherwise allowance charged conservatively')
            self.persist()

    def task(self,key,kind,allowance,gpu=True,**kw):
        all_old=[a for a in self.state['attempts'] if a['key']==key]
        old=[a for a in all_old if not a.get('superseded_by_source_migration')]
        for previous in reversed(old):
            path=Path(previous['output'])/'result.json'
            if previous.get('exit_code')==0 and path.exists(): return read(path),Path(previous['output'])
        if len(old)>=3: return dict(status='ATTEMPT_LIMIT'),None
        cap=min(allowance,self.remaining(gpu))
        if cap<60: return dict(status='BUDGET_EXHAUSTED'),None
        out=self.root/'attempts'/f'{key}-{len(all_old)+1:02d}'; out.mkdir(parents=True,exist_ok=False)
        job=dict(root=str(self.root),output=str(out),kind=kind,gpu=gpu,**kw); save(out/'job.json',job)
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--job',str(out/'job.json')]
        attempt=dict(key=key,kind=kind,gpu=gpu,output=str(out),command=command,
                     started=now(),allowance_seconds=cap,charged_seconds=0.)
        self.state['attempts'].append(attempt); self.persist()
        env=dict(os.environ,BAYESFILTER_WORKER_WALL_DEADLINE=str(time.monotonic()+cap-30))
        start=time.monotonic()
        with (out/'worker.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            attempt['pid']=process.pid; self.persist()
            while process.poll() is None:
                elapsed=time.monotonic()-start
                attempt['charged_seconds']=min(elapsed,cap); self.persist()
                if elapsed>=cap:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
                    break
                time.sleep(10)
        attempt.update(charged_seconds=time.monotonic()-start,exit_code=process.returncode,finished=now()); self.persist()
        if (out/'result.json').exists(): return read(out/'result.json'),out
        return dict(status='WORKER_FAILED',exit_code=process.returncode,error=str(out/'error.json')),out

    def run(self,until='all'):
        self.recover()
        if 'reset_repair' in self.state:return self.continue_reset_repair()
        self.phase('prepare','Create disjoint data and fixed particle designs on CPU')
        result,_=self.task('prepare','prepare',300,gpu=False)
        if result['status']!='PREPARED': return self.finish('PREPARATION_REPAIR_REQUIRED')
        self.phase('pilot','Measure FP32/XLA N4032 native batch feasibility on GPU0')
        for model in ('model_a','model_b'):
            result,_=self.task(model+'-pilot','probe',1800,model=model,horizon=2,controls=CONTROLS[0])
            if result['status']!='PROBE_VALID': return self.finish('PILOT_REPAIR_REQUIRED')
        if until=='pilot': return self.finish('PILOT_COMPLETE_CAMPAIGN_PENDING')
        self.phase('integration','Check actual XLA HMC endpoint, force and canonical map wiring')
        result,_=self.task('model_b-training-setup','training_setup',900,
            model='model_b',horizon=2,controls=CONTROLS[0])
        if result['status']!='TRAINING_SETUP_VALID':return self.finish('TRAINING_SETUP_REPAIR_REQUIRED')
        for model in ('model_a','model_b'):
            result,_=self.task(model+'-integration','integration',1200,model=model,horizon=2,controls=CONTROLS[0])
            if result['status']!='INTEGRATION_VALID':return self.finish('INTEGRATION_REPAIR_REQUIRED')
        if until=='integration':return self.finish('INTEGRATION_COMPLETE_CAMPAIGN_PENDING')
        return self.continue_phases()

    def continue_phases(self):
        self.phase('references','Independent Kalman, refined integration and replicated PF references')
        result,_=self.task('references','references',6900,gpu=False)
        if result['status']!='REFERENCES_COMPUTED': return self.finish('REFERENCE_REPAIR_REQUIRED')
        self.phase('scope_tuning','Freeze controls separately for each model, horizon, data regime and design')
        for model,horizon in SCOPES:
            key=scope_key(model,horizon)
            if key in self.state['selection']: continue
            candidates=[]
            reference=read(self.root/'references'/f'{key}-calibration.json')['estimates']
            for index,controls in enumerate(CONTROLS):
                result,out=self.task(f'{key}-cal-{index}','probe',1200,
                    model=model,horizon=horizon,controls=controls,partition='calibration')
                if result['status']!='PROBE_VALID': continue
                row=result['rows'][-1]
                # Nomination loss compares likelihood shape, not arbitrary constants.
                centered=[(v-row['value'][0])-(r['mean'][0]-reference[0]['mean'][0])
                          for v,r in zip(row['value'],reference)]
                score_errors=[[s-r for s,r in zip(score,ref['mean'][1:])]
                              for score,ref in zip(row['score'],reference)]
                loss=sum(x*x for x in centered)/len(centered)
                loss+=sum(x*x/(1+r*r) for errors,ref in zip(score_errors,reference)
                          for x,r in zip(errors,ref['mean'][1:]))/(len(reference)*len(score_errors[0]))
                candidates.append(dict(index=index,controls=controls,nomination_loss=loss,
                    centered_errors=centered,score_errors=score_errors,result=str(out/'result.json')))
            if not candidates: return self.finish('SCOPE_TUNING_REPAIR_REQUIRED')
            candidates.sort(key=lambda x:x['nomination_loss'])
            chosen=None
            for candidate in candidates:
                validation,out=self.task(f'{key}-val-{candidate["index"]}','probe',1200,
                    model=model,horizon=horizon,controls=candidate['controls'],partition='validation')
                if validation['status']=='PROBE_VALID':
                    chosen=candidate;break
            if chosen is None: return self.finish('SCOPE_VALIDATION_REPAIR_REQUIRED')
            data,design=prepared_paths(self.root,model,horizon,'final')
            selection=dict(model=model,horizon=horizon,controls=chosen['controls'],particles=4032,
                dtype='float32',tf32=False,chunk=2016,source_identity=source_identity(),
                data_sha256=hash_file(data),design_sha256=hash_file(design),
                calibration=candidates,validation=str(out/'result.json'),
                selection_rule='minimum calibration shape/normalized score loss among finite candidates',
                ranking='descriptive nomination only, no statistical superiority claimed',
                approximation_promoted=False)
            path=self.root/'selections'/f'{key}.json';save(path,selection)
            self.state['selection'][key]=str(path);self.persist()
        self.phase('untouched_probes','Evaluate frozen settings on final data without retuning')
        for model,horizon in SCOPES:
            selection=read(self.state['selection'][scope_key(model,horizon)])
            result,_=self.task(f'{scope_key(model,horizon)}-final','probe',1200,
                model=model,horizon=horizon,controls=selection['controls'],partition='final')
            if result['status']!='PROBE_VALID':
                self.state.setdefault('failed_scopes',[]).append(scope_key(model,horizon));self.persist()
        self.phase('neutra_training','Model B own-target capacity/rate/seed ladder, then frozen map')
        if 'model_b-T50' in self.state.get('failed_scopes',[]):
            return self.finish('FINAL_SCOPE_REPAIR_REQUIRED')
        selection_path=self.state['selection']['model_b-T50'];selection=read(selection_path)
        arms=[]
        # Capacity/rate hypotheses and two seeds; shared heldout nominates only.
        for width,rate in ((18,1e-3),(36,3e-4)):
            for seed in (17,29):
                key=f'B-train-w{width}-s{seed}'
                result,out=self.task(key,'training',5400,model='model_b',horizon=50,
                    controls=selection['controls'],selection=selection_path,width=width,rate=rate,seed=seed,updates=256)
                if result['status']=='TRAINING_CANDIDATE_VALID':
                    arms.append(dict(key=key,result=result,out=str(out),width=width,rate=rate,seed=seed,
                        mean_loss=sum(result['heldout_losses'])/len(result['heldout_losses'])))
        if not arms: return self.finish('TRAINING_REPAIR_REQUIRED')
        arms.sort(key=lambda x:x['mean_loss'])
        save(self.root/'training-nomination.json',dict(arms=arms,
            rule='heldout ELBO nominates; frozen-map HMC determines downstream usefulness',
            statistically_supported_ranking=False))
        winner=arms[0]
        final,out=self.task('B-train-extend','training',14400,model='model_b',horizon=50,
            controls=selection['controls'],selection=selection_path,width=winner['width'],rate=winner['rate'],
            seed=winner['seed'],updates=1024,resume=winner['result']['checkpoint'])
        if final['status']!='TRAINING_CANDIDATE_VALID': return self.finish('TRAINING_EXTENSION_INCOMPLETE')
        self.state['frozen_map']=final['frozen_map'];self.persist()
        self.phase('adaptive_hmc','Adaptive warmup and cumulative posterior sampling to Rhat <=1.01 or caps')
        for model in ('model_a','model_b'):
            if scope_key(model,50) in self.state.get('failed_scopes',[]):continue
            selection_path=self.state['selection'][scope_key(model,50)]
            selection=read(selection_path)
            args=dict(model=model,horizon=50,controls=selection['controls'],selection=selection_path)
            if model=='model_b': args['frozen_map']=self.state['frozen_map']
            result,out=self.task(model+'-hmc','hmc',36000,**args)
            # Resume an intact sequential checkpoint rather than restarting adaptation.
            if not result.get('posterior_sampling_complete') and result.get('manifest'):
                result,out=self.task(model+'-hmc-continued','hmc',18000,
                    resume_manifest=result['manifest'],**args)
            self.state.setdefault('hmc',{})[model]=dict(result=result,output=str(out));self.persist()
        self.phase('analysis','Assemble diagnostics, independent reference discrepancies and unresolved promotion criteria')
        assessment_result,_=self.task('posterior-assessment','assessment',300,gpu=False)
        self.state['assessment']=assessment_result;self.persist()
        self.report()
        done=len(self.state['hmc'])==2 and all(x['result'].get('posterior_sampling_complete',False) for x in self.state['hmc'].values())
        return self.finish('CAMPAIGN_EXECUTED_SEE_REFERENCE_LIMITS' if done else 'SAMPLING_INCOMPLETE')

    def report(self):
        rows=[]
        for model,horizon in SCOPES:
            key=scope_key(model,horizon)
            attempt=next((a for a in reversed(self.state['attempts']) if a['key']==key+'-final' and a.get('exit_code')==0),None)
            if not attempt: continue
            candidate=read(Path(attempt['output'])/'result.json')['rows'][-1]
            reference_record=read(self.root/'references'/f'{key}-final.json')
            reference=reference_record['estimates']
            rows.append(dict(scope=key,candidate=candidate,reference=reference,
                candidate_artifact=str(Path(attempt['output'])/'result.json'),
                precision_policy='original_uniform',
                interpretation='Historical original-precision probes; reset repair has separate evidence' if self.state.get('reset_repair') else 'Frozen final probes',
                heuristics=reference_record.get('heuristics',[]),
                centered_likelihood_errors=[(v-candidate['value'][0])-(r['mean'][0]-reference[0]['mean'][0])
                    for v,r in zip(candidate['value'],reference)],
                score_errors=[[x-y for x,y in zip(s,r['mean'][1:])] for s,r in zip(candidate['score'],reference)]))
        save(self.root/'analysis.json',dict(scopes=rows,hmc=self.state.get('hmc',{}),
            reset_repair=self.state.get('reset_repair'),
            assessment=self.state.get('assessment',{}),
            heuristic_dominance=dict(verdict='NO_COMPUTATIONAL_ADVANTAGE_CLAIM',
                rationale='Gaussian/reference outputs reported by scope; no paired runtime ladder or statistical ranking'),
            decision_table=dict(decision='scope-specific evidence; no automatic algorithm promotion',
                primary_criterion='HMC convergence separate from approximation accuracy',
                veto='See per-scope numerical and reference status',uncertainty='PF bias and finite-design posterior error',
                next_action='Quantify posterior/reference agreement where unresolved',
                not_concluded='Unbiased likelihood, exact nonlinear posterior, superiority or default readiness'),
            inference_status=dict(hard_veto_screen='per-scope artifacts',statistically_supported_ranking=False,
                descriptive_only='timing, tuning losses, point-score differences',default_readiness=False,
                next_evidence='posterior/reference agreement with MCSE and reference refinement')))

    def finish(self,status):
        self.state['decision']=status; self.persist()
        print(json.dumps(dict(status=status,state=str(self.path),remaining_gpu_hours=self.remaining(True)/3600)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','resume','status','worker'))
    parser.add_argument('--output');parser.add_argument('--job')
    parser.add_argument('--migration-note',help='Existing reviewed repair note; preserves spent budget and invalidates stale numerical receipts')
    parser.add_argument('--repair-model-b-reset',action='store_true',help='Reviewed scoped precision repair preserving completed Model A')
    parser.add_argument('--gpu-hours',type=float);parser.add_argument('--until',choices=('pilot','integration','all'),default='all')
    args=parser.parse_args()
    if args.command=='worker': return worker(args.job)
    if args.command=='status':
        state=read(Path(args.output)/'state.json')
        print(json.dumps({k:state[k] for k in ('phase','decision','next_action','updated')},indent=2));return
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign=Campaign(output,args.gpu_hours if args.command=='run' else None,args.migration_note,args.repair_model_b_reset)
        campaign.run(args.until)


if __name__=='__main__': main()
