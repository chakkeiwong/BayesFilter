"""FP64 GPU/XLA diagnostic attribution; shared flow/training numerical authority.

Exact CPU samples remove teacher quality from this experiment. The affine DSF
readout and alternate conditioner are diagnostic interventions, not defaults.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig, NeuTraOptimizerConfig
from bayesfilter.inference.neutra_joint_training import JointNeuTraTrainer
from bayesfilter.testing.neutra_generic_targets import ExactTargetEvaluator
from bayesfilter.testing.neutra_source_fit_remedy import train_block
from bayesfilter.testing.neutra_scientific_campaign import (
    evaluate_student, freeze_and_assess, save_tensor, serializable, write,
)

F64 = tf.float64


def configuration(dimension, seed, conditioner, *, affine=False, width=64):
    if conditioner.startswith('iaf'):
        config = NeuTraTransportConfig.hoffman_author_iaf(dimension,
            conditional_scale_cap=2., seed=(seed,731))
        return replace(config, hidden_layers=(width,width),
            scale_transform='identity' if conditioner=='iaf_uncapped' else 'neutra_conditional_tanh')
    return replace(NeuTraTransportConfig.huang_dsf(dimension, hidden_layers=(width,width),
        stages=3, mixture_components=4, seed=(seed,731)),
        naf_conditioner=conditioner, diagnostic_naf_affine=affine)


def make_trainer(config, target, rate, *, parameters=None, jit_compile=True):
    flow=NeuTraTransport(config)
    if parameters is not None:
        flow.restore_parameters(parameters)
    optimizer=NeuTraOptimizerConfig(64,'standard',rate,.9,.999,1e-8,1000.,jit_compile)
    return JointNeuTraTrainer(flow,target.value_score,optimizer,
        target_signature=target.signature,teacher_id='attribution-exact-iid',
        forward_weight=1.,reverse_weight=0.)


def prepare(specification, seed, updates, output, *, reference_rows=131072):
    tasks=(('train',updates*64,23001),('validation',32768,23002),('reference',reference_rows,23003))
    def generate(task):
        name,n,stream=task
        with tf.device('/CPU:0'):
            evaluator=ExactTargetEvaluator(specification)
            sample=tf.function(lambda key:evaluator.sample(n,key),
                input_signature=[tf.TensorSpec([2],tf.int32)],jit_compile=True,autograph=False)
            rows=sample(tf.constant([seed,stream],tf.int32))
            digest=save_tensor(output/(name+'.tensor'),rows)
        return name,rows,dict(rows=n,seed=[seed,stream],sha256=digest,device=rows.device)
    with ThreadPoolExecutor(max_workers=2) as pool:
        generated=list(pool.map(generate,tasks))
    write(output/'data-provenance.json',dict(cpu_workers=2,
        worker_kind='threads_with_native_TensorFlow_kernels',
        target_specification=specification,banks={n:p for n,_,p in generated}))
    return {n:r for n,r,_ in generated}


def density_rows(flow, rows):
    """Fixed batch signature bounds tracing/compilation across references."""
    batch=4096
    if int(rows.shape[0]) % batch:
        raise ValueError('reference rows must be divisible by 4096')
    fn=tf.function(flow.log_prob,input_signature=[tf.TensorSpec([batch,flow.parameter_dim],F64)],
                   jit_compile=True,autograph=False)
    return tf.concat([fn(rows[i:i+batch]) for i in range(0,int(rows.shape[0]),batch)],0)


def mean_se(rows):
    mean=tf.reduce_mean(rows)
    n=tf.cast(tf.size(rows),F64)
    se=tf.sqrt(tf.reduce_sum(tf.square(rows-mean))/(n*(n-1.)))
    return {'mean':float(mean),'standard_error':float(se),'rows':int(tf.size(rows)),
            'normal_approximation_99_lower':float(mean-2.576*se),
            'normal_approximation_99_upper':float(mean+2.576*se)}


def parameter_inventory(flow):
    stages=[]
    for stage in flow.stages:
        stages.append(dict(nominal_parameters=sum(int(tf.size(v)) for v in stage.trainable_variables),
            mask_nonzero_weight_entries=sum(int(tf.reduce_sum(m)) for m in stage.masks),
            total_masked_matrix_entries=sum(int(tf.size(m)) for m in stage.masks),
            bias_entries=sum(int(tf.size(v)) for v in stage.biases),
            extra_parameters=sum(int(tf.size(v)) for group in stage.extra_parameters.values() for v in group)))
    return dict(stages=stages,
        interpretation='mask connectivity counts, not identifiable functional dimension; '
        'cMADE normalizes final directions before masking, so masked entries can affect normalization')


def run_fit(config,target,evaluator,data,seed,output,label,*,updates,rate=.0003,
            parameters=None,pricing=False):
    trainer=make_trainer(config,target,.001,parameters=parameters)
    initial=trainer.checkpoint()
    write(output/(label+'-initial.json'),initial)
    initial_metrics=evaluate_student(evaluator,trainer.transport,data['validation'],seed=seed)
    initial_hash=hashlib.sha256(json.dumps(initial['base']['parameters'],sort_keys=True).encode()).hexdigest()
    parameter_count=sum(int(tf.size(v)) for v in trainer.variables)
    history=[]
    offset=0
    for phase,learning_rate in enumerate((.001,rate)):
        if phase:
            trainer=make_trainer(config,target,learning_rate,
                parameters=trainer.transport.parameter_state())
        remaining=updates//2
        while remaining:
            block=min(2048 if not pricing else 512,remaining)
            record=train_block(trainer,data['train'],fresh=True,updates=block,
                seed=[seed,23004],counter=offset)
            history.append({**record,'learning_rate':learning_rate,'phase':phase})
            write(output/(label+'-history.json'),history)
            if not record['finite'] or record['updates']!=block:
                write(output/(label+'-failed-checkpoint.json'),trainer.checkpoint())
                raise ValueError('invalid attribution training: '+label)
            offset+=block
            remaining-=block
    endpoint=freeze_and_assess(output,label,trainer,evaluator,data['validation'],seed)
    probe=json.loads((output/(label+'-post-training-1000.json')).read_text())
    if not probe['complete'] or not probe['finite'] or probe['valid_rows']!=1000:
        raise ValueError('attribution endpoint numerical probe invalid: '+label)
    logq=density_rows(trainer.transport,data['reference'])
    tf.debugging.assert_all_finite(logq,'reference density invalid')
    save_tensor(output/(label+'-reference-logq.tensor'),logq)
    endpoint.update(initial_parameter_hash=initial_hash,initial=initial_metrics,
        parameter_count=parameter_count,training=history,
        parameter_inventory=parameter_inventory(trainer.transport),
        training_wall_seconds=sum(r['wall_seconds'] for r in history),
        updates=sum(r['updates'] for r in history),config=config.payload())
    return endpoint,logq,trainer


def run_attribution(specification,profile,seed,output):
    output=Path(output)
    if profile.get('action')=='equal_compute':
        return run_equal_compute(specification,profile,seed,output)
    updates=profile.get('updates',16384)
    pricing=profile.get('pricing_only',False)
    data=prepare(specification,seed,updates,output,
        reference_rows=4096 if pricing else 131072)
    evaluator=ExactTargetEvaluator(specification)
    conditioner=profile.get('conditioner','author_cmade')
    capacity=conditioner.startswith('iaf')
    endpoint_data={}
    densities={}
    configs={name:configuration(evaluator.dimension,seed,conditioner,affine=name=='affine',
        width=profile.get('width',64)) for name in (('canonical',) if capacity else ('affine','nonlinear'))}
    initial=NeuTraTransport(next(iter(configs.values()))).parameter_state()
    order=list(configs)
    if seed%4==1:
        order.reverse()
    for label in order:
        endpoint,logq,trainer=run_fit(configs[label],evaluator.target,evaluator,data,seed,output,label,
            updates=updates,rate=profile.get('final_rate',.0003),parameters=initial,pricing=pricing)
        endpoint_data[label]=endpoint
        densities[label]=logq
    result=dict(status='attribution_pair_complete',seed=seed,conditioner=conditioner,
        target_signature=evaluator.target.signature,profile=profile,endpoints=endpoint_data,
        order=order,scientific_promotion=False,downstream_hmc='not_run',
        dtype='float64_diagnostic_exception',role=profile.get('role','development'))
    with tf.device('/CPU:0'):
        logp=evaluator.target.log_prob_kernel(data['reference'])
    save_tensor(output/'reference-logp.tensor',logp)
    for label,logq in densities.items():
        endpoint_data[label]['forward_kl_reference']=mean_se(logp-logq)
    if not capacity:
        if endpoint_data['affine']['initial_parameter_hash']!=endpoint_data['nonlinear']['initial_parameter_hash']:
            raise ValueError('paired initialization differs')
        if endpoint_data['affine']['parameter_count']!=endpoint_data['nonlinear']['parameter_count']:
            raise ValueError('paired parameter counts differ')
        difference=densities['nonlinear']-densities['affine']
        save_tensor(output/'paired-difference.tensor',difference)
        result['paired_kl_benefit']=mean_se(difference)
        result['same_initial_parameters']=True
    if specification.get('theorem_control'):
        with tf.device('/CPU:0'):
            x=data['reference'][:,0]
            optimal=-.5*(tf.math.log(tf.constant(20.*math.pi,F64))+x*x/10.)
        result['optimal_gaussian_kl']=mean_se(logp-optimal)
        result['affine_kl_lower_bound']=.5*math.log(10)-math.log(2)
        if not capacity:
            result['nonlinear_below_proven_bound']=endpoint_data['nonlinear']['forward_kl_reference']['normal_approximation_99_upper']<result['affine_kl_lower_bound']
    write(output/'result.json',result)
    return result


def run_equal_compute(specification,profile,seed,output):
    """Continue saved affine Adam until the paired nonlinear training time.

    This is a separate comparator; the original fixed-update pair stays intact.
    """
    parent=Path(profile['parent'])
    parent_result=json.loads((parent/'result.json').read_text())
    checkpoint=json.loads((parent/'affine-checkpoint.json').read_text())
    evaluator=ExactTargetEvaluator(specification)
    if checkpoint['base']['target_signature']!=evaluator.target.signature:
        raise ValueError('equal-compute parent target mismatch')
    config=NeuTraTransportConfig(**parent_result['endpoints']['affine']['config'])
    rate=parent_result['profile'].get('final_rate',.0003)
    trainer=make_trainer(config,evaluator.target,rate)
    trainer.restore(checkpoint)
    original_time=parent_result['endpoints']['affine']['training_wall_seconds']
    required_time=parent_result['endpoints']['nonlinear']['training_wall_seconds']
    original_updates=parent_result['endpoints']['affine']['updates']
    maximum_extra=7*original_updates
    # Only extra training data is new; final references stay common to the pair.
    data=prepare(specification,seed+50000,maximum_extra,output)
    with tf.device('/CPU:0'):
        for label in ('validation','reference'):
            path=parent/(label+'.tensor')
            provenance=json.loads((parent/'data-provenance.json').read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest()!=provenance['banks'][label]['sha256']:
                raise ValueError('equal-compute reference hash mismatch')
            data[label]=tf.io.parse_tensor(tf.io.read_file(str(path)),out_type=F64)
    write(output/'assessment-data-provenance.json',dict(
        source=str(parent),banks={label:provenance['banks'][label] for label in ('validation','reference')},
        unused_new_assessment_banks='prepare saved independent banks; assessment deliberately uses the parent paired banks'))
    records=[]
    offset=0
    used=original_time
    while used<required_time and offset<maximum_extra:
        block=min(2048,maximum_extra-offset)
        record=train_block(trainer,data['train'],fresh=True,updates=block,
            seed=[seed+50000,23004],counter=offset)
        records.append(record)
        write(output/'continuation-history.json',records)
        if not record['finite'] or record['updates']!=block:
            raise ValueError('equal-compute affine continuation invalid')
        used+=record['wall_seconds']
        offset+=block
    endpoint=freeze_and_assess(output,'affine-extended',trainer,evaluator,data['validation'],seed)
    probe=json.loads((output/'affine-extended-post-training-1000.json').read_text())
    if not probe['complete'] or not probe['finite'] or probe['valid_rows']!=1000:
        raise ValueError('equal-compute numerical probe invalid')
    logq=density_rows(trainer.transport,data['reference'])
    with tf.device('/CPU:0'):
        nonlinear=tf.io.parse_tensor(tf.io.read_file(str(parent/'nonlinear-reference-logq.tensor')),out_type=F64)
        logp=evaluator.target.log_prob_kernel(data['reference'])
    endpoint['forward_kl_reference']=mean_se(logp-logq)
    difference=nonlinear-logq
    save_tensor(output/'paired-difference.tensor',difference)
    save_tensor(output/'affine-extended-reference-logq.tensor',logq)
    result=dict(status='equal_compute_complete',seed=seed,target_signature=evaluator.target.signature,
        parent=str(parent),endpoint=endpoint,extra_updates=offset,total_updates=original_updates+offset,
        affine_total_training_seconds=used,nonlinear_training_seconds=required_time,
        achieved_time_ratio=used/required_time,time_target_reached=used>=required_time,
        paired_kl_benefit=mean_se(difference),scientific_promotion=False,downstream_hmc='not_run',
        stopped='time_target_reached' if used>=required_time else 'update_cap',
        parent_optimizer_preserved=True,extra_data_seed=seed+50000,
        reference_source=str(parent/'data-provenance.json'))
    write(output/'result.json',result)
    return result
