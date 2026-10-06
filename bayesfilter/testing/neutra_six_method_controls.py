"""Six-method mechanics and pricing; never posterior or training promotion.

All numerical consumers call the existing shared authorities. These bounded
controls price compiled operations and expose interface/target errors before
calibration. Counts are explicit engineering hypotheses in the execution plan.
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig
from bayesfilter.inference.neutra_weighted_training import WeightedForwardKLNeuTraTrainer, WeightedNeuTraConfig
from bayesfilter.inference.neutra_warm_start_tf import AnnealedSMC, SMCConfig, GabrieProgram
from bayesfilter.inference.neutra_flow_smc_tf import FlowTransportSMC
from bayesfilter.inference.neutra_fab import FABConfig, FABTrainer
from bayesfilter.inference.neutra_post_training import PostTrainingProbe
from bayesfilter.testing.neutra_generic_targets import ExactTargetEvaluator, fixed_specification

F64 = tf.float64
METHODS = ('fab','gabrie','ais','smc','aft','craft')


def serializable(value):
    if isinstance(value,tf.Tensor): return value.numpy().tolist()
    if isinstance(value,dict): return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [serializable(v) for v in value]
    return value


def write(path, value):
    Path(path).write_text(json.dumps(serializable(value),indent=2,allow_nan=False)+'\n')


class IsotropicProposal:
    """Declared broad Gaussian proposal; scale is a calibrated hypothesis."""
    def __init__(self, dimension, scale):
        self.dimension,self.scale=dimension,tf.constant(scale,F64)
    def log_prob(self,x):
        return -.5*tf.reduce_sum((x/self.scale)**2,1)-self.dimension*(.5*math.log(2*math.pi)+tf.math.log(self.scale))
    def sample(self,count,seed):
        return self.scale*tf.random.stateless_normal([count,self.dimension],seed,dtype=F64)


def flow_config(dimension, seed, width=16):
    width = math.ceil(width/dimension)*dimension
    return dataclasses.replace(NeuTraTransportConfig.hoffman_author_iaf(
        dimension,conditional_scale_cap=2.,seed=(seed,731)),hidden_layers=(width,width))


def initialize_affine_fixture(flow, scale=2.):
    """Exact FAB objective fixture only; not canonical learned-map evidence."""
    for variable in flow.trainable_variables: variable.assign(tf.zeros_like(variable))
    d = flow.parameter_dim
    for stage in flow.stages:
        stage.biases[-1].assign(tf.concat((tf.fill([d],tf.constant(math.log(scale)/len(flow.stages),F64)),tf.zeros([d],F64)),0))


def affine_fab_integrability(flow, maximum_variance):
    """Exact for the zero-network affine mechanics fixture; no neural tail claim."""
    d=flow.parameter_dim
    for stage in flow.stages:
        if any(bool(tf.reduce_any(w != 0).numpy()) for w in stage.weights):
            return {'applicability':'not_checked','reason':'nonlinear_tail_argument_required'}
    # Coordinate reversals permute diagonal covariance at each stage.
    scale=tf.ones([d],F64)
    for index,stage in enumerate(flow.stages):
        scale*=tf.exp(stage.biases[-1][:d])
        if index+1<len(flow.stages): scale=tf.reverse(scale,[0])
    margin=2/tf.constant(maximum_variance,F64)-1/(scale*scale)
    return {'applicability':'eligible' if bool(tf.reduce_all(margin>0).numpy()) else 'mathematically_inapplicable',
            'minimum_precision_margin':float(tf.reduce_min(margin).numpy()),
            'scope':'affine_fixture_only','variance':(scale*scale).numpy().tolist()}


def make_forward_trainer(flow, *, jit_compile=True):
    return WeightedForwardKLNeuTraTrainer(WeightedNeuTraConfig(
        dimension=flow.parameter_dim,hidden_layers=flow.config.hidden_layers,
        stages=flow.config.stages,learning_rate=.001,beta1=.9,beta2=.999,
        epsilon=1e-8,gradient_clip_norm=1000.,jit_compile=jit_compile),transport=flow)


def forward_block(trainer, count, pool_count, batch, dimension, *, jit_compile=True):
    """Categorical-with-replacement minibatches, one compiled batched update loop."""
    def run(pool,lw,seed):
        def body(i,loss,valid):
            key=tf.random.experimental.stateless_fold_in(seed,i)
            indices=tf.random.stateless_categorical(lw[None,:],batch,key)[0]
            result=trainer._train_step_impl(tf.stop_gradient(tf.gather(pool,indices)),tf.zeros([batch],F64))
            return i+1,result[0],valid & result[-1]
        return tf.while_loop(lambda i,*_:i<count,body,(tf.constant(0),tf.constant(0.,F64),tf.constant(True)))
    return tf.function(run,input_signature=[tf.TensorSpec([pool_count,dimension],F64),
        tf.TensorSpec([pool_count],F64),tf.TensorSpec([2],tf.int32)],jit_compile=jit_compile,autograph=False)


def _timed(call):
    start=time.monotonic();value=call()
    # Materializing results synchronizes the actual compiled work.
    for tensor in tf.nest.flatten(value):
        if isinstance(tensor,tf.Tensor): tensor.numpy()
    return value,time.monotonic()-start


def run_control(method, target_name, output, seed, *, jit_compile=True):
    if method not in METHODS: raise ValueError('unknown required method')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    specification=fixed_specification(target_name)
    # This control is isotropic so FAB integrability and identity reductions are
    # exact. The correlated Gaussian remains a separate benchmark test.
    if target_name=='gaussian': specification={'kind':'gaussian','mean':[0.,0.],'covariance':[[1.,0.],[0.,1.]]}
    with tf.device('/CPU:0'):
        evaluator=ExactTargetEvaluator(specification,jit_compile=jit_compile)
        reference=evaluator.sample(4096,tf.constant([seed,1001]))
        write(output/'reference-summary.json',evaluator.summary(reference,tf.zeros([4096],F64)))
    target=evaluator.target
    write(output/'target.json',specification)
    started=time.monotonic()
    report={'method':method,'target_signature':target.signature,'target_name':target_name,
            'stage':'mechanics_and_pricing','scientific_promotion':False,'jit_compile':jit_compile,
            'dtype':'float64','sampling_quality':'not_established','downstream_hmc':'not_run',
            'claim_limit':'bounded GPU reference controls; not full calibration or canonical trained-map evidence',
            'configuration':{'particles':256,'batch':64,'stages':8,'mutation_steps':4,
                             'native_updates':16,'student_updates':32,'proposal_scale':4.,
                             'step_size':.1,'learning_rate':.001,'clip':1000.}}
    if method=='fab' and target_name!='gaussian':
        report.update(status='prerequisite_failed',reason=(
            'Native nonlinear IAF auxiliary integrability is unresolved for the mixture; '
            'the quadratic-shear target has a non-integrable auxiliary density for the bounded-scale ELU IAF.'),
            prerequisite='target_and_actual_map_tail_proof',native_training='not_run')
        report['wall_seconds']=time.monotonic()-started
        return report
    key=tf.constant([seed,4001])
    config=flow_config(target.parameter_dim,seed)
    proposal=IsotropicProposal(target.parameter_dim,4.)
    native_flow=None
    if method in ('ais','smc'):
        cfg=SMCConfig(256,4,8,.8,.5,.1,jit_compile=jit_compile,
                      temperature_schedule=tuple(i/8 for i in range(9)),use_resampling=method=='smc')
        sampler=AnnealedSMC(target,proposal,cfg)
        result,cold=_timed(lambda:sampler.run(key))
        warm_result,warm=_timed(lambda:sampler.run(tf.constant([seed,4002])))
        if not result['complete'] or not warm_result['complete']: raise ValueError('invalid frozen annealing control')
        x,lw=result['particles'],result['log_weights']
        report.update(native_cold_seconds=cold,native_warm_seconds=warm,
            native_detail={k:v for k,v in result.items() if k not in ('particles','log_weights','roots')},
            source_route='neutra_warm_start_tf.AnnealedSMC.fixed_program')
    elif method in ('aft','craft'):
        sampler=FlowTransportSMC(target,proposal,config,particles=256,stages=8,
            inner_updates=2,passes=2,learning_rate=.001,gradient_clip=1000.,
            mutation_steps=4,step_size=.1,resampling_fraction=.5,jit_compile=jit_compile)
        run=sampler.run_aft if method=='aft' else sampler.run_craft
        result,cold=_timed(lambda:run(key))
        sampler.restore(sampler.initial)
        warm_result,warm=_timed(lambda:run(tf.constant([seed,4002])))
        if not result['complete'] or not warm_result['complete']: raise ValueError('invalid flow-SMC control')
        x,lw=result['particles'],result['log_weights']
        report.update(native_cold_seconds=cold,native_warm_seconds=warm,
            native_detail={k:v for k,v in result.items() if k not in ('particles','log_weights')},
            source_route='neutra_flow_smc_tf.FlowTransportSMC.'+('run_aft' if method=='aft' else 'run_craft'),
            full_controller_equivalence='not_established_by_this_control')
    elif method=='gabrie':
        native_flow=NeuTraTransport(config)
        sampler=GabrieProgram(native_flow,target,walkers=16,steps=4,jit_compile=jit_compile)
        trainer=make_forward_trainer(native_flow,jit_compile=jit_compile)
        x=proposal.sample(16,key)
        rows=[]
        # The source retry controller is deliberately not claimed by this
        # primitive pricing path; its missing coverage stays a matrix blocker.
        for i in range(16):
            then=time.monotonic()
            x,bank,ga,la,bad=sampler.run(x,tf.constant(.1,F64),tf.constant([seed,5000+i]))
            update=trainer.train_step(bank,tf.zeros([64],F64))
            if int(bad.numpy()) or not bool(tf.math.is_finite(update.loss).numpy()): raise ValueError('invalid Gabrié primitive control')
            rows.append({'iteration':i,'seconds':time.monotonic()-then,'global_acceptance':ga,'local_acceptance':la})
        x,bank,ga,la,bad=sampler.run(x,tf.constant(.1,F64),tf.constant([seed,6000]))
        x,lw=bank,tf.zeros([64],F64)
        write(output/'native-checkpoint.json',trainer.state_payload())
        report.update(native_cold_seconds=rows[0]['seconds'],native_warm_seconds=sum(r['seconds'] for r in rows[1:])/15,
            native_detail=rows,source_route='neutra_warm_start_tf.GabrieProgram + WeightedForwardKLNeuTraTrainer',
            full_controller_equivalence='not_established; source retry and loss-jump controller still required',
            bank_role='correlated_nonstationary_debug_bank_not_qualified_teacher')
    else:
        native_flow=NeuTraTransport(config)
        initialize_affine_fixture(native_flow)
        cfg=FABConfig(64,8,3,1,.1,.001,.9,.999,1e-8,0,0,1,None,None,False,.65,1.02,
                      jit_compile=jit_compile,transition_operator='hmc')
        sampler=FABTrainer(native_flow,target.value_score,cfg,target_signature=target.signature,seed=(seed,5001))
        rows=[]
        for i in range(16):
            applicability=affine_fab_integrability(native_flow,1.)
            if applicability['applicability']!='eligible': raise ValueError('FAB auxiliary control became ineligible')
            result,elapsed=_timed(sampler.step)
            rows.append({'pass':i,'seconds':elapsed,'integrability':applicability,'valid':result['valid']})
        result,correction_seconds=_timed(lambda:sampler.step(train=False))
        x=result['x']
        lw=result['log_w']+native_flow.log_prob(x)-target.log_prob(x)
        write(output/'native-checkpoint.json',sampler.checkpoint())
        report.update(native_cold_seconds=rows[0]['seconds'],native_warm_seconds=sum(r['seconds'] for r in rows[1:])/15,
            native_detail=rows,posterior_correction_seconds=correction_seconds,
            source_route='neutra_fab.FABTrainer',initialization='zero-network affine objective fixture only',
            replay='covered by independent tests; fresh-AIS control here',
            canonical_initialization_tested=False)
    with tf.device('/CPU:0'):
        write(output/'native-summary.json',evaluator.summary(x,lw))
    tf.io.write_file(str(output/'bank.tensor'),tf.io.serialize_tensor(x))
    tf.io.write_file(str(output/'log-weights.tensor'),tf.io.serialize_tensor(lw))
    flow=NeuTraTransport(config)
    trainer=make_forward_trainer(flow,jit_compile=jit_compile)
    block=forward_block(trainer,32,int(x.shape[0]),64,target.parameter_dim,jit_compile=jit_compile)
    update,fit_cold=_timed(lambda:block(x,lw,tf.constant([seed,7001])))
    update,fit_warm=_timed(lambda:block(x,lw,tf.constant([seed,7002])))
    if not bool(update[-1].numpy()): raise ValueError('invalid student optimizer control')
    write(output/'student-checkpoint.json',trainer.state_payload())
    write(output/'student-frozen.json',flow.frozen_payload(target_signature=target.signature))
    # The post-training diagnostic validates the beta=1 target. It is a Python
    # scalar configuration value; passing a symbolic Tensor here reaches the
    # diagnostic's static beta validation and is a graph-construction error.
    probe=PostTrainingProbe(flow,target,1.0,jit_compile=jit_compile)
    probe_result,probe_seconds=_timed(lambda:probe(tf.constant([seed,8001])))
    write(output/'post-training-1000.json',probe_result)
    report.update(status='mechanics_control_completed',student_cold_seconds=fit_cold,
        student_warm_seconds=fit_warm,student_updates_per_block=32,probe_seconds=probe_seconds,
        teacher_admitted=False,student_training='debug_only_from_unqualified_bank',
        full_pipeline_seconds_lower_bound=time.monotonic()-started,
        posterior_confirmation='not_run; native calibration, teacher qualification and HMC pricing still required')
    return report
