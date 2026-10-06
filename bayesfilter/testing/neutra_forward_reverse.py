"""Diagnostic NAF training from approximate teachers, then exact-target RKL.

CPU preparation owns mode search, native sampling and exact evaluator samples.
The GPU worker consumes only saved approximate training banks. Exact references
are used for assessment, never for gradient updates. No sampler promotion.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference.neutra_joint_training import JointNeuTraTrainer
from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraOptimizerConfig
from bayesfilter.inference.neutra_warm_start_tf import AnnealedSMC, SMCConfig, discover_modes
from bayesfilter.testing.neutra_generic_targets import ExactTargetEvaluator
from bayesfilter.testing.neutra_scientific_campaign import (
    flow_config, freeze_and_assess, teacher_screen, save_tensor, serializable, write,
)
from bayesfilter.testing.neutra_warm_start_closure import LaplaceMixtureProposal
from bayesfilter.testing.neutra_warm_start_targets_tf import BroadStudentProposal
from bayesfilter.testing.neutra_scientific_design import FORWARD_REVERSE_POLICY
from bayesfilter.testing.neutra_forward_reverse_criteria import assess_pair_criteria

F64 = tf.float64


def prepare_teacher(specification, profile, seed, output, *, method='smc'):
    """Independent native populations conditional on one discovered proposal."""
    if method not in ('smc', 'ais'):
        raise ValueError('this complete native route supports SMC and AIS only')
    output = Path(output)
    if tf.config.get_visible_devices('GPU'):
        raise RuntimeError('native teacher preparation requires GPUs intentionally hidden')
    jit = profile.get('jit_compile', True)
    evaluator = ExactTargetEvaluator(specification, jit_compile=jit)
    target = evaluator.target
    starts = BroadStudentProposal(target.parameter_dim).sample(
        profile['mode_starts'], tf.constant([seed,31001],tf.int32))
    modes = discover_modes(target, starts, max_iterations=profile['mode_iterations'],
        score_tolerance=1e-8, merge_distance=1e-6, curvature_tolerance=1e-7,
        jit_compile=False)
    write(output/'discovery.json', modes)
    base = dict(policy=FORWARD_REVERSE_POLICY, target_signature=target.signature,
        method=method, profile=profile, seed=seed, scientific_promotion=False,
        cpu_only=True, cpu_workers=2, reference_training_leakage=False,
        discovery_exhaustive=False, mode_optimizer_jit_exception='bounded CPU preparation')
    if modes['mode_count']==0:
        return {**base,'status':'teacher_failed','reason':'no_stationary_positive_curvature_modes'}
    proposal = LaplaceMixtureProposal(target, modes['representatives'])
    write(output/'proposal.json',dict(centers=proposal.centers,covariance=proposal.covariance,
        weights=proposal.weights,defensive_weight=.1,
        provenance='local_laplace_mass_approximation_corrected_by_native_importance_weights'))
    n, reps = profile['particles'], profile['replications']
    if reps < 2 or n < 2:
        raise ValueError('independent populations and nontrivial batch required')

    cfg = SMCConfig(n, profile['mutation_steps'], profile['stages'], .8, .5,
        profile['step_size'], jit_compile=jit,
        temperature_schedule=tuple(i/profile['stages'] for i in range(profile['stages']+1)),
        use_resampling=method=='smc')
    sampler = AnnealedSMC(target, proposal, cfg)
    # The fixed program has no mutable state; inputs own all particles/RNG.
    # Finish shared derivative graph construction before concurrent execution.
    sampler.fixed_program.get_concrete_function()

    def population(index):
        key = tf.constant([seed,32000+index],tf.int32)
        raw = proposal.sample(n,tf.random.experimental.stateless_fold_in(key,900))
        raw_weights = target.log_prob(raw)-proposal.log_prob(raw)
        result = sampler.run(key)
        if not result['complete']:
            raise ValueError('native population did not reach beta one')
        return (result['particles'],result['log_weights']), (raw,raw_weights), serializable({
            k:v for k,v in result.items() if k not in ('particles','log_weights','roots')})

    with ThreadPoolExecutor(max_workers=2) as pool:
        populations = list(pool.map(population, range(2*reps)))
    banks = [row[0] for row in populations]
    references = {}
    hashes = {}
    # Exact draws stay in the evaluator partition. They never enter bank files.
    for label,salt in (('teacher',33001),('map',33002)):
        references[label]=evaluator.sample(profile.get('reference_rows',32768),
            tf.constant([seed,salt],tf.int32))
        hashes[label+'-reference.tensor']=save_tensor(output/(label+'-reference.tensor'),references[label])
    screens={label:teacher_screen(evaluator, bank, references['teacher']) for label,bank in (
        ('training',banks[:reps]),('validation',banks[reps:]))}
    raw_screen=teacher_screen(evaluator,[x[1] for x in populations[:reps]],references['teacher'])
    for label,bank in (('training',banks[:reps]),('validation',banks[reps:])):
        for i,(x,lw) in enumerate(bank):
            for suffix,value in (('points',x),('logweights',lw)):
                name=f'{label}-{i}-{suffix}.tensor'; hashes[name]=save_tensor(output/name,value)
    report={**base,'status':'teacher_passed' if all(s['passed'] for s in screens.values()) else 'teacher_failed',
        'screens':screens,'raw_proposal_importance_baseline':raw_screen,
        'population_details':[p[2] for p in populations], 'artifact_sha256':hashes,
        'population_seeds':[[seed,32000+i] for i in range(2*reps)],
        'native_graph_traced_before_cpu_threads':True,
        'independence':'complete populations conditional on the frozen discovered proposal',
        'teacher_data_kind':'approximate_weighted_native_populations',
        'target_specification':specification}
    write(output/'teacher.json',report)
    return report


def load_teacher(path, signature):
    path=Path(path)
    report=json.loads((path/'teacher.json').read_text())
    if report['status']!='teacher_passed' or report['target_signature']!=signature:
        raise ValueError('teacher not admitted for this target')
    if report.get('teacher_data_kind')!='approximate_weighted_native_populations':
        raise ValueError('exact reference is not a native teacher')
    for name,digest in report['artifact_sha256'].items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('teacher artifact changed: '+name)
    def tensor(name):
        with tf.device('/CPU:0'):
            return tf.io.parse_tensor(tf.io.read_file(str(path/name)),out_type=F64)
    banks={}
    reps=report['profile']['replications']
    for label in ('training','validation'):
        points=[tensor(f'{label}-{i}-points.tensor') for i in range(reps)]
        weights=[tensor(f'{label}-{i}-logweights.tensor') for i in range(reps)]
        banks[label]=(tf.concat(points,0),tf.concat([tf.nn.log_softmax(w)-math.log(reps) for w in weights],0))
    return banks,tensor('map-reference.tensor'),report


def trainer_for(target, seed, profile, rate, *, parameters=None, reverse=False):
    flow=NeuTraTransport(flow_config(target.parameter_dim,seed,profile['width']))
    if parameters is not None: flow.restore_parameters(parameters)
    opt=NeuTraOptimizerConfig(profile['batch'],'standard',rate,.9,.999,1e-8,
        profile.get('gradient_clip',1000.),profile.get('jit_compile',True))
    return JointNeuTraTrainer(flow,target.value_score,opt,target_signature=target.signature,
        teacher_id='native_weighted_forward_reverse',forward_weight=0. if reverse else 1.,
        reverse_weight=1. if reverse else 0.)


def train_block(trainer, bank, count, seed, offset=0):
    """Categorical sampling gives the exact empirical weighted FKL gradient."""
    points,weights=bank
    batch,d=trainer.config.batch_size,trainer.transport.parameter_dim
    if batch<=1: raise ValueError('training must be batched')
    cache=getattr(trainer,'_native_blocks',{})
    cache_key=(tuple(points.shape),tuple(weights.shape))
    if cache_key not in cache:
        def program(x,lw,key,n,start):
            def body(i,loss,maximum,clips,valid):
                subkey=tf.random.experimental.stateless_fold_in(key,start+i)
                z=tf.random.stateless_normal([batch,d],subkey,dtype=F64)
                indices=tf.random.stateless_categorical(lw[None],batch,
                    tf.random.experimental.stateless_fold_in(subkey,1))[0]
                step=trainer._joint_step(z,tf.gather(x,indices),tf.zeros([batch],F64))
                return i+1,loss+step['loss'],tf.maximum(maximum,step['gradient_norm']),clips+tf.cast(
                    step['gradient_norm']>step['clipped_gradient_norm']*(1+1e-12),tf.int32),valid&step['valid']
            return tf.while_loop(lambda i,loss,maximum,clips,valid:(i<n)&valid,body,
                (tf.constant(0),tf.constant(0.,F64),tf.constant(0.,F64),tf.constant(0),tf.constant(True)))
        cache[cache_key]=tf.function(program,input_signature=[tf.TensorSpec(points.shape,F64),
            tf.TensorSpec(weights.shape,F64),tf.TensorSpec([2],tf.int32),tf.TensorSpec([],tf.int32),
            tf.TensorSpec([],tf.int32)],jit_compile=trainer.config.jit_compile,autograph=False)
        trainer._native_blocks=cache
    start=time.monotonic()
    values=cache[cache_key](points,weights,tf.constant(seed,tf.int32),tf.constant(count),tf.constant(offset))
    row=serializable(dict(updates=values[0],mean_loss=values[1]/tf.cast(values[0],F64),
        max_gradient_norm=values[2],clipped_updates=values[3],finite=values[4],
        batch_size=batch,wall_seconds=time.monotonic()-start,device=values[1].device,
        jit_compile=trainer.config.jit_compile,samplewise_loop=False,next_counter=offset+values[0]))
    if not row['finite'] or row['updates']!=count:
        raise ValueError('invalid numerical training block')
    return row


def teacher_loss(flow, bank, *, jit_compile=True):
    x,lw=bank
    fn=tf.function(lambda y:flow.log_prob(y),input_signature=[tf.TensorSpec(x.shape,F64)],
                   jit_compile=jit_compile,autograph=False)
    return float(-tf.reduce_sum(tf.nn.softmax(lw)*fn(x)))


def run_fit(specification,profile,seed,output):
    output=Path(output)
    evaluator=ExactTargetEvaluator(specification,jit_compile=profile.get('jit_compile',True))
    banks,reference,teacher=load_teacher(profile['teacher_output'],evaluator.target.signature)
    history=[]; counter=0; parameters=None
    for stage,(updates,rate) in enumerate(zip(profile['forward_updates'],profile['forward_rates'],strict=True)):
        trainer=trainer_for(evaluator.target,seed,profile,rate,parameters=parameters)
        done=0
        while done<updates:
            count=min(2048,updates-done)
            row=train_block(trainer,banks['training'],count,[seed,34001],counter)
            history.append({**row,'phase':'forward','stage':stage,'rate':rate})
            done+=count; counter+=count
            write(output/'training-history.json',history)
            if row['clipped_updates']>count/2:
                return dict(status='fit_failed',reason='majority_clipping',history=history,scientific_promotion=False)
        parameters=trainer.transport.parameter_state()
    forward=freeze_and_assess(output,'forward',trainer,evaluator,reference,seed,
        jit_compile=profile.get('jit_compile',True))
    forward['teacher_cross_entropy']={name:teacher_loss(trainer.transport,bank,
        jit_compile=profile.get('jit_compile',True)) for name,bank in banks.items()}
    parent=trainer.transport.parameter_state()
    parent_hash=forward['transport_hash']
    branches=[]
    # Every rate starts from identical FKL parameters and new Adam state.
    for rate in profile['rkl_rates']:
        trainer=trainer_for(evaluator.target,seed,profile,rate,parameters=parent,reverse=True)
        done=0; branch_valid=True
        for rung in profile['rkl_rungs']:
            row=train_block(trainer,banks['training'],rung-done,[seed,34002],done)
            history.append({**row,'phase':'reverse','rate':rate,'rung':rung})
            write(output/'training-history.json',history)
            label=f'reverse-{rate}-{rung}'
            endpoint=freeze_and_assess(output,label,trainer,evaluator,reference,seed,
                jit_compile=profile.get('jit_compile',True))
            branch_valid &= row['clipped_updates']<=row['updates']/2
            branches.append(dict(rate=rate,updates=rung,endpoint=endpoint,parent_hash=parent_hash,
                passed=forward['passed'] and endpoint['passed'] and branch_valid))
            done=rung
    result=dict(policy=FORWARD_REVERSE_POLICY,status='fit_complete',teacher_output=profile['teacher_output'],
        teacher_kind=teacher['teacher_data_kind'],target_signature=evaluator.target.signature,
        seed=seed,profile=profile,forward=forward,branches=branches,training=history,
        scientific_promotion=False,scope='approximate_FKL_then_RKL_density_and_coverage_screen')
    result=assess_pair_criteria(result,len(specification.get('weights',[])))
    write(output/'result.json',result)
    return result
