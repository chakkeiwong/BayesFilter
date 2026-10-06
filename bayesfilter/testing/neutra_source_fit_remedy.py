"""FP64 diagnostic fitting study; shared canonical IAF and joint Adam authority.

Exact sampling is confined to the CPU reference preparation. This module does
not establish posterior correctness or change the NeuTra runtime default.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference.neutra_joint_training import JointNeuTraTrainer
from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraOptimizerConfig, NeuTraTransportConfig
from bayesfilter.testing.neutra_generic_targets import ExactTargetEvaluator
from bayesfilter.testing.neutra_scientific_campaign import (
    flow_config, freeze_and_assess, serializable, save_tensor, write, evaluate_student,
)

F64 = tf.float64
OBJECTIVES = {'forward': (1., 0.), 'rkl': (0., 1.), 'joint': (1., 1.)}


def make_trainer(target, seed, profile, objective='forward', parameters=None):
    config = (NeuTraTransportConfig.huang_dsf(target.parameter_dim,
        hidden_layers=(profile['width'],profile['width']),stages=3,mixture_components=4,seed=(seed,731))
        if profile.get('kind')=='naf_dsf' else flow_config(target.parameter_dim,seed,profile['width'],kind='iaf'))
    flow = NeuTraTransport(config)
    if parameters is not None:
        flow.restore_parameters(parameters)
    # Preserve the existing large clip as a tested hypothesis, never silently
    # remove a numerical protection while changing the objective/data mechanism.
    config = NeuTraOptimizerConfig(profile['batch'], 'standard', profile['learning_rate'],
        .9, .999, 1e-8, 1000., profile.get('jit_compile', True))
    fw, rw = OBJECTIVES[objective]
    return JointNeuTraTrainer(flow, target.value_score, config,
        target_signature=target.signature, teacher_id='exact-reference-study',
        forward_weight=fw, reverse_weight=rw)


def branch_trainer(parent, target, seed, profile, objective):
    """All branches copy identical map and reset identical Adam state explicitly."""
    return make_trainer(target, seed, profile, objective,
                        parameters=parent.checkpoint()['base']['parameters'])


def train_block(trainer, data, *, fresh, updates, seed, counter=0):
    batch, dimension = trainer.config.batch_size, trainer.transport.parameter_dim
    if fresh and len(data) < (counter+updates)*batch:
        raise ValueError('fresh data exhausted; reuse is forbidden')

    def run(rows, key, count, offset):
        def body(i, loss_sum, gradient_max, clips, valid):
            key_i = tf.random.experimental.stateless_fold_in(key, i+offset)
            if fresh:
                x = tf.slice(rows, [(i+offset)*batch, 0], [batch, dimension])
            else:
                indices = tf.random.stateless_uniform([batch], key_i, 0,
                    tf.shape(rows)[0], dtype=tf.int32)
                x = tf.gather(rows, indices)
            z = tf.random.stateless_normal([batch, dimension],
                tf.random.experimental.stateless_fold_in(key_i, 1), dtype=F64)
            result = trainer._joint_step(z, x, tf.zeros([batch], F64))
            clipped = result['gradient_norm'] > result['clipped_gradient_norm']*(1.+1e-12)
            return (i+1, loss_sum+result['loss'], tf.maximum(gradient_max,result['gradient_norm']),
                    clips+tf.cast(clipped,tf.int32), valid & result['valid'])
        return tf.while_loop(lambda i, loss, grad, clips, valid: (i<count)&valid,
            body, (tf.constant(0),tf.constant(0.,F64),tf.constant(0.,F64),
                   tf.constant(0),tf.constant(True)))

    key = (fresh, tuple(data.shape))
    cache = getattr(trainer, '_remedy_blocks', {})
    if key not in cache:
        cache[key] = tf.function(run, input_signature=[tf.TensorSpec(data.shape,F64),
            tf.TensorSpec([2],tf.int32),tf.TensorSpec([],tf.int32),tf.TensorSpec([],tf.int32)],
            jit_compile=trainer.config.jit_compile,autograph=False)
        trainer._remedy_blocks = cache
    started = time.monotonic()
    values = cache[key](data,tf.constant(seed,tf.int32),tf.constant(updates),tf.constant(counter))
    return serializable({'updates':values[0], 'mean_loss':values[1]/tf.cast(values[0],F64),
        'max_gradient_norm':values[2], 'clipped_updates':values[3], 'finite':values[4],
        'wall_seconds':time.monotonic()-started,'device':values[1].device,
        'fresh_data':fresh,'data_counter_next':counter+values[0],
        'adam_iteration':trainer.optimizer.iterations, 'seed':seed,
        'batch_size':batch,'samplewise_loop':False,'jit_compile':trainer.config.jit_compile,
        'traces':cache[key].experimental_get_tracing_count()})


def prepare_data(specification, profile, seed, output):
    """Two CPU generation workers, disjoint stateless streams, no GPU sampling."""
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    count = (profile['updates']+profile['continuation_updates'])*profile['batch']
    tasks = [('fixed',4096,2001), ('fresh',count,2002), ('heldout',32768,2003)]
    def generate(task):
        label,n,stream = task
        with tf.device('/CPU:0'):
            evaluator = ExactTargetEvaluator(specification, jit_compile=True)
            sample = tf.function(lambda key:evaluator.sample(n,key),
                input_signature=[tf.TensorSpec([2],tf.int32)],jit_compile=True,autograph=False)
            value = sample(tf.constant([seed,stream],tf.int32))
            digest = save_tensor(output/(label+'.tensor'),value)
            return label,value,{'rows':n,'seed':[seed,stream],'sha256':digest,'device':value.device}
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(generate,tasks))
    write(output/'data-provenance.json',{'cpu_workers':2,'worker_kind':'threads_with_native_TensorFlow_kernels',
        'target_signature':hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest(),
        'banks':{label:detail for label,_,detail in rows}, 'learner_oracle_parameters':False})
    return {label:value for label,value,_ in rows}


def run_matched_fit(specification, profile, seed, output):
    output = Path(output)
    data = prepare_data(specification,profile,seed,output)
    evaluator = ExactTargetEvaluator(specification,jit_compile=True)
    endpoints, parents, initial_parameters = {}, {}, {}
    for mode in ('fixed','fresh'):
        trainer = make_trainer(evaluator.target,seed,profile)
        initial = trainer.checkpoint()
        initial_parameters[mode] = initial['base']['parameters']
        write(output/(mode+'-initial.json'),initial)
        history = []
        counter = 0
        for rung in (profile['updates']//2,profile['updates']):
            record = train_block(trainer,data[mode],fresh=mode=='fresh',updates=rung-counter,
                                 seed=[seed,3101],counter=counter)
            history.append(record)
            counter = rung
            write(output/(mode+'-history.json'),history)
            if not record['finite']:
                raise ValueError('nonfinite shared training computation')
        parents[mode] = trainer
        endpoints[mode] = freeze_and_assess(output,mode,trainer,evaluator,data['heldout'],seed)
        endpoints[mode]['training'] = history

    # Use the fresh endpoint by predeclaration, never choose a parent using the
    # observed losses. All branches start from the identical parameters and Adam
    # iteration zero, with common latent/data streams and equal further updates.
    parent = parents['fresh']
    parent_hash = endpoints['fresh']['transport_hash']
    start = profile['updates']*profile['batch']
    continuation = data['fresh'][start:]
    for objective in ('forward','rkl','joint'):
        trainer = branch_trainer(parent,evaluator.target,seed,profile,objective)
        write(output/(objective+'-branch-initial.json'),trainer.checkpoint())
        record = train_block(trainer,continuation,fresh=True,updates=profile['continuation_updates'],
                             seed=[seed,3201])
        if not record['finite']:
            raise ValueError('nonfinite shared continuation computation')
        label = 'continue-'+objective
        endpoints[label] = freeze_and_assess(output,label,trainer,evaluator,data['heldout'],seed)
        endpoints[label].update(training=record,parent_transport_hash=parent_hash,
                               optimizer_policy='reset_all_branches',objective=objective)
    result = {'status':'comparison_complete','target_signature':evaluator.target.signature,
        'seed':seed,'profile':profile,'endpoints':endpoints,
        'same_initial_parameters':initial_parameters['fixed'] == initial_parameters['fresh'],
        'scientific_promotion':False,'downstream_hmc':'not_run',
        'statistical_ranking':'not_established','dtype':'float64_diagnostic_reference_exception'}
    write(output/'result.json',result)
    return result


def run_optimizer_repair(specification, profile, seed, output):
    """Matched LR continuation from the prespecified fresh forward endpoint."""
    output=Path(output)
    parent_path=Path(profile['parent'])
    parent=json.loads((parent_path/'fresh-checkpoint.json').read_text())
    parent_result=json.loads((parent_path/'result.json').read_text())
    evaluator=ExactTargetEvaluator(specification,jit_compile=True)
    if parent['base']['target_signature']!=evaluator.target.signature:
        raise ValueError('parent target mismatch')
    if parent_result['seed']!=seed or not parent_result['same_initial_parameters']:
        raise ValueError('parent pairing invalid')
    data=prepare_data(specification,{**profile,'continuation_updates':0},seed+20000,output)
    endpoints={}
    for rate,label in ((.001,'lr-original'),(.0003,'lr-lower')):
        effective={**profile,'learning_rate':rate}
        trainer=make_trainer(evaluator.target,seed,effective,
                             parameters=parent['base']['parameters'])
        write(output/(label+'-initial.json'),trainer.checkpoint())
        history=[]
        counter=0
        for rung in (profile['updates']//2,profile['updates']):
            record=train_block(trainer,data['fresh'],fresh=True,updates=rung-counter,
                               seed=[seed+20000,4101],counter=counter)
            counter=rung
            record['heldout']=evaluate_student(evaluator,trainer.transport,data['heldout'],
                                               seed=seed+20000)
            history.append(record)
            write(output/(label+'-history.json'),history)
            if not record['finite']:
                raise ValueError('nonfinite optimizer repair')
        endpoints[label]=freeze_and_assess(output,label,trainer,evaluator,data['heldout'],seed+20000)
        endpoints[label].update(training=history,learning_rate=rate,
            optimizer_policy='reset_all_branches',parent_transport_hash=parent_result['endpoints']['fresh']['transport_hash'])
    result={'status':'comparison_complete','target_signature':evaluator.target.signature,
        'seed':seed,'profile':profile,'endpoints':endpoints,'parent':str(parent_path),
        'scientific_promotion':False,'downstream_hmc':'not_run','statistical_ranking':'not_established',
        'dtype':'float64_diagnostic_reference_exception'}
    write(output/'result.json',result)
    return result


def run_representation(specification,profile,seed,output):
    output=Path(output)
    evaluator=ExactTargetEvaluator(specification,jit_compile=True)
    pricing=profile.get('pricing_only',False)
    def reuse(directory,label):
        directory=Path(directory)
        provenance=json.loads((directory/'data-provenance.json').read_text())
        rows={}
        for name in ('fresh','heldout'):
            path=directory/(name+'.tensor')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=provenance['banks'][name]['sha256']:
                raise ValueError('matched data hash changed')
            with tf.device('/CPU:0'):
                rows[name]=tf.io.parse_tensor(tf.io.read_file(str(path)),out_type=F64)
        write(output/(label+'-reused-data.json'),{'source':str(directory),'provenance':provenance})
        return rows
    data=(prepare_data(specification,{**profile,'continuation_updates':0},seed,output)
          if pricing else reuse(profile['parent'],'initial'))
    effective={**profile,'kind':'naf_dsf','learning_rate':.001}
    trainer=make_trainer(evaluator.target,seed,effective)
    write(output/'initial-checkpoint.json',trainer.checkpoint())
    parameter_count=sum(int(tf.size(v)) for v in trainer.variables)
    # Actual configured inverse before optimizer work, including non-central input.
    z=tf.constant([[-4.,1.],[-1.,.2],[0.,0.],[1.,-.3],[4.,-1.]],F64)
    x,ld=trainer.transport.forward_and_logdet(z)
    recovered,recovered_ld=trainer.transport.inverse_and_forward_logdet(x)
    roundtrip=float(tf.reduce_max(tf.abs(recovered-z)))
    logdet_error=float(tf.reduce_max(tf.abs(recovered_ld-ld)))
    if not roundtrip<1e-8 or not logdet_error<1e-8:
        raise ValueError('configured NAF inverse control failed')
    history=[]
    previous=0
    rungs=[512,1024] if pricing else [4096,8192]
    for rung in rungs:
        record=train_block(trainer,data['fresh'],fresh=True,updates=rung-previous,
                           seed=[seed,3101],counter=previous)
        history.append(record)
        previous=rung
        write(output/'forward-history.json',history)
        if not record['finite']:
            raise ValueError('NAF forward computation invalid')
    endpoints={'forward':freeze_and_assess(output,'forward',trainer,evaluator,data['heldout'],seed)}
    if not pricing:
        continuation=reuse(profile['optimizer_parent'],'continuation')
        lower=make_trainer(evaluator.target,seed,{**effective,'learning_rate':.0003},
            parameters=trainer.checkpoint()['base']['parameters'])
        write(output/'lower-initial-checkpoint.json',lower.checkpoint())
        counter=0
        for rung in (4096,8192):
            record=train_block(lower,continuation['fresh'],fresh=True,updates=rung-counter,
                seed=[seed+20000,4101],counter=counter)
            counter=rung
            history.append(record)
            write(output/'all-history.json',history)
            if not record['finite']:
                raise ValueError('NAF lower-LR computation invalid')
        endpoints['lr-lower']=freeze_and_assess(output,'lr-lower',lower,evaluator,continuation['heldout'],seed+20000)
    result=dict(status='pricing_complete' if pricing else 'comparison_complete',
        target_signature=evaluator.target.signature,seed=seed,profile=profile,endpoints=endpoints,
        training=history,parameter_count=parameter_count,roundtrip_error=roundtrip,logdet_error=logdet_error,
        scientific_promotion=False,posterior_confirmation='not_run',statistical_ranking='not_established')
    write(output/'result.json',result)
    return result


def diagnose_representation_failure(specification, profile, seed, output):
    """Debug-only exact continuation replay; preserve the rejected batch/state."""
    from bayesfilter.inference import neutra_transport_core as core
    output=Path(output)
    failed=Path(profile['failed_output'])
    checkpoint=json.loads((failed/'lower-initial-checkpoint.json').read_text())
    source=Path(profile['optimizer_parent'])
    provenance=json.loads((source/'data-provenance.json').read_text())
    data_path=source/'fresh.tensor'
    if hashlib.sha256(data_path.read_bytes()).hexdigest()!=provenance['banks']['fresh']['sha256']:
        raise ValueError('replay training data changed')
    with tf.device('/CPU:0'):
        data=tf.io.parse_tensor(tf.io.read_file(str(data_path)),out_type=F64)
    evaluator=ExactTargetEvaluator(specification,jit_compile=True)
    trainer=make_trainer(evaluator.target,seed,{**profile,'learning_rate':.0003})
    trainer.restore(checkpoint)
    replay=train_block(trainer,data,fresh=True,updates=4096,seed=[seed+20000,4101])
    write(output/'replay.json',replay)
    write(output/'last-accepted-checkpoint.json',trainer.checkpoint())
    index=int(replay['updates'])-1
    batch=trainer.config.batch_size
    x=data[index*batch:(index+1)*batch]
    key=tf.random.experimental.stateless_fold_in(tf.constant([seed+20000,4101],tf.int32),index)
    z=tf.random.stateless_normal([batch,evaluator.dimension],
        tf.random.experimental.stateless_fold_in(key,1),dtype=F64)
    save_tensor(output/'rejected-physical.tensor',x)
    save_tensor(output/'rejected-latent.tensor',z)
    evaluated=trainer.evaluate_joint(z,x,tf.zeros([batch],F64))
    details={'valid':evaluated['valid'],'loss':evaluated['loss'],
        'nonfinite_gradients':[i for i,g in enumerate(evaluated['gradients'])
            if not bool(tf.reduce_all(tf.math.is_finite(g)))],
        'parameters_finite':all(bool(tf.reduce_all(tf.math.is_finite(v))) for v in trainer.variables)}
    write(output/'objective.json',details)

    def inspect(physical):
        values=physical
        report={}
        for number,layer in reversed(tuple(enumerate(trainer.transport.components))):
            if not hasattr(layer,'pseudo_parameters'):
                values=layer.inverse(values)
                continue
            solved_rows=tf.zeros_like(values)
            for coordinate in range(evaluator.dimension):
                slopes,offsets,logits=layer.pseudo_parameters(solved_rows)
                a,b,w=slopes[:,coordinate],offsets[:,coordinate],logits[:,coordinate]
                y=values[:,coordinate]
                solved,ld,valid,iterations=core.sigmoid_inverse(y,a,b,w,
                    atol=layer.config.inverse_atol,rtol=layer.config.inverse_rtol,
                    max_iterations=layer.config.inverse_max_iterations)
                recovered,_,_=core.sigmoid_mixture(solved,a,b,w)
                report[f'layer{number}_coordinate{coordinate}']={
                    'valid':valid,'iterations':iterations,'output':y,'inverse':solved,
                    'residual':recovered-y,'log_derivative':ld,
                    'slopes':tf.exp(a),'offsets':b,'weights':tf.nn.softmax(w)}
                solved_rows+=(solved-solved_rows[:,coordinate])[:,None]*tf.one_hot(coordinate,evaluator.dimension,dtype=F64)
            values=solved_rows
        report['latent']=values
        return report
    compiled=tf.function(inspect,input_signature=[tf.TensorSpec(x.shape,F64)],jit_compile=True,autograph=False)
    inverse=serializable(compiled(x))
    write(output/'inverse-detail.json',inverse)
    summary={name:{'invalid_rows':[i for i,ok in enumerate(value['valid']) if not ok],
        'iterations':value['iterations'],'maximum_absolute_residual':max(abs(v) for v in value['residual'] if v is not None)}
        for name,value in inverse.items() if isinstance(value,dict)}
    result={'status':'debug_replay_complete','target_signature':evaluator.target.signature,
        'replay':replay,'objective':serializable(details),'inverse_summary':summary,
        'scientific_promotion':False,'comparison_role':'numerical_localization_only'}
    write(output/'result.json',result)
    return result
