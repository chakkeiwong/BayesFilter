"""Executable benchmark phases; no q20/default/posterior promotion authority."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import resource
import time
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.neutra_transport import (
    NeuTraTransport,NeuTraTransportConfig,NeuTraTransportTrainer,NeuTraOptimizerConfig)
from bayesfilter.inference.neutra_weighted_training import WeightedForwardKLNeuTraTrainer,WeightedNeuTraConfig
from bayesfilter.inference.neutra_post_training import PostTrainingProbe
from bayesfilter.inference.neutra_warm_start_tf import (
    GabrieProgram,AnnealedSMC,SMCConfig,MALAProgram,discover_modes,CandidateFailure)
from bayesfilter.testing.neutra_warm_start_targets_tf import (
    WarmStartTarget,BroadStudentProposal,wiggle_quadrature,F64)


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if tf.is_tensor(value) or isinstance(value,tf.Variable):return clean(value.numpy().tolist())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def save_tensor(path,value):
    tf.io.write_file(str(path),tf.io.serialize_tensor(value))


def read_tensor(path):
    return tf.io.parse_tensor(tf.io.read_file(str(path)),out_type=F64)


def prepare(target_name,output,config):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    target=WarmStartTarget(target_name,jit_compile=False)
    proposal=BroadStudentProposal(target.parameter_dim)
    ti=config['targets'].index(target_name)
    normalizer=0.; reference_check={'exact_sampler':target_name!='wiggle'}
    if target_name=='wiggle':
        estimates=[]
        for extent,resolution in config['wiggle_quadrature']:
            grid,lw,z=wiggle_quadrature(target,extent=extent,resolution=resolution)
            weights=tf.exp(lw)
            mean=tf.reduce_sum(weights[:,None]*grid,axis=0)
            second=tf.reduce_sum(weights[:,None]*grid**2,axis=0)
            estimates.append({'extent':extent,'resolution':resolution,'log_z':z,'mean':mean,'second':second})
        last,previous=estimates[-1],estimates[-2]
        relative=tf.reduce_max(tf.abs((last['second']-previous['second'])/(1+tf.abs(last['second']))))
        passed=bool((tf.abs(last['log_z']-previous['log_z'])<config['quadrature_tolerance']).numpy()) and \
               bool((relative<config['quadrature_tolerance']).numpy())
        reference_check={'exact_sampler':False,'quadrature':estimates,'passed':passed,
                         'tail_control':'domain expansion plus radial Gaussian energy decay; finite refinement evidence'}
        normalizer=float(z.numpy())
        if not passed:
            write_json(output/'preparation.json',{'target':target.specification,'reference_check':reference_check})
            raise ValueError('wiggle quadrature reference not stable')
    for j,role in enumerate(('training','validation','confirmation')):
        count=config['reference_rows'][role]
        seed=tf.constant([701+ti,100+j],tf.int32)
        if target_name=='wiggle':
            idx=tf.random.stateless_categorical(lw[None,:],count,seed)[0]
            samples=tf.gather(grid,idx)
        else:samples=target.reference_sample(count,seed)
        save_tensor(output/(role+'.tensor'),samples)
    starts=proposal.sample(config['mode_starts'],tf.constant([811,ti],tf.int32))
    modes=discover_modes(target,starts,max_iterations=config['mode_max_iterations'],
        score_tolerance=config['mode_score_tolerance'],merge_distance=config['mode_merge_distance'],
        curvature_tolerance=config['mode_curvature_tolerance'])
    expected={'gaussian':1,'mixture':2,'warped_mixture':2}.get(target_name)
    mode_check=(modes['mode_count']==expected) if expected is not None else None
    save_tensor(output/'discovered_modes.tensor',modes['representatives'])
    confirmation=read_tensor(output/'confirmation.tensor')
    report={'target':target.specification,'signature':target.signature,'log_normalizer':normalizer,
        'reference_check':reference_check,'reference_regions':tf.reduce_mean(target.region_features(confirmation),axis=0),
        'mode_search':modes,'expected_known_mode_count':expected,'mode_count_matches_reference':mode_check,
        'cpu_only':True,'gpu_devices_intentionally_hidden':True,
        'mode_optimizer_xla':False,'mode_optimizer_exception':'small CPU reference/discovery diagnostic; no training updates'}
    write_json(output/'preparation.json',report)
    return report


def make_transport(target,width,seed,*,variance_scale=None):
    config=dataclasses.replace(NeuTraTransportConfig.hoffman_author_iaf(target.parameter_dim,
        conditional_scale_cap=2.,seed=seed),hidden_layers=(width,width))
    if variance_scale is not None:config=dataclasses.replace(config,iaf_variance_scale=variance_scale)
    return NeuTraTransport(config)


class TrainingBlock:
    """Host-free update blocks around the existing shared numerical trainers."""
    def __init__(self,flow,target,*,batch,learning_rate,clip,kind,walkers,walk_steps,jit_compile=True):
        self.flow,self.kind=flow,kind
        if kind=='rkl':
            self.trainer=NeuTraTransportTrainer(flow,target.value_score,NeuTraOptimizerConfig(
                batch,'standard',learning_rate,.9,.999,1e-8,clip,jit_compile),target_signature=target.signature)
        else:
            self.trainer=WeightedForwardKLNeuTraTrainer(WeightedNeuTraConfig(
                dimension=target.parameter_dim,hidden_layers=flow.config.hidden_layers,stages=flow.config.stages,
                learning_rate=learning_rate,beta1=.9,beta2=.999,epsilon=1e-8,
                gradient_clip_norm=clip,jit_compile=jit_compile),transport=flow)
        sampler=GabrieProgram(flow,target,walkers=walkers,steps=walk_steps,jit_compile=jit_compile) if kind=='gabrie' else None
        if sampler is not None and walkers*walk_steps!=batch:raise ValueError('Gabrié block must equal training batch')
        dim=target.parameter_dim
        # Actual rows used by the adaptive forward objective, including their
        # temporal dependence. These totals are descriptive, never iid errors.
        self.measure_names=tuple([f'x{i}' for i in range(dim)]+[f'x{i}_squared' for i in range(dim)]+
            [f'region{i}' for i in range(int(target.region_features(tf.zeros([1,dim],F64)).shape[1]))]+
            (['valley_abs_x0_lt2'] if target.name in ('mixture','warped_mixture') else []))
        self.measure_sum=tf.Variable(tf.zeros([len(self.measure_names)],F64),trainable=False)
        self.measure_count=tf.Variable(0,dtype=tf.int64,trainable=False)
        def block(seed,count,pool,lw,current,dt):
            def body(i,current,loss_total,norm_total,clipped_total,global_acc,local_acc,invalid,okay):
                key=tf.random.experimental.stateless_fold_in(seed,i)
                ga,la,bad=tf.constant(0.,F64),tf.constant(0.,F64),tf.constant(0)
                if kind=='rkl':
                    z=tf.random.stateless_normal([batch,dim],key,dtype=F64)
                    result=self.trainer.train_step(z)
                    loss,norm,clipped,valid=result['loss'],result['gradient_norm'],result['clipped_gradient_norm'],result['valid']
                else:
                    if sampler is not None:
                        current,physical,ga,la,bad=sampler.run(current,dt,key)
                        columns=[physical,physical**2,target.region_features(physical)]
                        if target.name in ('mixture','warped_mixture'):
                            columns.append(tf.cast(tf.abs(physical[:,:1])<2.,F64))
                        self.measure_sum.assign_add(tf.reduce_sum(tf.concat(columns,axis=1),axis=0))
                        self.measure_count.assign_add(tf.cast(tf.shape(physical)[0],tf.int64))
                    else:
                        indices=tf.random.stateless_categorical(lw[None,:],batch,key)[0]
                        physical=tf.gather(pool,indices)
                    # Sampling rows according to normalized cloud weights makes
                    # the equal-weight minibatch gradient unbiased for that cloud.
                    result=self.trainer._train_step_impl(tf.stop_gradient(physical),tf.zeros([batch],F64))
                    loss,norm,clipped,valid=result[0],result[4],result[5],result[-1]
                    valid=valid & tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(v)) for v in flow.trainable_variables]))
                return (i+1,current,loss_total+loss,norm_total+norm,
                    clipped_total+tf.cast(norm>clipped*(1+1e-12),F64),global_acc+ga,local_acc+la,invalid+bad,okay&valid)
            output=tf.while_loop(lambda i,*rest:(i<count)&rest[-1],body,
                (tf.constant(0),current,tf.constant(0.,F64),tf.constant(0.,F64),tf.constant(0.,F64),
                 tf.constant(0.,F64),tf.constant(0.,F64),tf.constant(0),tf.constant(True)))
            return output
        self.run=tf.function(block,input_signature=[tf.TensorSpec([2],tf.int32),tf.TensorSpec([],tf.int32),
            tf.TensorSpec([None,dim],F64),tf.TensorSpec([None],F64),tf.TensorSpec([walkers,dim],F64),
            tf.TensorSpec([],F64)],jit_compile=jit_compile,autograph=False)

    def measure_summary(self):
        return {'rows':self.measure_count.read_value(),'names':self.measure_names,
            'mean':tf.math.divide_no_nan(self.measure_sum,tf.cast(self.measure_count,F64)),
            'sum':self.measure_sum.read_value(),
            'scope':'cumulative current trainer invocation; exact detached rows fed to forward loss',
            'role':'explanatory_correlated_training_measure_not_equilibrium_proof'}

    def checkpoint(self):
        return self.trainer.checkpoint() if self.kind=='rkl' else self.trainer.state_payload()

    def restore(self,checkpoint):
        """Resume map and optimizer together; validate everything before mutation."""
        if self.kind=='rkl':return self.trainer.restore(checkpoint)
        body={k:v for k,v in checkpoint.items() if k!='state_hash'}
        digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if checkpoint.get('state_hash')!=digest:raise ValueError('weighted checkpoint hash mismatch')
        if (checkpoint.get('schema')!='bayesfilter.neutra.weighted_forward_kl_state.v1' or
            checkpoint.get('config')!=self.trainer.config.manifest_payload() or
            checkpoint.get('transport_config')!=self.flow.config.manifest_payload()):
            raise ValueError('weighted checkpoint configuration mismatch')
        pending=[]
        for variables,name in ((self.trainer.variables,'variables'),
                               (self.trainer.optimizer.variables,'optimizer_variables')):
            saved=checkpoint[name]
            if len(saved)!=len(variables):raise ValueError('weighted checkpoint variable count mismatch')
            for variable,raw in zip(variables,saved,strict=True):
                value=tf.convert_to_tensor(raw,dtype=variable.dtype)
                if value.shape!=variable.shape:raise ValueError('weighted checkpoint variable shape mismatch')
                if tf.as_dtype(value.dtype).is_floating and not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
                    raise ValueError('nonfinite weighted checkpoint')
                pending.append((variable,value))
        if checkpoint['step']!=checkpoint['optimizer_variables'][0]:
            raise ValueError('weighted checkpoint optimizer iteration mismatch')
        for variable,value in pending:variable.assign(value)
        self.trainer.step.assign(checkpoint['step'])


def gradient_calibration(flow,pool,config,seed,*,target=None,kind='forward',learning_rate=.001,log_weights=None):
    batch=config['batch_size']
    @tf.function(input_signature=[tf.TensorSpec([batch,flow.parameter_dim],F64)],jit_compile=True,autograph=False)
    def measure(x):
        with tf.GradientTape() as tape:
            if kind=='rkl':
                physical,ld=flow.forward_and_logdet(x)
                loss=tf.reduce_mean(-target.log_prob(physical)-ld)
            else:loss=-tf.reduce_mean(flow.log_prob(x))
        gradients=tuple(tape.gradient(loss,flow.trainable_variables))
        return tf.linalg.global_norm(gradients),gradients
    norms=[]; gradients=[]
    for j in range(config['gradient_pilot_batches']):
        if kind=='rkl':
            x=tf.random.stateless_normal([batch,flow.parameter_dim],tf.constant([seed,j]),dtype=F64)
        else:
            if log_weights is None:
                idx=tf.random.stateless_uniform([batch],tf.constant([seed,j],tf.int32),
                    minval=0,maxval=tf.shape(pool)[0],dtype=tf.int32)
            else:
                idx=tf.random.stateless_categorical(log_weights[None,:],batch,tf.constant([seed,j]))[0]
            x=tf.gather(pool,idx)
        norm,grad=measure(x);norms.append(norm);gradients.append(grad)
    values=tf.stack(norms)
    clip=float((config['clip_pilot_multiplier']*tf.reduce_max(values)).numpy())
    if not math.isfinite(clip) or clip<=0:raise ValueError('invalid gradient pilot')
    # Compare actual Adam steps with persistent moments on identical recorded
    # gradient sequences. These scratch variables cannot change the learned map.
    copies=[[tf.Variable(v,trainable=True) for v in flow.trainable_variables] for _ in range(2)]
    optimizers=[tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=.9,beta_2=.999,epsilon=1e-8) for _ in range(2)]
    for opt,vs in zip(optimizers,copies):opt.build(vs)
    def update(*grad):
        deltas=[]
        for index,(opt,vs) in enumerate(zip(optimizers,copies)):
            before=[tf.identity(v) for v in vs]
            used=tf.clip_by_global_norm(grad,clip)[0] if index else grad
            opt.apply_gradients(zip(used,vs))
            deltas.append(tf.concat([tf.reshape(v-b,[-1]) for v,b in zip(vs,before)],0))
        a,b=deltas;na,nb=tf.linalg.norm(a),tf.linalg.norm(b)
        return tf.math.divide_no_nan(nb,na),tf.math.divide_no_nan(tf.reduce_sum(a*b),na*nb)
    update=tf.function(update,input_signature=[tf.TensorSpec(v.shape,v.dtype) for v in flow.trainable_variables],jit_compile=True,autograph=False)
    steps=[update(*grad) for grad in gradients]
    flat=tf.stack([tf.concat([tf.reshape(g,[-1]) for g in grad],0) for grad in gradients])
    noise=tf.reduce_mean(tf.reduce_sum((flat-tf.reduce_mean(flat,axis=0))**2,axis=1))
    signal=tf.reduce_sum(tf.reduce_mean(flat,axis=0)**2)
    return clip,{'objective':kind,'raw_norms':values,'clip':clip,'provenance':'pilot maximum times declared multiplier',
        'adam_clipped_to_unclipped_step_norm_ratio':[r[0] for r in steps],
        'adam_clipped_to_unclipped_step_cosine':[r[1] for r in steps],
        'batch_gradient_noise_to_squared_mean':tf.math.divide_no_nan(noise,signal),
        'optimizer_test':'identical gradient sequence, separate persistent Adam states; local calibration only',
        'pilot_clipped_fraction':float(tf.reduce_mean(tf.cast(values>clip,F64)).numpy()),
        'calibration_not_optimization':True,'long_run_sensible_not_established':True}


def assessment(flow,target,reference,normalizer,seed,*,probe=None):
    def evaluate(x):
        q=flow.log_prob(x)
        return tf.reduce_mean(q),tf.reduce_mean(target.log_prob(x)-normalizer-q),tf.reduce_all(tf.math.is_finite(q))
    cache=getattr(flow,'_warm_start_assessment',None)
    if cache is None:
        cache=tf.function(evaluate,input_signature=[tf.TensorSpec([None,target.parameter_dim],F64)],jit_compile=True,autograph=False)
        flow._warm_start_assessment=cache
    mean_log_q,fkl,finite=cache(reference)
    z=tf.random.stateless_normal([8192,target.parameter_dim],seed,dtype=F64)
    samples,_=flow.forward_and_logdet(z)
    sample_features=target.region_features(samples)
    ref_features=target.region_features(reference)
    actual,expected=tf.reduce_mean(sample_features,axis=0),tf.reduce_mean(ref_features,axis=0)
    error=actual-expected
    se=tf.sqrt(tf.math.reduce_variance(sample_features,axis=0)/8192+
               tf.math.reduce_variance(ref_features,axis=0)/tf.cast(tf.shape(reference)[0],F64))
    # 0.05 is the declared coarse warm-start margin, not posterior precision.
    coverage=bool(tf.reduce_all(tf.abs(error)<=.05+4*se).numpy())
    result={'finite_density':bool(finite.numpy()),'heldout_mean_log_q':mean_log_q,
        'heldout_forward_kl_estimate':fkl,'reference_regions':expected,'flow_regions':actual,
        'region_errors':error,'independent_draw_standard_errors':se,'coverage_screen_passed':coverage,
        'coverage_screen_margin':.05,'coverage_screen_sigma_multiplier':4.,
        'scientific_ranking_established':False,'posterior_qualified':False}
    if probe is not None:result['post_training_1000']=probe(tf.random.experimental.stateless_fold_in(seed,17))
    return result


def initial_walkers(target,config,seed,*,representatives=None):
    reps=target.known_representatives() if representatives is None else representatives
    if int(tf.shape(reps)[0])==0:raise CandidateFailure('no valid discovered representatives')
    centers=tf.gather(reps,tf.range(config['walkers'])%tf.shape(reps)[0])
    return centers+config['walker_initial_scale']*tf.random.stateless_normal(
        [config['walkers'],target.parameter_dim],seed,dtype=F64)


def gpu_preflight(target,output):
    """Bounded graph/XLA equivalence on deterministic inputs, not fit evidence."""
    reference=WarmStartTarget(target.name,jit_compile=False)
    x=tf.reshape(tf.linspace(tf.constant(.1,F64),tf.constant(.8,F64),4*target.parameter_dim),[4,target.parameter_dim])
    start=time.monotonic();actual=target.value_score(x);first=time.monotonic()-start
    expected=reference.value_score(x)
    errors=[tf.reduce_max(tf.abs(a-b)) for a,b in zip(actual[:2],expected[:2])]
    passed=all(bool((e<1e-9).numpy()) for e in errors) and bool(tf.reduce_all(actual[2]).numpy())
    start=time.monotonic();target.value_score(x)[0].numpy();steady=time.monotonic()-start
    ops={op.type for op in target.value_score.get_concrete_function().graph.get_operations()}
    passed=passed and not bool(ops & {'PyFunc','EagerPyFunc','PyFuncStateless'})
    report={'passed':passed,'target_value_score_max_abs_errors':errors,'tolerance':1e-9,
        'tolerance_provenance':'FP64 deterministic equivalence screen; not a statistical threshold',
        'first_call_seconds':first,'repeat_call_seconds':steady,'batch_native':True,
        'jit_compile':True,'dtype':'float64','xla_status':'executed value/score kernel',
        'gpu_device':actual[0].device,'callback_ops':sorted(ops & {'PyFunc','EagerPyFunc','PyFuncStateless'})}
    write_json(Path(output)/'gpu-preflight.json',report)
    if not passed or 'GPU' not in actual[0].device:raise ValueError('GPU graph/XLA preflight failed')
    return report


def calibrate(target_name,prepared,output,config):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=WarmStartTarget(target_name)
    gpu_preflight(target,output)
    prepared=Path(prepared)
    pool,validation=read_tensor(prepared/'training.tensor'),read_tensor(prepared/'validation.tensor')
    prep=json.loads((prepared/'preparation.json').read_text())
    widths=config['widths_funnel'] if target_name=='funnel' else config['widths_2d']
    records=[];best=None
    for width in widths:
        for lr in config['learning_rates']:
            flow=make_transport(target,width,(910,13))
            clip,grad=gradient_calibration(flow,pool,config,910,learning_rate=lr)
            trainer=TrainingBlock(flow,target,batch=config['batch_size'],learning_rate=lr,clip=clip,kind='forward',
                walkers=config['walkers'],walk_steps=config['walker_steps'])
            initial=initial_walkers(target,config,tf.constant([912,1]))
            start=time.monotonic()
            result=trainer.run(tf.constant([913,1]),tf.constant(config['pilot_updates']),pool,
                tf.zeros([tf.shape(pool)[0]],F64),initial,tf.constant(.01,F64))
            elapsed=time.monotonic()-start
            metric=assessment(flow,target,validation,prep['log_normalizer'],tf.constant([914,1]))
            row={'width':width,'learning_rate':lr,'clip':clip,'gradient_calibration':grad,
                'updates':int(result[0].numpy()),'seconds_including_compile':elapsed,'validation':metric}
            records.append(row)
            score=float(metric['heldout_forward_kl_estimate'].numpy())
            if bool(result[-1].numpy()) and metric['finite_density'] and math.isfinite(score):
                if best is None or score<best[0]:best=(score,row)
            write_json(output/'progress.json',{'candidates':records})
    if best is None:raise ValueError('no finite target-specific training calibration candidate')
    # MALA calibrates the physical target from declared representatives, using
    # an independent pilot and actual displacement; no retained draws are used.
    mala=MALAProgram(lambda x,beta:target.log_prob(x),target.parameter_dim,steps=4)
    x=initial_walkers(target,config,tf.constant([915,1]))
    candidates=[]
    for dt in config['mala_step_grid']:
        y,_,acc,invalid=mala.run(x,tf.constant(1.,F64),tf.constant(dt,F64),tf.constant([916,1]))
        jump=tf.reduce_mean(tf.reduce_sum((y-x)**2,axis=1))
        candidates.append({'step_size':dt,'acceptance':float(acc.numpy()),'squared_displacement':float(jump.numpy()),
                           'invalid_proposals':int(invalid.numpy())})
    from bayesfilter.testing.neutra_warm_start_policy import select_mala_candidate
    write_json(output/'kernel-calibration.json',candidates)
    chosen=select_mala_candidate(candidates)
    selected={k:best[1][k] for k in ('width','learning_rate','clip')}
    selected['mala_step_size']=chosen['step_size']
    report={'target':target_name,'selected':selected,'candidates':records,'kernel_calibration':candidates,
        'selection':'heldout forward loss nominates settings; no statistical method ranking',
        'calibration_teacher':'exact_examples' if target_name!='wiggle' else 'checked_quadrature_examples',
        'kernel_calibration_role':'bounded sensibility pilot, not mixing/convergence proof'}
    write_json(output/'calibration.json',report)
    return report


def train(target_name,arm,seed,prepared,calibration,output,config):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=WarmStartTarget(target_name)
    prep=json.loads((Path(prepared)/'preparation.json').read_text())
    cal=json.loads(Path(calibration).read_text())['selected']
    if config.get('capacity_width') is not None:
        cal={**cal,'width':config['capacity_width'],'width_role':'declared capacity candidate; no default promotion'}
    flow=make_transport(target,cal['width'],(seed,91))
    reference=read_tensor(Path(prepared)/'confirmation.tensor')
    validation=read_tensor(Path(prepared)/'validation.tensor')
    pool=read_tensor(Path(prepared)/'training.tensor')
    lw=tf.zeros([tf.shape(pool)[0]],F64)
    reps=read_tensor(Path(prepared)/'discovered_modes.tensor') if arm=='gabrie_discovered' else None
    current=initial_walkers(target,config,tf.constant([seed,5]),representatives=reps)
    if arm.startswith('gabrie'):
        pool=current;lw=tf.zeros([config['walkers']],F64)
    teacher_report=None
    restored_before=None
    warm_checkpoint=None;warm_parent=config.get('warm_parent')
    prior_warm_history=[]
    if warm_parent:
        if arm not in ('oracle','smc','waste_free','aft','craft'):
            raise ValueError('warm continuation requires a fixed saved teacher; adaptive walkers must be archived first')
        parent=Path(warm_parent);previous=json.loads((parent/'result.json').read_text())
        if (previous['target'],previous['arm'],previous['seed'])!=(target_name,arm,seed):
            raise ValueError('warm continuation parent target/arm/seed mismatch')
        prior_warm_history=[row for row in previous['history'] if row['phase']=='warm']
        if not prior_warm_history:raise ValueError('no saved warm phase to continue')
        count=prior_warm_history[-1]['total_updates']
        if config['warm_total_updates']<=count:raise ValueError('warm continuation must extend the parent')
        warm_checkpoint=json.loads((parent/f'warm-{count}-checkpoint.json').read_text())
        write_json(output/'warm-continuation.json',{'parent':str(parent),'parent_updates':count,
            'terminal_updates':config['warm_total_updates'],'optimizer_preserved':True,
            'parent_state_hash':warm_checkpoint['state_hash'],'independent_replication':False})
        if arm!='oracle':
            pool=read_tensor(parent/'teacher-particles.tensor');lw=read_tensor(parent/'teacher-log-weights.tensor')
            teacher_report=json.loads((parent/'teacher.json').read_text())
            write_json(output/'teacher.json',teacher_report)
            save_tensor(output/'teacher-particles.tensor',pool);save_tensor(output/'teacher-log-weights.tensor',lw)
            write_json(output/'teacher-reuse.json',{'parent':str(parent),'role':'same fixed teacher; no new sampler evidence'})
    if config.get('repair_rkl_collapse'):
        parent=Path(config['repair_parent'])
        frozen=json.loads((parent/'warm-frozen.json').read_text())
        if frozen['target_signature']!=target.signature:raise ValueError('RKL repair target mismatch')
        flow.restore_parameters(frozen['parameters'])
        restored_before=json.loads((parent/'warm-assessment.json').read_text())
        cal={**cal,'learning_rate':min(config['learning_rates'])}
        write_json(output/'warm-frozen.json',frozen);write_json(output/'warm-assessment.json',restored_before)
        write_json(output/'warm-reuse.json',{'parent':str(parent),'role':'exact preserved pre-RKL checkpoint; no new warm-start evidence'})
    if arm in ('smc','waste_free','aft','craft') and restored_before is None and warm_checkpoint is None:
        proposal=BroadStudentProposal(target.parameter_dim)
        if arm in ('smc','waste_free'):
            sampler=AnnealedSMC(target,proposal,SMCConfig(config['particles'],config['mutation_steps'],
                config['max_smc_stages'],config['cess_fraction'],config['resampling_fraction'],
                cal['mala_step_size'],arm=='waste_free'))
            teacher=sampler.run(tf.constant([seed,7]),callback=lambda row:write_json(output/'teacher-progress.json',row))
        else:
            from bayesfilter.inference.neutra_flow_smc_tf import FlowTransportSMC
            sampler=FlowTransportSMC(target,proposal,flow.config,particles=config['flow_particles'],
                stages=config['flow_stages'],inner_updates=config['aft_inner_updates'],passes=config['craft_passes'],
                learning_rate=cal['learning_rate'],gradient_clip=cal['clip'],mutation_steps=config['mutation_steps'],
                step_size=cal['mala_step_size'],resampling_fraction=config['resampling_fraction'])
            teacher=(sampler.run_aft if arm=='aft' else sampler.run_craft)(tf.constant([seed,7]),
                callback=lambda row:write_json(output/'teacher-progress.json',row))
        teacher_report={k:v for k,v in teacher.items() if k not in ('particles','log_weights','roots')}
        write_json(output/'teacher.json',teacher_report)
        if not teacher['complete']:raise CandidateFailure('SMC teacher failed to reach the actual target: '+teacher.get('reason','unknown'))
        pool,lw=teacher['particles'],teacher['log_weights']
        observed=tf.reduce_sum(tf.nn.softmax(lw)[:,None]*target.region_features(pool),axis=0)
        expected=tf.reduce_mean(target.region_features(validation),axis=0)
        teacher_report['region_diagnostic']={'weighted_regions':observed,'reference_regions':expected,
            'difference':observed-expected,'role':'explanatory; weighted/correlated particles are not independent replications'}
        write_json(output/'teacher.json',teacher_report)
        save_tensor(output/'teacher-particles.tensor',pool);save_tensor(output/'teacher-log-weights.tensor',lw)
    history=list(prior_warm_history)
    probe=PostTrainingProbe(flow,target,1.,rows=1000,batch_size=1000,jit_compile=True)
    before=restored_before
    for phase in (['rkl'] if arm=='rkl' or restored_before else ['warm','rkl']):
        kind='rkl' if phase=='rkl' else ('gabrie' if arm.startswith('gabrie') else 'forward')
        continuing=phase=='warm' and warm_checkpoint is not None
        if continuing:
            phase_clip=warm_checkpoint['config']['gradient_clip_norm']
            phase_gradient={'role':'inherited measured setting to isolate training-duration change',
                'clip':phase_clip,'parent':warm_parent}
        else:
            phase_clip,phase_gradient=gradient_calibration(flow,pool,config,seed+200,
                target=target,kind='rkl' if phase=='rkl' else 'forward',learning_rate=cal['learning_rate'],log_weights=lw)
        write_json(output/f'{phase}-gradient-calibration.json',phase_gradient)
        trainer=TrainingBlock(flow,target,batch=config['batch_size'],learning_rate=cal['learning_rate'],
            clip=phase_clip,kind=kind,walkers=config['walkers'],walk_steps=config['walker_steps'])
        total=0;rungs=config['training_rungs']
        if phase=='warm' and config.get('warm_total_updates') and not continuing:
            endpoint=config['warm_total_updates']
            rungs=[x for x in rungs if x<=endpoint]
            if not rungs:raise ValueError('warm endpoint below the first reviewed rung')
            while rungs[-1]<endpoint:rungs.append(min(2*rungs[-1],endpoint))
        if continuing:
            trainer.restore(warm_checkpoint);total=int(warm_checkpoint['step'])
            rungs=[];next_rung=total
            while next_rung<config['warm_total_updates']:
                next_rung=min(2*next_rung,config['warm_total_updates']);rungs.append(next_rung)
        for rung in rungs:
            begin=time.monotonic()
            result=trainer.run(tf.constant([seed,100+total+(0 if phase=='warm' else 100000)]),
                tf.constant(rung-total),pool,lw,current,tf.constant(cal['mala_step_size'],F64))
            count=int(result[0].numpy());current=result[1];total+=count
            metrics=assessment(flow,target,validation,prep['log_normalizer'],tf.constant([seed,41]))
            history.append({'phase':phase,'total_updates':total,'seconds':time.monotonic()-begin,
                'mean_loss':result[2]/max(count,1),'mean_gradient_norm':result[3]/max(count,1),
                'clipped_fraction':result[4]/max(count,1),'global_acceptance':result[5]/max(count,1),
                'local_acceptance':result[6]/max(count,1),'invalid_proposals':result[7],
                'valid':bool(result[-1].numpy()),'validation':metrics})
            write_json(output/'history.json',history)
            write_json(output/f'{phase}-{total}-checkpoint.json',trainer.checkpoint())
            if not bool(result[-1].numpy()):raise CandidateFailure('nonfinite training update; optimizer/capacity repair needed')
        metrics=assessment(flow,target,reference,prep['log_normalizer'],tf.constant([seed,501]),probe=probe)
        checkpoint=trainer.checkpoint()
        checkpoint_hash=checkpoint.get('checkpoint_hash',checkpoint.get('state_hash'))
        frozen=flow.frozen_payload(target_signature=target.signature,training_state_hash=checkpoint_hash)
        write_json(output/f'{phase}-frozen.json',frozen)
        write_json(output/f'{phase}-assessment.json',metrics)
        if phase=='warm':before=metrics
        else:after=metrics
    report={'target':target_name,'arm':arm,'seed':seed,'settings':cal,'teacher':teacher_report,
        'before_rkl':before,'after_rkl':after,'history':history,
        'finite_candidate':after['finite_density'] and after['post_training_1000']['finite'],
        'coverage_screen_passed':after['coverage_screen_passed'],
        'coverage_lost_during_rkl':bool(before and before['coverage_screen_passed'] and not after['coverage_screen_passed']),
        'training_quality_established':False,'posterior_qualified':False,
        'next_action':'frozen-map downstream qualification if viable; otherwise inspect stage and extend/repair',
        'xla':True,'training_batch_native':True,'sample_wise_target_loop':False,'dtype':'float64'}
    if warm_parent:report['warm_continuation_parent']=warm_parent
    report['continuing_improvement_at_cap']={phase:bool(len(rows)>1 and
        float(rows[-1]['validation']['heldout_forward_kl_estimate'].numpy()) <
        float(rows[-2]['validation']['heldout_forward_kl_estimate'].numpy()))
        for phase in ('warm','rkl') if (rows:=[r for r in history if r['phase']==phase])}
    report['continued_improvement_role']='descriptive repair trigger; not a statistical plateau test'
    write_json(output/'result.json',report)
    return report


def price_flow(target_name,calibration,output,config):
    """Instrument two full AFT stage loops; synchronized diagnostic, not throughput."""
    from bayesfilter.inference.neutra_flow_smc_tf import FlowTransportSMC
    target=WarmStartTarget(target_name);cal=json.loads(Path(calibration).read_text())['selected']
    flow=make_transport(target,cal['width'],(11,91))
    sampler=FlowTransportSMC(target,BroadStudentProposal(target.parameter_dim),flow.config,
        particles=config['flow_particles'],stages=config['flow_stages'],
        inner_updates=config['aft_inner_updates'],passes=1,
        learning_rate=cal['learning_rate'],gradient_clip=cal['clip'],mutation_steps=config['mutation_steps'],
        step_size=cal['mala_step_size'],resampling_fraction=config['resampling_fraction'])
    records=[]
    def progress(active):
        write_json(Path(output)/'pricing.json',{'diagnostic_only':True,'pricing_revision':2,
            'scope':'first two full AFT stages; explicit synchronization adds diagnostic overhead',
            'inner_updates':config['aft_inner_updates'],'population_count':3,
            'records':records,'active_operation':active,
            'traces':{'gradient':sampler.evaluate.experimental_get_tracing_count(),
                'validation':sampler.weights_only.experimental_get_tracing_count(),
                'optimizer':sampler.apply.experimental_get_tracing_count(),
                'weight_and_resample':sampler.weight_and_resample.experimental_get_tracing_count(),
                'mutation':sampler.mutation.run.experimental_get_tracing_count()}})
    def timed(label,fn):
        progress(label)
        start=time.monotonic();before=resource.getrusage(resource.RUSAGE_SELF)
        result=fn()
        for tensor in tf.nest.flatten(result):
            if tf.is_tensor(tensor):tensor.numpy()
        after=resource.getrusage(resource.RUSAGE_SELF)
        records.append({'operation':label,'wall_seconds':time.monotonic()-start,
            'cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime})
        progress(None)
        return result
    seed=tf.constant([11,7])
    populations=[sampler.initial_population(tf.random.experimental.stateless_fold_in(seed,j)) for j in range(3)]
    for stage in range(1,min(2,config['flow_stages'])+1):
        previous=tf.constant((stage-1)/config['flow_stages'],F64);beta=tf.constant(stage/config['flow_stages'],F64)
        timed(f'{stage}:restore',lambda:sampler.restore(sampler.initial))
        best=None;best_value=math.inf
        for iteration in range(config['aft_inner_updates']):
            prefix=f'{stage}:{iteration}'
            evaluated=timed(prefix+':gradient',lambda:sampler.evaluate(*populations[0][:2],previous,beta))
            value=float(timed(prefix+':validation',lambda:sampler.weights_only(*populations[1][:2],previous,beta))[0].numpy())
            if math.isfinite(value) and value<best_value:
                best_value=value;best=timed(prefix+':snapshot',sampler.snapshot)
            if not bool(timed(prefix+':optimizer',lambda:sampler.apply(*evaluated[1])).numpy()):
                raise CandidateFailure('nonfinite AFT optimizer in pricing diagnostic')
        if best is None:raise CandidateFailure('no finite AFT validation candidate in pricing diagnostic')
        timed(f'{stage}:restore_best',lambda:sampler.restore(best))
        for j in range(3):
            populations[j],_=timed(f'{stage}:advance:{j}',lambda:sampler.advance(
                populations[j],previous,beta,tf.random.experimental.stateless_fold_in(seed,100*stage+j)))
    return records


def diagnose_flow_composition(target_name,calibration,output,config):
    """Controlled nested/composed XLA comparison including actual AFT updates."""
    from bayesfilter.inference.neutra_flow_smc_tf import FlowTransportSMC
    target=WarmStartTarget(target_name);cal=json.loads(Path(calibration).read_text())['selected']
    flow=make_transport(target,cal['width'],(11,91));records=[];results={}
    for inline in (False,True):
        sampler=FlowTransportSMC(target,BroadStudentProposal(target.parameter_dim),flow.config,
            particles=config['flow_particles'],stages=config['flow_stages'],inner_updates=1,passes=1,
            learning_rate=cal['learning_rate'],gradient_clip=cal['clip'],mutation_steps=config['mutation_steps'],
            step_size=cal['mala_step_size'],resampling_fraction=config['resampling_fraction'],inline_target=inline)
        population=sampler.initial_population(tf.constant([11,71]))
        previous=tf.constant(0.,F64);beta=tf.constant(1/config['flow_stages'],F64)
        for operation in ('gradient','advance'):
            for repeat in range(3):
                label=f'{"composed" if inline else "nested"}:{operation}:{repeat}'
                write_json(Path(output)/'composition.json',{'diagnostic_only':True,'active_operation':label,'records':records})
                start=time.monotonic();before=resource.getrusage(resource.RUSAGE_SELF)
                value=(sampler.evaluate(*population[:2],previous,beta) if operation=='gradient' else
                    sampler.advance(population,previous,beta,tf.constant([11,83]))[0])
                tensors=tuple(tf.nest.flatten(value))
                for tensor in tensors:tensor.numpy()
                after=resource.getrusage(resource.RUSAGE_SELF)
                records.append({'operation':label,'wall_seconds':time.monotonic()-start,
                    'cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
                    'device':tensors[0].device,'gradient_traces':sampler.evaluate.experimental_get_tracing_count(),
                    'mutation_traces':sampler.mutation.run.experimental_get_tracing_count()})
                if operation=='gradient':
                    # Preserve the causal condition missing from revision 1:
                    # the actual loop interleaves a separate validation graph
                    # and optimizer mutation between gradient evaluations.
                    for name,fn in (('validation',lambda:sampler.weights_only(*population[:2],previous,beta)[0]),
                                    ('optimizer',lambda:sampler.apply(*value[1]))):
                        detail=label+':'+name
                        write_json(Path(output)/'composition.json',{'diagnostic_only':True,'revision':2,
                            'active_operation':detail,'records':records})
                        start=time.monotonic();before=resource.getrusage(resource.RUSAGE_SELF)
                        checked=fn();checked.numpy();after=resource.getrusage(resource.RUSAGE_SELF)
                        records.append({'operation':detail,'wall_seconds':time.monotonic()-start,
                            'cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime})
                        if name=='optimizer' and not bool(checked.numpy()):raise CandidateFailure('diagnostic AFT optimizer invalid')
            results[(inline,operation)]=tensors
    equivalence={}
    for operation in ('gradient','advance'):
        normalized=[tf.reduce_max(tf.abs(a-b)/(1+tf.abs(a)))
            for a,b in zip(results[(False,operation)],results[(True,operation)],strict=True)]
        maximum=float(tf.reduce_max(tf.stack(normalized)).numpy())
        equivalence[operation]={'max_normalized_error':maximum,'tolerance':1e-10,'passed':math.isfinite(maximum) and maximum<=1e-10}
    report={'diagnostic_only':True,'revision':2,'active_operation':None,'records':records,
        'equivalence':equivalence,'passed':all(v['passed'] for v in equivalence.values()),
        'interpretation':'same analytic target composed inside the outer compiled function; no method-quality inference'}
    write_json(Path(output)/'composition.json',report)
    if not report['passed']:raise ValueError('nested/composed target numerical equivalence failed')
    return report
