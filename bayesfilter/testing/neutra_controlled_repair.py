"""Controlled benchmark phases; shared IAF/Adam/HMC, no q20 authority."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig, NeuTraOptimizerConfig
from bayesfilter.inference.neutra_joint_training import JointNeuTraTrainer
from bayesfilter.inference.neutra_post_training import PostTrainingProbe
from bayesfilter.testing.neutra_warm_start_campaign import read_tensor, save_tensor, write_json, make_transport
from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget, F64
from bayesfilter.testing.neutra_rare_regions_tf import (
    Proposal, ImportanceProgram, event_features, known_masses, stratified_bank, seed_fold)
from bayesfilter.testing.neutra_rare_region_phases import probability_screen, bank_report, directed_geometry
from bayesfilter.testing.neutra_controlled_reference import ExactMixtureTransport, StandardGaussianReference, oracle_bank, marginal_cdf


def identity(path):
    digest=hashlib.sha256()
    for name in ('rows.tensor','log-weights.tensor'):
        digest.update((Path(path)/name).read_bytes())
    return digest.hexdigest()


def save_bank(path, rows, weights):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    save_tensor(path/'rows.tensor',rows);save_tensor(path/'log-weights.tensor',tf.nn.log_softmax(weights))


def prepare(target_name, output, cfg, repair=0):
    target=WarmStartTarget(target_name);out=Path(output)
    n=cfg['teacher_particles']*(2**repair)
    proposal=Proposal(target,bridge_weight=.5)
    program=ImportanceProgram(target,proposal,n)
    rows,weights,estimates=[],[],[]
    for i,seed in enumerate(cfg['teacher_seeds']):
        x,w,estimate,se=program.run(tf.constant([seed+100000*repair,312],tf.int32))
        save_bank(out/f'replication-{i}',x,w)
        rows.append(x);weights.append(w-math.log(len(cfg['teacher_seeds'])))
        estimates.append(estimate.numpy().tolist())
        write_json(out/f'replication-{i}/report.json',{'estimate':estimate,'se':se,'seed':[seed+100000*repair,312]})
    pooled_x,pooled_w=tf.concat(rows,0),tf.concat(weights,0)
    save_bank(out/'estimated',pooled_x,pooled_w)
    screen=probability_screen(estimates)
    pooled=bank_report(target,pooled_x,pooled_w)
    # Distinct oracle, development and final-confirmation streams.
    x,w=oracle_bank(target,cfg['oracle_per_stratum'],[44001,1])
    save_bank(out/'oracle',x,w)
    for role,seed in [('validation',44002),('confirmation',44003)]:
        x=target.reference_sample(cfg['reference_count'],tf.constant([seed,1],tf.int32))
        save_tensor(out/(role+'.tensor'),x)
    x,w=oracle_bank(target,cfg['oracle_per_stratum'],[44004,1])
    save_bank(out/'directed-reference',x,w)
    return {'target':target.specification,'teacher_screen':screen,'pooled_bank':pooled,
            'teacher_id':identity(out/'estimated'),'oracle_id':identity(out/'oracle'),
            'repair':repair,'all_replication_rows_saved':True,'passed':screen['passed'],
            'oracle_role':'separate exact-teacher diagnostic, no practical-teacher promotion'}


def reference_check(target_name, output, cfg):
    target=WarmStartTarget(target_name);transport=ExactMixtureTransport(target_name=='warped_mixture')
    # Central inverse images and broad tails; avoid inverse-CDF saturation.
    physical=tf.linspace(tf.constant(-12.,F64),tf.constant(12.,F64),1001)
    from bayesfilter.testing.neutra_controlled_reference import NORMAL
    p=marginal_cdf(physical)
    mask=(p>1e-14)&(p<1.-1e-14)
    z=tf.stack((NORMAL.quantile(tf.boolean_mask(p,mask)),tf.zeros_like(tf.boolean_mask(p,mask))),1)
    x,ve,se,ce=transport.check(z)
    value_error=float(tf.reduce_max(tf.abs(ve)))
    score_error=float(tf.reduce_max(tf.abs(se)))
    cdf_error=float(tf.reduce_max(tf.abs(ce)))
    save_tensor(Path(output)/'reference-z.tensor',z)
    save_tensor(Path(output)/'reference-x.tensor',x)
    # The score involves cancellation; this FP64 reference tolerance is
    # independently tested on ordinary and valley points, not reused for FP32.
    passed=value_error<1e-9 and score_error<1e-7 and cdf_error<1e-12
    return {'passed':passed,'value_error':value_error,'score_error':score_error,
            'cdf_error':cdf_error,'rows':int(tf.shape(z)[0]),'tolerance_role':'FP64 analytic-reference roundoff screen',
            'learned_map':False,'exact_in_real_arithmetic':True}


class StratifiedJointBlock:
    """Batch-native joint objective; common restorable Adam and unique streams."""
    def __init__(self,flow,target,rows,lw,teacher_id,*,batch,lr,forward,reverse):
        indices,conditional,masses=stratified_bank(rows,lw)
        k=int(indices.shape[0]);counts=[batch//k+(i<batch%k) for i in range(k)]
        optimizer=NeuTraOptimizerConfig(batch,'standard',lr,.9,.999,1e-8,None,True)
        self.trainer=JointNeuTraTrainer(flow,target.value_score,optimizer,target_signature=target.signature,
            teacher_id=teacher_id,forward_weight=forward,reverse_weight=reverse)
        self.flow=flow
        def draw(seed):
            xs,ws=[],[]
            for j,count in enumerate(counts):
                index=tf.random.stateless_categorical(conditional[j:j+1],count,seed_fold(seed,j))[0]
                xs.append(tf.gather(rows,tf.gather(indices[j],index)))
                ws.append(tf.fill([count],masses[j]-tf.math.log(tf.cast(count,F64))))
            return tf.concat(xs,0),tf.concat(ws,0)
        self.draw=tf.function(draw,input_signature=[tf.TensorSpec([2],tf.int32)],jit_compile=True,autograph=False)
        def run(seed,count):
            def body(i,loss,fn,rn,norm,valid):
                s=seed_fold(seed,tf.cast(self.trainer.optimizer.iterations,tf.int32))
                x,w=draw(s)
                z=tf.random.stateless_normal([batch,2],seed_fold(s,701),dtype=F64)
                r=self.trainer._joint_step(z,x,w)
                return i+1,loss+r['loss'],fn+r['forward_loss'],rn+r['reverse_loss'],norm+r['gradient_norm'],valid&r['valid']
            return tf.while_loop(lambda i,l,f,r,n,v:(i<count)&v,body,
                (tf.constant(0),tf.constant(0.,F64),tf.constant(0.,F64),tf.constant(0.,F64),tf.constant(0.,F64),tf.constant(True)))
        self.run=tf.function(run,input_signature=[tf.TensorSpec([2],tf.int32),tf.TensorSpec([],tf.int32)],jit_compile=True,autograph=False)

    def gradient_telemetry(self,seed):
        """Four independent minibatches at a frozen map; descriptive noise only."""
        batches=[]
        for j in range(4):
            s=tf.constant([seed,18001+j],tf.int32);x,w=self.draw(s)
            z=tf.random.stateless_normal([self.trainer.config.batch_size,2],seed_fold(s,701),dtype=F64)
            batches.append(self.trainer.evaluate_joint(z,x,w)['gradients'])
        result=[]
        for i,var in enumerate(self.flow.trainable_variables):
            gradients=tf.stack([g[i] for g in batches]);mean=tf.reduce_mean(gradients,0)
            variance=tf.reduce_sum(tf.math.reduce_variance(gradients,0))
            result.append({'variable':var.name,'mean_gradient_norm':tf.linalg.norm(mean),
                'gradient_noise_rms':tf.sqrt(variance),'parameter_norm':tf.linalg.norm(var)})
        return {'independent_batches':4,'role':'descriptive frozen-map gradient variability',
                'clipping_enabled':False,'layers':result}


def metric_programs(flow,target):
    cached=getattr(flow,'_controlled_metric_programs',None)
    if cached is not None:
        if cached['target_signature']!=target.signature:raise ValueError('metric target changed')
        return cached
    @tf.function(input_signature=[tf.TensorSpec([None,2],F64)],jit_compile=True,autograph=False)
    def logq(x):return flow.log_prob(x)
    @tf.function(input_signature=[tf.TensorSpec([None,2],F64)],jit_compile=True,autograph=False)
    def generate(z):return flow.forward_and_logdet(z)
    @tf.function(input_signature=[tf.TensorSpec([None,2],F64)],jit_compile=True,autograph=False)
    def posterior_geometry(x):
        latent=flow.inverse_theta_to_z_batch(x)
        _,score,valid=target.value_score(x)
        pullback=flow.pullback_score_batch(latent,score)
        determinant_score=flow.log_abs_det_jacobian_score_batch(latent)
        residual=pullback+determinant_score+latent
        return tf.linalg.norm(residual,axis=1),tf.linalg.norm(pullback,axis=1),tf.linalg.norm(determinant_score,axis=1),valid
    cached={'target_signature':target.signature,'logq':logq,'generate':generate,
            'posterior_geometry':posterior_geometry,'probe':PostTrainingProbe(flow,target,1.)}
    flow._controlled_metric_programs=cached
    return cached


def flow_metrics(flow,target,validation,reference,reference_weights,seed,*,full=False):
    programs=metric_programs(flow,target);logq=programs['logq']
    log_density=logq(validation)
    z=tf.random.stateless_normal([4096,2],tf.constant([seed,513],tf.int32),dtype=F64)
    x,ld=programs['generate'](z)
    reverse_terms=-.5*tf.reduce_sum(z*z,1)-math.log(2*math.pi)-target.log_prob(x)-ld
    kl=tf.reduce_mean(reverse_terms)
    # E_pi[q/pi * I] over an independently stratified exact reference.
    ratio=tf.exp(logq(reference)-target.log_prob(reference))
    event=event_features(target,reference)
    masses=tf.reduce_sum(tf.exp(reference_weights)[:,None]*ratio[:,None]*event[:,:3],0)
    zinv=flow.inverse_theta_to_z_batch(reference)
    truth=tf.constant(known_masses()[:3],F64)
    coverage=bool((tf.abs(masses[2]-truth[2])<.03)&(masses[0]>truth[0]*.25)&(masses[0]<truth[0]*4.))
    report={'heldout_fkl':-tf.reduce_mean(log_density),'estimated_rkl':kl,
            'proposal_event_masses_reference_integration':masses,'coverage_screen':coverage,
            'coverage_screen_role':'nomination only; approximate integration and loose initial shape screen',
            'finite':bool(tf.reduce_all(tf.math.is_finite(log_density))) and bool(tf.reduce_all(tf.math.is_finite(zinv)))}
    if full:
        # Conditional contributions preserve the stratum probabilities.
        reference_loss=-logq(reference)
        per_stratum=tf.reshape(tf.exp(reference_weights)*reference_loss,[5,-1])
        report['forward_loss_contributions_by_stratum']=tf.reduce_sum(per_stratum,1)
        report['reference_stratum_mass']=tf.reduce_sum(tf.reshape(tf.exp(reference_weights),[5,-1]),1)
        report['post_training_1000']=programs['probe'](seed=(seed,711))
        report['directed']=directed_geometry(flow,target,offsets=(-2.,-1.,0.,1.,2.))
        norms,pullback,determinant_score,valid=programs['posterior_geometry'](validation[:1000])
        report['posterior_reference_geometry']={'rows':1000,'residual_norm':norms,
            'pullback_score_norm':pullback,'logdet_score_norm':determinant_score,
            'finite':bool(tf.reduce_all(valid)&tf.reduce_all(tf.math.is_finite(norms))),
            'role':'independent posterior-reference diagnostic; not Gaussian-base probes'}
        report['finite'] &= report['post_training_1000']['finite'] and report['directed']['finite']
        report['finite'] &= report['posterior_reference_geometry']['finite']
    return report,{'forward':-log_density,'reverse':reverse_terms}


def paired_objective_progress(previous,current,forward,reverse):
    """Paired independent reference banks for the actual weighted objective."""
    gains={};variances={}
    for key in ('forward','reverse'):
        change=previous[key]-current[key]
        gains[key]=float(tf.reduce_mean(change))
        variances[key]=float(tf.math.reduce_variance(change)/tf.cast(tf.size(change)-1,F64))
    gain=forward*gains['forward']+reverse*gains['reverse']
    se=math.sqrt(forward**2*variances['forward']+reverse**2*variances['reverse'])
    return gain,se,gains,variances


def lifetime_updates(path):
    result=json.loads((Path(path)/'result.json').read_text())
    if 'lifetime_updates' in result:return result['lifetime_updates']
    if not result.get('parent') or result.get('repair'):return result['history'][-1]['step']
    return lifetime_updates(result['parent'])+sum(r['additional_updates'] for r in result['history'])


def train(target_name,prepared,output,cfg,seed,teacher,*,parent=None,arm='parent',repair=0,continuation=False):
    target=WarmStartTarget(target_name);out=Path(output);prepared=Path(prepared)
    bank=prepared/teacher;rows=read_tensor(bank/'rows.tensor');lw=read_tensor(bank/'log-weights.tensor')
    tid=identity(bank);validation=read_tensor(prepared/'validation.tensor')[:8192]
    ref=read_tensor(prepared/'directed-reference/rows.tensor');rw=read_tensor(prepared/'directed-reference/log-weights.tensor')
    started=time.monotonic();history=[];trials=[];parent_path=Path(parent) if parent else None
    forward,reverse={'parent':(1.,0.),'continue':(1.,0.),'forward':(1.,0.),'reverse':(0.,1.),'joint':(1.,1.)}[arm]
    if parent_path is None or repair:
        chosen=None;score=math.inf
        widths=[64] if repair else cfg['widths']
        for width in widths:
            for lr in cfg['learning_rates']:
                if repair:lr*=.5
                flow=make_transport(target,width,(seed,width),variance_scale=.2)
                block=StratifiedJointBlock(flow,target,rows,lw,tid,batch=cfg['batch_size'],lr=lr,forward=forward,reverse=reverse)
                tick=time.monotonic();r=block.run(tf.constant([seed,901+repair],tf.int32),tf.constant(cfg['pilot_updates']))
                metrics,_=flow_metrics(flow,target,validation,ref,rw,seed)
                trials.append({'width':width,'lr':lr,'seconds':time.monotonic()-tick,'updates':int(r[0]),'metrics':metrics})
                write_json(out/'pilots.json',trials)
                value=float(metrics['heldout_fkl'])
                if bool(r[-1]) and metrics['finite'] and value<score:
                    score=value;chosen=(flow,block,lr,width)
        if chosen is None:raise ValueError('all objective pilots invalid')
        flow,block,lr,width=chosen
    else:
        parent_result=json.loads((parent_path/'result.json').read_text())
        checkpoint=json.loads((parent_path/parent_result['selected_checkpoint']).read_text())
        base=checkpoint['base'];conf=NeuTraTransportConfig(**{**base['transport_config'],
            'hidden_layers':tuple(base['transport_config']['hidden_layers']),
            'seed':tuple(base['transport_config']['seed'])})
        flow=NeuTraTransport(conf);flow.restore_parameters(base['parameters'])
        lr=base['optimizer_config']['learning_rate'];width=conf.hidden_layers[0]
        block=StratifiedJointBlock(flow,target,rows,lw,tid,batch=cfg['batch_size'],lr=lr,forward=forward,reverse=reverse)
        if arm=='continue' or continuation:block.trainer.restore(checkpoint)
        write_json(out/'restoration.json',{'parent':str(parent_path),'teacher_id':tid,'optimizer_restored':arm=='continue' or continuation,
            'optimizer_start_step':int(block.trainer.optimizer.iterations),'objective':[forward,reverse]})
        if checkpoint['teacher_id']!=tid:raise ValueError('changed teacher under continuation')
    initial_step=int(block.trainer.optimizer.iterations);previous=None;flat_count=0;checkpoints=[]
    root_seed=tf.constant([seed,911 if arm=='parent' else 921],tf.int32)
    max_seconds=cfg['train_wall_seconds']
    stop='budget_limited';chosen_cp=None
    rungs=cfg.get('continuation_rungs',[8192,32768,65536]) if continuation else cfg['training_rungs']
    for rung in rungs:
        desired=rung if arm=='parent' or repair else initial_step+rung
        needed=desired-int(block.trainer.optimizer.iterations)
        if needed<=0:continue
        elapsed=time.monotonic()-started
        if elapsed>max_seconds*.85:break
        before=[v.read_value() for v in flow.trainable_variables]
        tick=time.monotonic();r=block.run(root_seed,tf.constant(needed));train_seconds=time.monotonic()-tick
        if not bool(r[-1]):stop='numerical_failure';break
        step=int(block.trainer.optimizer.iterations)
        tick=time.monotonic();metrics,logs=flow_metrics(flow,target,validation,ref,rw,seed,full=True)
        telemetry=block.gradient_telemetry(seed+step)
        telemetry['net_parameter_changes']=[{'variable':v.name,
            'norm':tf.linalg.norm(v-b),'relative_norm':tf.linalg.norm(v-b)/(1e-12+tf.linalg.norm(b))}
            for v,b in zip(flow.trainable_variables,before)]
        write_json(out/f'step-{step}-optimizer-diagnostic.json',telemetry)
        val_seconds=time.monotonic()-tick
        gain=None;se=None;term_gains={};term_variances={}
        if previous is not None:
            gain,se,term_gains,term_variances=paired_objective_progress(previous,logs,forward,reverse)
            flat_count=flat_count+1 if abs(gain)+3*se<cfg['plateau_delta'] else 0
        previous=logs
        tag=f'step-{step}'
        write_json(out/(tag+'-checkpoint.json'),block.trainer.checkpoint())
        write_json(out/(tag+'-frozen.json'),flow.frozen_payload(target_signature=target.signature))
        write_json(out/(tag+'-metrics.json'),metrics)
        entry={'stage':tag,'filename':tag+'-frozen.json','eligible':metrics['finite'],
               'transport_hash':flow.frozen_payload(target_signature=target.signature)['transport_hash']}
        checkpoints.append(entry);chosen_cp=tag+'-checkpoint.json'
        history.append({'step':step,'additional_updates':int(r[0]),'train_seconds':train_seconds,'validation_seconds':val_seconds,
            'priced_min_updates_for_10pct_validation_share':math.ceil(9*val_seconds/(train_seconds/max(needed,1))),
            'loss':r[1]/tf.cast(r[0],F64),'mean_gradient_norm':r[4]/tf.cast(r[0],F64),
            'paired_fkl_gain':term_gains.get('forward'),'paired_gain_se':math.sqrt(term_variances['forward']) if term_variances else None,
            'paired_objective_gain':gain,'paired_objective_se':se,
            'paired_term_gains':term_gains,'metrics':metrics})
        write_json(out/'history.json',history)
        write_json(out/'checkpoint-candidates.json',{'candidates':checkpoints,'selection_order':'earliest numerically valid preserved stage; posterior qualification required'})
        if not metrics['finite']:stop='numerical_failure';break
        if flat_count>=2:stop='operational_plateau';break
        # Price the next allocation using measured update and validation work.
        projected=(rungs[-1]-rung)*train_seconds/max(needed,1)
        if time.monotonic()-started+min(projected,train_seconds*3)+val_seconds>max_seconds:break
    if not history:
        raise RuntimeError('no post-pilot checkpoint completed within training budget')
    final=history[-1]['metrics']
    lifetime=(lifetime_updates(parent_path)+sum(r['additional_updates'] for r in history)
              if parent_path and not repair else history[-1]['step'])
    result={'target':target_name,'seed':seed,'teacher':teacher,'teacher_id':tid,'arm':arm,'repair':repair,
            'width':width,'learning_rate':lr,'objective_weights':[forward,reverse],
            'stop_reason':stop,'history':history,'pilots':trials,'selected_checkpoint':chosen_cp,
            'selected_frozen':chosen_cp.replace('-checkpoint','-frozen'),'finite':final['finite'],
            'coverage_screen':final['coverage_screen'],'training_converged':False,
            'canonical_architecture':flow.config.payload(),'dtype':'float64 diagnostic reference',
            'batch_native':True,'sample_wise_target_loop':False,'jit_compile':True,
            'continuation':continuation,'objective_progress_role':'paired change in actual branch objective; operational repeated-look diagnostic',
            'lifetime_updates':lifetime,
            'parent':str(parent_path) if parent_path else None}
    write_json(out/'result.json',result)
    return result


def exact_hmc(target_name,prepared,output,cfg,seed,repair=0):
    from bayesfilter.testing.neutra_warm_start_qualification import qualify
    ref=StandardGaussianReference()
    flow=make_transport(ref,4,(11,4))
    for var in flow.trainable_variables:var.assign(tf.zeros_like(var))
    write_json(Path(output)/'identity-frozen.json',flow.frozen_payload(target_signature=ref.signature))
    # The analytic control must not consume a learned-map confirmation bank.
    control_reference=Path(output)/'control-reference.tensor'
    save_tensor(control_reference,WarmStartTarget(target_name).reference_sample(
        cfg['reference_count'],tf.constant([seed,84003],tf.int32)))
    return qualify(target_name,seed,prepared,output,output,cfg,frozen_filename='identity-frozen.json',
                   reference_control=True,confirm=True,reference_path=control_reference)


def recover_confirmed_control(previous):
    """Recover the specific duplicate-export failure without new simulation."""
    previous=Path(previous)
    read=lambda name:json.loads((previous/name).read_text())
    manifest=read('manifest.json');result=read('qualification.json')
    if not (manifest.get('error_type')=='FileExistsError' and
            str(previous/'verified-member.json') in manifest.get('error','') and
            result.get('reference_control') is True):
        raise ValueError('not the supported completed-control export failure')
    posterior=read('posterior.json');reference=read('final-reference-check.json')
    screen=read('member-screen.json');member=read('verified-member.json')
    selected=next((row for row in screen if row['passed']),None)
    if (selected is None or member['candidate_id']!=selected['candidate_id'] or
            posterior.get('selection_screen_passed') is not True or
            posterior.get('final_reference_check')!=reference or
            posterior.get('passed') is not reference.get('passed')):
        raise ValueError('completed control evidence is inconsistent')
    return {**result,'qualified':reference['passed'],'heldout_consumed':True,
        'selection_screen_passed':True,'selected_candidate_id':selected['candidate_id'],
        'member_screen':screen,'posterior':str(previous/'posterior.json'),
        'warmup_results':selected['warmup_results'],'retained_results':selected['retained_results'],
        'reason':'declared_checks_passed' if reference['passed'] else 'final_reference_failed',
        'recovered_from':str(previous),'numerical_source_manifest':manifest['source_manifest'],
        'recovery_role':'finalization only; no new simulation or new tuning authority'}


def iid_assessment_control(target_name,output,cfg):
    """Independent exact samples exercise the declared retained assessment."""
    from bayesfilter.testing.neutra_warm_start_qualification import (
        features,feature_names,retained_event_check,final_reference_check)
    from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
    from bayesfilter.inference.hmc_posterior_assessment import HMCPosteriorAssessmentPolicy,assess_posterior
    from bayesfilter.inference.hmc_precision import HMCPrecisionPolicy,HMCPrecisionTarget
    target=WarmStartTarget(target_name);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    profile=cfg['hmc']['diagnostic_profile'];names=feature_names(target,profile=profile)
    policy=HMCPosteriorAssessmentPolicy(retained_bulk_ess_min=400,retained_tail_ess_min=400,
        precision=HMCPrecisionPolicy(tuple(HMCPrecisionTarget(n,mcse_sd_ratio_max=.03) for n in names)),
        quantities_id=profile,binary_quantity_names=names[4:])
    def quantities(draws):
        values=features(target,draws,profile=profile)
        return {n:values[...,i] for i,n in enumerate(names) if i>=2}
    reference=out/'independent-reference.tensor'
    save_tensor(reference,target.reference_sample(cfg['reference_count'],tf.constant([74001,1])))
    rows=[];all_finite=True
    for seed in cfg['teacher_seeds']:
        draws=tf.reshape(target.reference_sample(40000,tf.constant([seed,74002])),[10000,4,2])
        all_finite &= bool(tf.reduce_all(tf.math.is_finite(draws)))
        save_tensor(out/f'iid-{seed}.tensor',draws)
        result=assess_posterior(draws,names[:2],policy=policy,stage='retained',
            rhat=rank_normalized_split_rhat_summary(draws,rhat_max=1.01),quantities_fn=quantities,
            extra=retained_event_check(target,draws,profile=profile,broad_relative_mcse=cfg['hmc']['broad_relative_mcse']))
        agreement=final_reference_check(target,draws,reference,profile=profile)
        write_json(out/f'assessment-{seed}.json',result)
        write_json(out/f'reference-{seed}.json',agreement)
        rows.append({'seed':[seed,74002],'assessment_passed':result['passed'],
            'reference_passed':agreement['passed'],'broad_relative_mcse':result['broad_relative_mcse'],
            'failed_checks':result['failed_checks']})
    return {'target':target_name,'replications':rows,'iid_exact_reference':True,
        'retained_count_per_chain':10000,'chain_count':4,
        'role':'finite-sample diagnostic control; no nominal coverage or reliability certification',
        'all_inputs_finite':all_finite,'joint_pass_count':sum(r['assessment_passed'] and r['reference_passed'] for r in rows)}


def qualify_training(target_name,prepared,training,output,cfg,seed,*,confirm=False,repair=0,frozen_filename=None):
    from bayesfilter.testing.neutra_warm_start_qualification import qualify
    training=Path(training);output=Path(output)
    if confirm:
        if frozen_filename is None:raise ValueError('confirmation requires the preselected frozen checkpoint')
        return qualify(target_name,seed,prepared,training,output,cfg,frozen_filename=frozen_filename,
                       confirm=True,start_scope='modes')
    listing=json.loads((training/'checkpoint-candidates.json').read_text())
    candidates=[r for r in listing['candidates'] if r['eligible']]
    if frozen_filename is not None:
        candidates=[r for r in candidates if r['filename']==frozen_filename]
        if len(candidates)!=1:raise ValueError('requested checkpoint is absent or ineligible')
    deadline=time.monotonic()+cfg['hmc']['job_wall_seconds'];screens=[]
    for ordinal,row in enumerate(candidates):
        payload=json.loads((training/row['filename']).read_text())
        if payload['transport_hash']!=row['transport_hash']:raise ValueError('checkpoint identity mismatch')
        remaining=deadline-time.monotonic()
        if remaining<20.:break
        # Equal remaining allowance avoids starving preserved earlier maps.
        allowance=remaining/(len(candidates)-ordinal)
        local={**cfg,'hmc':{**cfg['hmc'],'job_wall_seconds':allowance}}
        destination=output/f'checkpoint-{ordinal:02d}-{row["stage"]}'
        result=qualify(target_name,seed+ordinal,prepared,training,destination,local,
            frozen_filename=row['filename'],confirm=False,start_scope='modes' if repair else 'map')
        screens.append({'stage':row['stage'],'path':str(destination),
            'passed':result.get('selection_screen_passed',False),'reason':result.get('reason')})
        write_json(output/'checkpoint-screen.json',screens)
        if result.get('selection_screen_passed'):
            return {**result,'selected_frozen':row['filename'],'checkpoint_screen':screens}
    return {'qualified':False,'selection_screen_passed':False,'heldout_consumed':False,
            'checkpoint_screen':screens,'reason':'no_preserved_checkpoint_passed_within_allocation'}
