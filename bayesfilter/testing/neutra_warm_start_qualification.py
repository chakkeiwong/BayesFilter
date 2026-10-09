"""Simpler-target NeuTra qualification through the existing public authorities.

The verified member delegates to run_sequential_neutra_hmc in
hmc_candidate_set_retained.py. No local chain runner or mass adaptation exists
here. Passing finite operational checks is not proof of exhaustive mode coverage.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import tensorflow as tf

from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget,F64
from bayesfilter.testing.neutra_warm_start_campaign import read_tensor,save_tensor,write_json
from bayesfilter.testing.neutra_warm_start_diagnostics import (
    features,feature_names,DIAGNOSTIC_REVISION,LEGACY_DIAGNOSTIC_PROFILE,
    SHAPE_DIAGNOSTIC_PROFILE,RARE_EVENT_DIAGNOSTIC_PROFILE)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
    FixedTransportHMCKernelTuningConfig,tune_fixed_transport_hmc_kernel)
from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig
from bayesfilter.inference.hmc_candidate_set_execution import HMCCandidateExecutionConfig
from bayesfilter.inference.hmc_candidate_set_retained import build_retained_bound_hmc_archive_runner_from_candidate_set_result
from bayesfilter.inference.hmc_verification import HMCAcceptancePolicy
from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig,NEUTRA_SEQUENTIAL_HMC_POLICY_ID
from bayesfilter.inference.hmc_posterior_assessment import HMCPosteriorAssessmentPolicy
from bayesfilter.inference.hmc_precision import HMCPrecisionPolicy,HMCPrecisionTarget,mean_precision
from bayesfilter.runtime.execution_budget import execution_budget,execution_budget_available,ExecutionBudgetExceeded


class BenchmarkAdapter:
    def __init__(self,target):
        self.target=target;self.parameter_dim=target.parameter_dim

    def adapter_signature(self):return self.target.signature

    def value_score_capability(self):
        return ValueScoreCapability(value_score_authority='graph_native',xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend='native_batched_tf_analytic_benchmark',target_scope=self.target.name,
            score_provenance='exact autodiff of analytic batched log density; finite-difference checked',
            evidence_path='tests/test_neutra_warm_start_pipeline.py',
            nonclaims=('analytic benchmark diagnostic compilation capability; actual numerical admission remains required',
                       'no q20 or learned-map quality claim'))

    def log_prob_and_grad(self,x):
        x=tf.convert_to_tensor(x,F64)
        scalar=x.shape.rank==1
        v,g,_=self.target.value_score(x[None] if scalar else x)
        return (v[0],g[0]) if scalar else (v,g)

    def log_prob(self,x):return self.log_prob_and_grad(x)[0]


def verified_member_order(result, limit=3):
    """Declared trajectory diversity among verified IDs, never metric ranking."""
    if not isinstance(limit,int) or limit<1:raise ValueError('member limit must be positive')
    records={c.candidate_id:c for c in result.candidates}
    first=[];rest=[];seen=set()
    for candidate_id in result.verified_candidate_ids:
        steps=records[candidate_id].leapfrog_steps
        (rest if steps in seen else first).append(candidate_id)
        seen.add(steps)
    return (first+rest)[:limit]


def reference_agreement(estimate,expected,mcse,ref_var,reference_count,*,continuous_count,strict_events):
    """Binary probabilities have no justified additive reference-SD allowance."""
    sampling_tolerance=4*tf.sqrt(mcse**2+ref_var/tf.cast(reference_count,F64))
    allowance=.03*tf.sqrt(ref_var)
    if strict_events:
        allowance=tf.concat((allowance[:continuous_count],tf.zeros_like(allowance[continuous_count:])),0)
    tolerance=sampling_tolerance+allowance
    errors=tf.abs(estimate-expected)
    return errors,tolerance


def retained_event_check(target,draws,*,profile, broad_relative_mcse=None):
    """Observed events in each chain; necessary evidence, never a mixing proof."""
    values=features(target,draws,profile=profile)[...,2*target.parameter_dim:]
    count=tf.reduce_sum(values,axis=0);n=tf.cast(tf.shape(values)[0],F64)
    observed=(count>0)&(count<n)
    result = {'passed':bool(tf.reduce_all(observed).numpy()),
        'quantity_names':feature_names(target,profile=profile)[2*target.parameter_dim:],
        'event_counts_per_chain':count,'both_outcomes_per_chain':observed,
        'role':'retained per-chain event observation requirement; no relative precision claim'}
    if broad_relative_mcse is not None:
        names = result['quantity_names']
        index = names.index('valley_abs_x0_lt2')
        summary = mean_precision(values[..., index:index+1])
        estimate = tf.reduce_mean(values[..., index])
        relative = summary['mcse'][0]/estimate
        precision_pass = bool(tf.math.is_finite(relative) & (estimate > 0.) & (relative <= broad_relative_mcse))
        result.update(broad_probability=estimate, broad_relative_mcse=relative,
                      broad_relative_mcse_limit=broad_relative_mcse,
                      broad_precision_passed=precision_pass,
                      role='per-chain event observations and declared broad-event relative MCSE',
                      passed=result['passed'] and precision_pass)
    return result


def final_reference_check(target, draws, path,*,profile=LEGACY_DIAGNOSTIC_PROFILE):
    """Open final holdout only after the member is selected without its values."""
    reference=features(target,read_tensor(path),profile=profile)
    expected=tf.reduce_mean(reference,0);ref_var=tf.math.reduce_variance(reference,0)
    values=features(target,draws,profile=profile);estimate=tf.reduce_mean(values,axis=(0,1))
    mcse=mean_precision(values)['mcse']
    errors,tolerance=reference_agreement(estimate,expected,mcse,ref_var,tf.shape(reference)[0],
        continuous_count=2*target.parameter_dim,strict_events=profile==RARE_EVENT_DIAGNOSTIC_PROFILE)
    return {'passed':bool(tf.reduce_all(tf.math.is_finite(errors)&(errors<=tolerance)).numpy()),
        'diagnostic_profile':profile,'quantity_names':feature_names(target,profile=profile),
        'estimate':estimate,'reference_mean':expected,'mcse':mcse,
        'reference_error':errors,'reference_tolerance':tolerance,
        'reference_rule':('binary events: 4 combined standard errors; continuous: additionally 0.03 reference SD; operational, not simultaneous coverage'
            if profile==RARE_EVENT_DIAGNOSTIC_PROFILE else
            '4 combined standard errors plus 0.03 reference SD; operational screen, not simultaneous coverage')}


def qualify(target_name,seed,prepared,training,output,config,*,frozen_filename='rkl-frozen.json',discovered_starts=False,confirm=True,
            reference_control=False,start_scope='map',reference_path=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=WarmStartTarget(target_name)
    numerical_target=target
    decoder=None
    if reference_control:
        from bayesfilter.testing.neutra_controlled_reference import StandardGaussianReference, ExactMixtureTransport
        numerical_target=StandardGaussianReference()
        decoder=ExactMixtureTransport(target_name=='warped_mixture')
    adapter=BenchmarkAdapter(numerical_target)
    training=Path(training)
    frozen=json.loads((training/frozen_filename).read_text())
    flow=load_frozen_neutra_artifact(frozen,expected_target_signature=numerical_target.signature).transport
    # Density modes locate peaks, not posterior volume. Generate the actual
    # chain bank from the frozen map, independently of discovery coordinates.
    latent=tf.random.stateless_normal([4,target.parameter_dim],tf.constant([seed,702]),dtype=F64)
    current=flow.forward_batch(latent)
    if start_scope=='modes':
        if reference_control: raise ValueError('reference control uses Gaussian initial states')
        centers=tf.repeat(target.known_representatives(),2,axis=0)
        current=centers+.1*tf.random.stateless_normal([4,target.parameter_dim],tf.constant([seed,703]),dtype=F64)
        latent=flow.inverse_theta_to_z_batch(current)
    elif start_scope!='map': raise ValueError('unknown start scope')
    values,scores,valid=numerical_target.value_score(current)
    if not bool(tf.reduce_all(valid).numpy()):raise ValueError('nonfinite frozen-map initial states')
    error=tf.reduce_max(tf.abs(flow.forward_batch(latent)-current))
    inverse_error=tf.reduce_max(tf.abs(flow.inverse_theta_to_z_batch(current)-latent))
    error=tf.maximum(error,inverse_error)
    if not bool((error<1e-8).numpy()):raise ValueError('frozen map initial-state roundtrip failed')
    save_tensor(output/'initial-physical.tensor',current)
    save_tensor(output/'initial-latent.tensor',latent)
    write_json(output/'initialization.json',{'source':'frozen_map_standard_normal' if start_scope=='map' else 'mode_spanning_within_mode_jitter',
        'seed':[seed,702],'transport_hash':frozen['transport_hash'],
        'roundtrip_max_error':error,'target_log_density':values,
        'physical_score_norm':tf.linalg.norm(scores,axis=1),'latent_norm':tf.linalg.norm(latent,axis=1),
        'posterior_convergence_established':False})
    knobs=config['hmc'];deadline=time.monotonic()+knobs['job_wall_seconds'];offset=knobs.get('seed_offset',0)
    profile=knobs.get('diagnostic_profile',LEGACY_DIAGNOSTIC_PROFILE)
    policy=HMCAcceptancePolicy(temporal_conflict_method=knobs.get(
        'temporal_conflict_method','raw_block_crossing_v1'))
    execution=HMCCandidateExecutionConfig(measurement_num_results=knobs.get('measurement_num_results',64),
        verification_num_results=knobs.get('verification_num_results',64),
        num_warmup_steps=0,seed=(seed,91001+offset),acceptance_policy=policy,target_status_trace_policy='none',
        use_xla=True,chain_mode='batched',chunk_max_results=256,reuse_leapfrog_graphs=True)
    ls=tuple(knobs['leapfrogs'])
    search=HMCControllerConfig(primary_l_grid=ls,initial_epsilon=knobs['initial_epsilon'],
        pilot_enabled=True,evidence_rungs=tuple(knobs.get('evidence_rungs',(1,2))),
        refinement_rounds=knobs.get('refinement_rounds',1),
        explore_failed_intervals=bool(knobs.get('explore_failed_intervals',False)),
        max_candidates=knobs['max_candidates'],total_budget_units=knobs['work_units'],
        repair_reserve_units=6,max_wall_time_seconds=knobs['job_wall_seconds'])
    tuning_config=FixedTransportHMCKernelTuningConfig(initial_step_size=knobs['initial_epsilon'],
        maximum_candidate_step_size=2.,leapfrog_grid=ls,target_scope=numerical_target.name,
        fixed_grid_max_attempts=knobs.get('fixed_grid_max_attempts',8),use_xla=True)
    write_json(output/'effective-tuning.json',{'initial_epsilon':knobs['initial_epsilon'],
        'fixed_grid_max_attempts':tuning_config.fixed_grid_max_attempts,
        'leapfrogs':ls,'work_units':knobs['work_units'],'max_candidates':knobs['max_candidates'],
        'measurement_num_results':execution.measurement_num_results,
        'verification_num_results':execution.verification_num_results,
        'policy':policy.payload(),'initialization':'initialization.json'})
    with execution_budget(deadline=deadline):
        run=tune_fixed_transport_hmc_kernel(base_adapter=adapter,fixed_transport=flow,initial_position=latent,
            config=tuning_config,search_config=search,execution_config=execution,
            frozen_transport_payload=frozen,output_dir=output/'tuning',
            target_lineage=numerical_target.specification,
            source_paths=(__file__,str(Path(__file__).with_name('neutra_warm_start_targets_tf.py')),
                          str(Path(__file__).with_name('neutra_controlled_reference.py'))))
        result={'target':target_name,'seed':seed,'training':str(training),'policy_id':NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
            'diagnostic_revision':4 if profile==RARE_EVENT_DIAGNOSTIC_PROFILE else (3 if profile==SHAPE_DIAGNOSTIC_PROFILE else DIAGNOSTIC_REVISION),
            'diagnostic_profile':profile,'transport_hash':frozen['transport_hash'],
            'tuning_completion':run.result.completion_status,'verified_members':list(run.result.verified_candidate_ids),
            'qualified':False,'heldout_consumed':False,'reference_control':reference_control,'start_scope':start_scope,
            'member_selection':'first health/convergence/precision pass; distinct L first; final reference once after selection',
            'mass_policy':'fixed identity in latent coordinates','method_ranking_established':False}
        write_json(output/'qualification.json',result)
        if not run.result.verified_candidate_ids:
            result['reason']='no verified pair within bounded search; map/kernel repair remains possible'
            write_json(output/'qualification.json',result);return result
        all_names=feature_names(target,profile=profile)
        names=all_names[:target.parameter_dim];extras=all_names[target.parameter_dim:]
        precision=HMCPrecisionPolicy(tuple(HMCPrecisionTarget(n,mcse_sd_ratio_max=.03) for n in (*names,*extras)))
        binary=all_names[2*target.parameter_dim:]
        assessment=HMCPosteriorAssessmentPolicy(retained_bulk_ess_min=400,retained_tail_ess_min=400,
            precision=precision,quantities_id=profile,binary_quantity_names=binary)
        def quantities(draws):
            values=features(target,draws,profile=profile)[...,target.parameter_dim:]
            return {n:values[...,i] for i,n in enumerate(extras)}
        # Pilot verification establishes that a fixed member is runnable.  The
        # old consumer selected the first member even when its retained
        # precision failed.  Screen verified members in declared order and
        # stop at the first member passing the existing posterior policy.
        member_screen=[];selected=None;selected_posterior=None
        order=verified_member_order(run.result,len(run.result.verified_candidate_ids))
        excluded={tuple(pair) for pair in knobs.get('excluded_kernel_pairs',())}
        records={c.candidate_id:c for c in run.result.candidates}
        order=[key for key in order if (records[key].epsilon,records[key].leapfrog_steps) not in excluded]
        order=order[:knobs.get('posterior_member_limit',3)]
        result['declared_member_order']=order
        for ordinal,candidate_id in enumerate(order):
            if not execution_budget_available():
                result['reason']='qualification_budget_exhausted';break
            member=build_retained_bound_hmc_archive_runner_from_candidate_set_result(
                candidate_set_result=run.result,candidate_id=candidate_id,
                retained_binding=run.adapter._execution_binding)
            member_dir=output/f'member-{ordinal:03d}-{candidate_id.rsplit(":",1)[-1]}'
            member_dir.mkdir(parents=True,exist_ok=False)
            member.export(member_dir/'verified-member.json')
            sequential=SequentialNeuTraHMCConfig(step_size=member.step_size,
                num_leapfrog_steps=member.num_leapfrog_steps,
                warmup_seed=(seed,191001+offset),retained_seed=(seed,291001+offset),
                assessment_policy=assessment,jit_compile=True)
            def archive(**kwargs):
                stage=kwargs['stage'];tag='cumulative' if kwargs.get('cumulative') else str(kwargs['chunk_index'])
                path=member_dir/f'{stage}-{tag}.tensor';save_tensor(path,kwargs['model_samples'])
                return {'path':str(path),'warmup_excluded_from_estimates':stage=='warmup'}
            try:
                event_options={}
                if profile==RARE_EVENT_DIAGNOSTIC_PROFILE:
                    event_options['retained_diagnostic_fn']=lambda draws:retained_event_check(target,draws,profile=profile,
                        broad_relative_mcse=knobs.get('broad_relative_mcse'))
                if decoder is not None:
                    event_options['model_transform']=decoder.decode
                posterior=member.run_sequential(config=sequential,parameter_names=names,quantities_fn=quantities,
                    archive_callback=archive,
                    budget_check=lambda _:execution_budget_available(),**event_options)
            except ExecutionBudgetExceeded:
                result['reason']='qualification_budget_exhausted';break
            for key in ('private_retained_raw','private_warmup_raw'):
                save_tensor(member_dir/(key+'.tensor'),posterior[key])
            public={k:v for k,v in posterior.items() if not k.startswith('private_')}
            write_json(member_dir/'posterior.json',public)
            row={'candidate_id':candidate_id,'candidate_ordinal':ordinal,
                 'step_size':member.step_size,'leapfrog_steps':member.num_leapfrog_steps,
                 'posterior':str(member_dir/'posterior.json'),'passed':bool(posterior['passed']),
                 'warmup_results':posterior['warmup_results_per_chain'],
                 'retained_results':posterior['retained_results_per_chain']}
            member_screen.append(row);write_json(output/'member-screen.json',member_screen)
            if posterior['passed']:
                selected=member;selected_posterior=posterior
                result['selected_retained_path']=str(member_dir/'private_retained_raw.tensor')
                break
        if selected is None:
            result.update(qualified=False,member_screen=member_screen,
                reason=result.get('reason','no_verified_member_passed_posterior_screen'),
                claim_limit='fixed-map member screen only; no method ranking or q20 promotion')
            write_json(output/'qualification.json',result);return result
        public={k:v for k,v in selected_posterior.items() if not k.startswith('private_')}
        result.update(selection_screen_passed=True,selected_candidate_id=selected.candidate.candidate_id,
            member_screen=member_screen,posterior=str(output/'posterior.json'),
            warmup_results=selected_posterior['warmup_results_per_chain'],
            retained_results=selected_posterior['retained_results_per_chain'])
        selected.export(output/'verified-member.json')
        if not confirm:
            public.update(selection_screen_passed=True,passed=False,confirmation_pending=True)
            write_json(output/'posterior.json',public)
            result.update(reason='selected_without_final_reference',qualified=False,heldout_consumed=False)
            write_json(output/'qualification.json',result)
            return result
        options={} if profile==LEGACY_DIAGNOSTIC_PROFILE else {'profile':profile}
        reference_result=final_reference_check(target,selected_posterior['private_retained_raw'],
                                              Path(reference_path) if reference_path is not None else Path(prepared)/'confirmation.tensor',**options)
        write_json(output/'final-reference-check.json',reference_result)
        public={k:v for k,v in selected_posterior.items() if not k.startswith('private_')}
        public.update(selection_screen_passed=True,final_reference_check=reference_result,
                      passed=reference_result['passed'])
        write_json(output/'posterior.json',public)
        result.update(qualified=reference_result['passed'],heldout_consumed=True,
            posterior=str(output/'posterior.json'),
            selected_candidate_id=selected.candidate.candidate_id,member_screen=member_screen,
            warmup_results=selected_posterior['warmup_results_per_chain'],
            retained_results=selected_posterior['retained_results_per_chain'],
            reason='declared_checks_passed' if reference_result['passed'] else 'final_reference_failed',
            claim_limit='one frozen-map member screen; no method ranking or q20 promotion')
        write_json(output/'qualification.json',result)
        return result


def confirm_qualification(target_name,prepared,output):
    """One holdout use after BOTH map and member selection; no new simulation."""
    output=Path(output)
    result=json.loads((output/'qualification.json').read_text())
    if result.get('heldout_consumed'):
        return result
    if not result.get('selection_screen_passed'):
        raise ValueError('cannot confirm before posterior selection passes')
    profile=result['diagnostic_profile']
    target=WarmStartTarget(target_name)
    checked=final_reference_check(target,read_tensor(result['selected_retained_path']),
        Path(prepared)/'confirmation.tensor',profile=profile)
    write_json(output/'final-reference-check.json',checked)
    public=json.loads((output/'posterior.json').read_text())
    public.update(passed=checked['passed'],confirmation_pending=False,final_reference_check=checked)
    write_json(output/'posterior.json',public)
    result.update(qualified=checked['passed'],heldout_consumed=True,
        reason='declared_checks_passed' if checked['passed'] else 'final_reference_failed')
    write_json(output/'qualification.json',result)
    return result


def qualify_shortlist(target_name,seed,prepared,training,output,config):
    """Fixed checkpoint order, downstream selection, exactly one final holdout."""
    training=Path(training);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    listing=json.loads((training/'checkpoint-candidates.json').read_text())
    if listing.get('schema')!='neutra.checkpoint_candidates.v1':
        raise ValueError('unsupported checkpoint shortlist')
    deadline=time.monotonic()+config['hmc'].get('shortlist_wall_seconds',3600.)
    screens=[];selected=None
    for ordinal,row in enumerate(listing['candidates']):
        if not row['eligible']:
            screens.append({'stage':row['stage'],'status':'excluded_by_recorded_screen'});continue
        frozen=json.loads((training/row['filename']).read_text())
        if frozen['transport_hash']!=row['transport_hash']:
            raise ValueError('shortlist frozen-map identity mismatch')
        probe=json.loads((training/row['probe_file']).read_text())
        if not (probe.get('finite') is True and probe.get('complete') is True
                and probe.get('rows')==1000 and probe.get('valid_rows')==1000
                and probe.get('transport_hash')==row['transport_hash']):
            raise ValueError('eligible shortlist map lacks a valid 1000-point probe')
        seconds=deadline-time.monotonic()
        if seconds<=0:break
        cfg={**config,'hmc':{**config['hmc'],
            'job_wall_seconds':min(seconds,config['hmc']['job_wall_seconds'])}}
        destination=output/f'checkpoint-{ordinal:02d}-{row["stage"]}'
        result=qualify(target_name,seed,prepared,training,destination,cfg,
            frozen_filename=row['filename'],confirm=False)
        screens.append({'stage':row['stage'],'output':str(destination),
            'selection_screen_passed':result.get('selection_screen_passed',False),
            'reason':result['reason'],'transport_hash':row['transport_hash']})
        write_json(output/'checkpoint-screen.json',screens)
        if result.get('selection_screen_passed'):
            selected=(row,destination);break
    if selected is None:
        result={'qualified':False,'heldout_consumed':False,'checkpoint_screen':screens,
            'reason':'no_checkpoint_passed_within_declared_budget'}
    else:
        row,destination=selected
        result=confirm_qualification(target_name,prepared,destination)
        result.update(selected_stage=row['stage'],checkpoint_screen=screens,
            selection_order=listing['selection_order'],selection_rule='first_downstream_pass_not_forward_KL_ranking')
    write_json(output/'qualification.json',result)
    return result
