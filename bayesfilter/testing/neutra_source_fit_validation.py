"""Development-only fixed-map validation through the shared public HMC APIs.

Exact mixture information is used by references/diagnostics, not the tuner.
No learner, local transition kernel, or mass adaptation is implemented here.
Verified members delegate to run_sequential_neutra_hmc in the shared authority.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
    FixedTransportHMCKernelTuningConfig, tune_fixed_transport_hmc_kernel,
)
from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig
from bayesfilter.inference.hmc_candidate_set_execution import HMCCandidateExecutionConfig
from bayesfilter.inference.hmc_candidate_set_retained import build_retained_bound_hmc_archive_runner_from_candidate_set_result
from bayesfilter.inference.hmc_verification import HMCAcceptancePolicy
from bayesfilter.inference.neutra_hmc import NEUTRA_SEQUENTIAL_HMC_POLICY_ID, SequentialNeuTraHMCConfig
from bayesfilter.inference.hmc_posterior_assessment import HMCPosteriorAssessmentPolicy, assess_posterior
from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
from bayesfilter.inference.hmc_precision import HMCPrecisionPolicy, HMCPrecisionTarget, mean_precision
from bayesfilter.runtime.execution_budget import execution_budget, execution_budget_available, ExecutionBudgetExceeded
from bayesfilter.testing.neutra_generic_targets import ExactTargetEvaluator
from bayesfilter.testing.neutra_scientific_campaign import write, save_tensor, flow_config
from bayesfilter.testing.neutra_warm_start_qualification import verified_member_order

F64=tf.float64
PLAN='docs/plans/bayesfilter-neutra-development-hmc-2026-10-05.md'
GAUSSIAN_CONTROL={'kind':'gaussian','mean':[0.,0.],'covariance':[[1.,0.],[0.,1.]]}


def identity_control_payload(evaluator,seed):
    from bayesfilter.inference.neutra_transport import NeuTraTransport
    # Zero IAF shift/log-scale gives identity. Zeroing a weight-normalized
    # cMADE conditioner instead divides by zero; this control is architectural
    # and must not inherit the learned student's default family.
    trainable=NeuTraTransport(flow_config(evaluator.dimension,seed,width=4,kind='iaf'))
    for variable in trainable.trainable_variables:
        variable.assign(tf.zeros_like(variable))
    identity=hashlib.sha256(b'configured_identity_gaussian_development_control').hexdigest()
    return trainable.frozen_payload(target_signature=evaluator.target.signature,training_state_hash=identity)


def development_protocol(seed,label,*,control=False):
    """Prospective hypotheses; target-specific pilots must measure each pair."""
    policy=HMCAcceptancePolicy()
    execution=HMCCandidateExecutionConfig(measurement_num_results=128,verification_num_results=128,
        num_warmup_steps=0,seed=(seed,6301),acceptance_policy=policy,target_status_trace_policy='none',
        use_xla=True,chain_mode='batched',chunk_max_results=256,reuse_leapfrog_graphs=True)
    leapfrogs=(3,5,9,13,18,25)
    search=HMCControllerConfig(primary_l_grid=leapfrogs,initial_epsilon=.1,pilot_enabled=True,
        evidence_rungs=(1,2,4),refinement_rounds=1,max_candidates=36,total_budget_units=120,
        repair_reserve_units=20,max_wall_time_seconds=300. if control else 900.)
    tuning=FixedTransportHMCKernelTuningConfig(initial_step_size=.1,maximum_candidate_step_size=2.,
        leapfrog_grid=leapfrogs,target_scope=label,fixed_grid_max_attempts=8,use_xla=True)
    return tuning,search,execution


class DevelopmentAdapter:
    def __init__(self,target,label):
        self.target=target
        self.label=label
        self.parameter_dim=target.parameter_dim

    def adapter_signature(self):
        return self.target.signature

    def value_score_capability(self):
        return ValueScoreCapability(value_score_authority='graph_native',xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,runtime_backend='native_batched_tf_analytic_benchmark',
            target_scope=self.label,score_provenance='autodiff of ExactTargetEvaluator batched density',
            evidence_path='tests/test_neutra_source_fit_validation.py',
            nonclaims=('development capability; actual target-specific numerical admission required',))

    def log_prob_and_grad(self,x):
        x=tf.convert_to_tensor(x,F64)
        scalar=x.shape.rank==1
        v,g,_=self.target.value_score(x[None] if scalar else x)
        return (v[0],g[0]) if scalar else (v,g)

    def log_prob(self,x):
        return self.log_prob_and_grad(x)[0]


class DevelopmentQuantities:
    """Unique evaluator quantities with explicit physical meanings."""
    def __init__(self,evaluator):
        self.evaluator=evaluator
        d=evaluator.dimension
        self.physical_names=tuple(f'x{j}' for j in range(d))
        if evaluator.specification['kind']=='gaussian':
            self.names=tuple(f'standardized_second_{j}' for j in range(d))+tuple(f'tail_abs_z{j}_gt2' for j in range(d))
            self.indices=tuple(range(d,3*d))
            self.binary_names=self.names[d:]
        else:
            k=len(evaluator.specification['weights'])
            names=[f'responsibility_{i}' for i in range(k)]
            indices=list(range(k))
            names.extend(f'component_{i}_first_{j}' for i in range(k) for j in range(d))
            indices.extend(range(k,k+k*d))
            offset=k+k*d
            for i in range(k):
                for j in range(d):
                    for m in range(j,d):
                        names.append(f'component_{i}_second_{j}_{m}')
                        indices.append(offset+i*d*d+j*d+m)
            names.extend(f'component_{i}_radial_tail' for i in range(k))
            indices.extend(range(offset+k*d*d,offset+k*d*d+k))
            self.names=tuple(names)
            self.indices=tuple(indices)
            self.binary_names=()
        self.all_names=self.physical_names+self.names
        self.diagnostic_names=(self.names if evaluator.specification['kind']=='gaussian' else
            tuple(f'responsibility_log_odds_{i}' for i in range(k))+self.names[k:])
        self.all_diagnostic_names=self.physical_names+self.diagnostic_names
        self.program=tf.function(self._features,input_signature=[tf.TensorSpec([None,d],F64)],
                                 jit_compile=True,autograph=False)
        self.diagnostic_program=tf.function(self._diagnostics,input_signature=[tf.TensorSpec([None,d],F64)],
                                            jit_compile=True,autograph=False)

    def _features(self,rows):
        extra=tf.gather(self.evaluator._features(rows),self.indices,axis=1)
        return tf.concat((rows,extra),axis=1)

    def _diagnostics(self,rows):
        features=self._features(rows)
        if self.evaluator.specification['kind']=='gaussian':
            return features
        logs=self.evaluator._component_logs(rows)
        k=len(self.evaluator.specification['weights'])
        others=tf.where(tf.eye(k,dtype=tf.bool)[None,:,:],tf.constant(float('-inf'),F64),logs[:,None,:])
        log_odds=logs-tf.reduce_logsumexp(others,axis=2)
        d=self.evaluator.dimension
        return tf.concat((rows,log_odds,features[:,d+k:]),axis=1)

    def values(self,draws,*,diagnostic=False):
        shape=tuple(draws.shape)
        program=self.diagnostic_program if diagnostic else self.program
        return tf.reshape(program(tf.reshape(draws,[-1,self.evaluator.dimension])),
                          (*shape[:-1],len(self.all_names)))

    def additional(self,draws):
        values=self.values(draws,diagnostic=True)[...,self.evaluator.dimension:]
        return {name:values[...,i] for i,name in enumerate(self.diagnostic_names)}

    def policy(self):
        precision=HMCPrecisionPolicy(tuple(HMCPrecisionTarget(name,mcse_sd_ratio_max=.03)
                                          for name in self.all_diagnostic_names))
        return HMCPosteriorAssessmentPolicy(retained_bulk_ess_min=400,retained_tail_ess_min=400,
            precision=precision,quantities_id='source_fit_development_log_odds_rank_features_v2',
            binary_quantity_names=self.binary_names)

    def original_mean_precision(self,draws):
        precision=mean_precision(self.values(draws))
        return {'passed':bool(tf.reduce_all(precision['valid']&(precision['mcse_sd_ratio']<=.03))),
                'names':self.all_names,'mcse_sd_ratio':precision['mcse_sd_ratio'],
                'role':'mean precision of original physical quantities including mode probabilities'}

    def reference_check(self,draws,reference):
        reference=self.program(reference)
        values=self.values(draws)
        expected=tf.reduce_mean(reference,axis=0)
        variance=tf.math.reduce_variance(reference,axis=0)
        precision=mean_precision(values)
        tolerance=4*tf.sqrt(tf.square(precision['mcse'])+variance/tf.cast(tf.shape(reference)[0],F64))+.03*tf.sqrt(variance)
        error=tf.abs(precision['estimate']-expected)
        precision_ok=tf.reduce_all(precision['valid']&(precision['mcse_sd_ratio']<=.03))
        return {'passed':bool(precision_ok&tf.reduce_all(tf.math.is_finite(error)&tf.math.is_finite(tolerance)&(error<=tolerance))),
            'names':self.all_names,'estimate':precision['estimate'],'reference_mean':expected,
            'mcse':precision['mcse'],'error':error,'tolerance':tolerance,
            'mean_precision_passed':bool(precision_ok),'mcse_sd_ratio':precision['mcse_sd_ratio'],
            'rule':'4 combined standard errors plus 0.03 reference SD; operational, not simultaneous coverage',
            'reference_rows':int(reference.shape[0])}


def prepare_reference(evaluator,seed,target_index,output,*,control=False):
    jobs=[('starts',4,[31000+target_index,6101]),('reference',32768,[seed,6201]),
          ('iid-control',4*4096,[31000+target_index,6202]),
          ('assessment-reference',32768,[31000+target_index,6203])]
    def generate(job):
        name,count,key=job
        with tf.device('/CPU:0'):
            sample=tf.function(lambda s:evaluator.sample(count,s),
                input_signature=[tf.TensorSpec([2],tf.int32)],jit_compile=True,autograph=False)
            value=sample(tf.constant(key,tf.int32))
            if 'CPU' not in value.device:
                raise ValueError('exact development sample generation must execute on CPU')
            digest=save_tensor(output/(name+'.tensor'),value)
        return name,value,{'rows':count,'seed':key,'sha256':digest,'device':value.device}
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(generate,jobs))
    write(output/'data-provenance.json',{'cpu_workers':2,'target_signature':evaluator.target.signature,
        'banks':{name:meta for name,_,meta in rows},'starts_role':'favorable_exact_target_benchmark_control_no_discovery_claim'})
    return {name:value for name,value,_ in rows}


def validate_development_map(specification,profile,seed,output):
    """One map's own tuning and sequential assessment, with reference isolation."""
    output=Path(output)
    started=time.monotonic()
    control=profile.get('control',False)
    evaluator=ExactTargetEvaluator(specification,jit_compile=True)
    quantities=DevelopmentQuantities(evaluator)
    data=prepare_reference(evaluator,seed,profile.get('target_index',0),output,control=control)
    iid=tf.reshape(data['iid-control'],[4096,4,evaluator.dimension])
    screen=assess_posterior(iid,quantities.physical_names,policy=quantities.policy(),stage='retained',
        rhat=rank_normalized_split_rhat_summary(iid,rhat_max=1.01),quantities_fn=quantities.additional)
    reference=quantities.reference_check(iid,data['assessment-reference'])
    wrong=quantities.reference_check(iid+tf.constant(1.,F64),data['assessment-reference'])
    write(output/'assessment-controls.json',{'iid_sequential':screen,'iid_reference':reference,'wrong_location':wrong,
        'fresh_posterior_reference_used':False})
    if not (screen['passed'] and reference['passed'] and not wrong['passed']):
        raise ValueError('common posterior/reference control failed')
    if control:
        if specification!=GAUSSIAN_CONTROL:
            raise ValueError('identity control requires the declared standard Gaussian')
        frozen=identity_control_payload(evaluator,seed)
    else:
        parent=Path(profile['parent'])
        manifest=json.loads((parent/'manifest.json').read_text())
        file=parent/'lr-lower-frozen.json'
        if hashlib.sha256(file.read_bytes()).hexdigest()!=manifest['artifact_sha256'][file.name]:
            raise ValueError('frozen map checksum mismatch')
        parent_result=json.loads((parent/'result.json').read_text())
        if not parent_result['endpoints']['lr-lower']['passed']:
            raise ValueError('map failed its required preliminary screen')
        frozen=json.loads(file.read_text())
    write(output/'frozen.json',frozen)
    flow=load_frozen_neutra_artifact(frozen,expected_target_signature=evaluator.target.signature).transport
    initial=flow.inverse_theta_to_z_batch(data['starts'])
    roundtrip=float(tf.reduce_max(tf.abs(flow.forward_batch(initial)-data['starts'])))
    if not roundtrip<1e-8:
        raise ValueError('development physical-start roundtrip failed')
    values,scores,valid=evaluator.target.value_score(data['starts'])
    if not bool(tf.reduce_all(valid)&tf.reduce_all(tf.math.is_finite(initial))):
        raise ValueError('invalid development physical start')
    write(output/'initialization.json',{'roundtrip_error':roundtrip,'latent':initial,
        'physical':data['starts'],'log_density':values,'score_norm':tf.linalg.norm(scores,axis=1),
        'source':'independent exact target starts; no mode-discovery claim'})
    label=profile.get('target_label','gaussian_control')
    adapter=DevelopmentAdapter(evaluator.target,label)
    deadline=started+profile['worker_wall_limit']-60.
    tuning,search,execution=development_protocol(seed,label,control=control)
    write(output/'effective-protocol.json',{'plan':PLAN,'tuning':tuning.payload(),
        'search':search.payload(),'execution':execution.payload(),'assessment':quantities.policy().payload(),
        'mass':'fixed_identity_latent','reference_consulted_during_selection':False,'jit_compile':True})
    result={'status':'development_not_confirmed','target_signature':evaluator.target.signature,
        'transport_hash':frozen['transport_hash'],'seed':seed,'control':control,
        'policy_id':NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        'scientific_promotion':False,'method_ranking':'not_established','heldout_consumed':False,
        'passed':False,'member_screen':[],'plan':PLAN}
    try:
        with execution_budget(deadline=deadline):
            run=tune_fixed_transport_hmc_kernel(base_adapter=adapter,fixed_transport=flow,initial_position=initial,
                config=tuning,search_config=search,execution_config=execution,frozen_transport_payload=frozen,
                output_dir=output/'tuning',target_lineage=specification,
                source_paths=(__file__,str(Path(__file__).with_name('neutra_generic_targets.py'))))
            result.update(tuning_completion=run.result.completion_status,
                          verified_members=list(run.result.verified_candidate_ids),
                          tuning_wall_seconds=time.monotonic()-started)
            write(output/'development-progress.json',result)
            order=verified_member_order(run.result,3)
            result['member_order']=order
            for ordinal,candidate_id in enumerate(order):
                if not execution_budget_available():
                    result['reason']='development_wall_budget_exhausted'
                    break
                member=build_retained_bound_hmc_archive_runner_from_candidate_set_result(
                    candidate_set_result=run.result,candidate_id=candidate_id,retained_binding=run.adapter._execution_binding)
                directory=output/f'member-{ordinal}'
                directory.mkdir(exist_ok=False)
                member.export(directory/'verified-member.json')
                sequential=SequentialNeuTraHMCConfig(step_size=member.step_size,num_leapfrog_steps=member.num_leapfrog_steps,
                    warmup_seed=(seed,6401+10*ordinal),retained_seed=(seed,6501+10*ordinal),
                    assessment_policy=quantities.policy(),jit_compile=True)
                write(directory/'sequential-protocol.json',asdict(sequential))
                def archive(**kwargs):
                    tag='cumulative' if kwargs.get('cumulative') else str(kwargs['chunk_index'])
                    path=directory/f'{kwargs["stage"]}-{tag}.tensor'
                    checksum=save_tensor(path,kwargs['model_samples'])
                    return {'path':str(path),'sha256':checksum,
                        'warmup_excluded_from_estimates':kwargs['stage']=='warmup'}
                posterior=member.run_sequential(config=sequential,parameter_names=quantities.physical_names,
                    quantities_fn=quantities.additional,archive_callback=archive,
                    retained_diagnostic_fn=quantities.original_mean_precision,
                    budget_check=lambda _:execution_budget_available())
                public={k:v for k,v in posterior.items() if not k.startswith('private_')}
                write(directory/'posterior.json',public)
                for key in ('private_retained_raw','private_warmup_raw'):
                    save_tensor(directory/(key+'.tensor'),posterior[key])
                result['member_screen'].append({'candidate_id':candidate_id,'passed':bool(posterior['passed']),
                    'decision':posterior['decision'],'hard_vetoes':posterior['hard_vetoes'],
                    'equilibration_status':posterior['equilibration_status'],
                    'precision_status':posterior['precision_status'],
                    'step_size':member.step_size,'leapfrogs':member.num_leapfrog_steps,
                    'warmup_results':posterior['warmup_results_per_chain'],'retained_results':posterior['retained_results_per_chain'],
                    'posterior':str(directory/'posterior.json')})
                write(output/'development-progress.json',result)
                if posterior['passed']:
                    agreement=quantities.reference_check(posterior['private_retained_raw'],data['reference'])
                    write(output/'fresh-reference-check.json',agreement)
                    result.update(heldout_consumed=True,reference_passed=agreement['passed'],
                        passed=agreement['passed'],selected_candidate_id=candidate_id,
                        status='development_screen_passed' if agreement['passed'] else 'development_reference_failed')
                    break
            if not order:
                result['reason']='no_verified_kernel_within_declared_search'
    except ExecutionBudgetExceeded:
        result['reason']='development_wall_budget_exhausted'
    result['wall_seconds']=time.monotonic()-started
    write(output/'result.json',result)
    return result
