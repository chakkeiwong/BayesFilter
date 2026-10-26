"""Broad source-bound MacroFinance diagnostic; no consumer/posterior admission."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
PLAN = 'docs/plans/bayesfilter-modern-hmc-broad-consumer-continuation-2026-10-25.md'
L_GRID = (3, 5, 9, 13, 18, 25)
CANDIDATE_CAP = 12
ROOT_EPSILON = .4
EPSILON_DOMAIN = (.025, 1.6)
PRICE_MARGIN = 1.5
RELOAD_RESERVE_SECONDS = 900.
CLEANUP_RESERVE_SECONDS = 120.


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+chr(10))
    tmp.replace(path)


def affordability(rows, remaining_seconds):
    """Price complete trials; reject incomplete or misleading short-call quotes."""
    if not math.isfinite(remaining_seconds) or remaining_seconds <= 0:
        raise ValueError('finite positive remaining allocation required')
    expected = {(length, repeat) for length in L_GRID for repeat in (0, 1)}
    seen, times = set(), {}
    for row in rows:
        key = (row['L'], row['repeat'])
        if key not in expected or key in seen:
            raise ValueError('pricing trial coverage differs from declared design')
        seen.add(key)
        if (row.get('epsilon') != ROOT_EPSILON or row.get('transition_profile') != 'hmc_uniform_duration'
                or row.get('discarded_prefix') != 0 or row.get('chain_count') != 4):
            raise ValueError('price scope differs from actual root execution')
        if row['num_results'] != 64 or row['health_failures']:
            raise ValueError('healthy full-horizon trials required for price')
        seconds = row['elapsed_seconds']
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('invalid full-trial time')
        times.setdefault(row['L'], []).append(seconds)
    if seen != expected:
        raise ValueError('incomplete full-trial price')
    projection = 64 * sum(max(times[length]) for length in L_GRID)
    allowance = remaining_seconds - RELOAD_RESERVE_SECONDS - CLEANUP_RESERVE_SECONDS
    return dict(root_native_projection_seconds=projection,
        conservative_root_allowance_seconds=PRICE_MARGIN*projection,
        operational_margin=PRICE_MARGIN, available_tuning_seconds=max(0., allowance),
        affordable=allowance >= PRICE_MARGIN*projection,
        basis='64 measurement-plus-verification trials per L times slower of two full-trial calls',
        priced_scope='six .4 roots: measurement and verification',
        optional_candidates_priced=False, full_search_funded=False,
        limit='root-only operational projection; optional epsilon repairs, diagnostics and contention remain unpriced')


def reviewed_design(raw):
    """Validate explicit prospective choices; historical configurations stay readable."""
    from bayesfilter.inference.hmc_discarded_epsilon_proposals import DiscardedEpsilonSearch
    from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy, replicated_evidence_preflight
    if raw.get('schema') != 'bayesfilter.broad_prospective_design.v1':
        raise ValueError('explicit prospective design required')
    p = HMCReplicatedAcceptancePolicy.from_payload(raw['policy'])
    pilot = DiscardedEpsilonSearch(**raw['discarded_proposals'])
    if tuple(l for l, _ in pilot.epsilon_by_l) != L_GRID:
        raise ValueError('prospective proposals must preserve every broad L root')
    if len(L_GRID)*pilot.attempts_per_l > p.max_candidates:
        raise ValueError('candidate cap must retain every healthy proposal')
    rungs = tuple(raw['evidence_rungs'])
    from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
    quote = replicated_evidence_preflight(p, evidence_rungs=rungs,
        candidate_cap=p.max_candidates, leapfrog_steps=(max(L_GRID),)*p.max_candidates,
        transition_profile=HMCTransitionProfile('hmc','uniform_duration'))
    if not raw.get('numeric_provenance'):
        raise ValueError('prospective numerical choices need provenance')
    return p, pilot, rungs, quote


def full_design_price(rows, remaining_seconds, design):
    p, pilot, rungs, quote = reviewed_design(design)
    if p.trial_num_results != 64 or p.discarded_prefix != 0 or p.chain_count != 4:
        raise ValueError('pricing measurements do not cover this horizon/prefix/start design')
    root = affordability(rows, remaining_seconds)
    slowest = max(row['elapsed_seconds'] for row in rows)
    trials = p.repetition_target(rungs[-1])
    # Each nominated candidate retains the measurement and fresh-verification
    # cap. Candidate-specific pilots are budgeted separately at their wall cap.
    projected = 2*p.max_candidates*trials*slowest
    required = PRICE_MARGIN*projected + pilot.max_wall_seconds + RELOAD_RESERVE_SECONDS + CLEANUP_RESERVE_SECONDS
    return dict(root_price=root, maximum_qualification_native_seconds=projected,
        total_required_seconds=required, affordable=remaining_seconds >= required,
        full_search_funded=remaining_seconds >= required,
        qualification_trials_per_stage=trials, candidate_cap=p.max_candidates,
        maximum_gradient_work=2*sum(quote['maximum_stage_gradient_work_by_candidate']),
        role='conservative operational projection, not a performance guarantee',
        limitation='fresh per-scope full-trial timing required; contention and new epsilons may change time')


def policy():
    from diagnose_modern_hmc_macrofinance_public import policy as baseline
    return replace(baseline(), max_candidates=CANDIDATE_CAP)


def controller_config(bound, remaining, *, design=None, epsilon_by_l=None):
    from bayesfilter.inference import HMCControllerConfig
    from bayesfilter.inference.hmc_acceptance_protocol import replicated_evidence_preflight
    if design is not None:
        p, pilot, rungs, quote = reviewed_design(design)
        if p != bound.config.acceptance_policy:
            raise ValueError('binding and prospective policy differ')
        if epsilon_by_l is None or tuple(l for l, _ in epsilon_by_l) != L_GRID or any(not eps for _, eps in epsilon_by_l):
            raise ValueError('every broad L needs at least one discarded healthy nominee')
        return HMCControllerConfig(primary_l_grid=L_GRID, epsilon_by_l=tuple((l,tuple(e)) for l,e in epsilon_by_l),
            total_budget_units=(2*rungs[-1]+3)*p.max_candidates, repair_reserve_units=p.max_candidates,
            candidate_reserve_units=3, max_candidates=p.max_candidates, evidence_rungs=rungs,
            refinement_rounds=1, epsilon_refinement_factors=(1/pilot.factor,pilot.factor),
            explore_failed_intervals=True,
            max_gradient_work=2*sum(quote['maximum_stage_gradient_work_by_candidate']),
            max_wall_time_seconds=remaining, replicated_acceptance_policy=p), quote
    quote = replicated_evidence_preflight(bound.config.acceptance_policy, evidence_rungs=(1,),
        candidate_cap=CANDIDATE_CAP, leapfrog_steps=L_GRID+(max(L_GRID),)*6,
        transition_profile=bound.config.transition_profile)
    return HMCControllerConfig(primary_l_grid=L_GRID, initial_epsilon=ROOT_EPSILON,
        total_budget_units=48, repair_reserve_units=12, candidate_reserve_units=3,
        max_candidates=CANDIDATE_CAP, evidence_rungs=(1,),
        refinement_rounds=1, epsilon_refinement_factors=(1.25,),
        max_gradient_work=quote['initial_cohort_mandatory_gradient_work'],
        max_wall_time_seconds=remaining, replicated_acceptance_policy=bound.config.acceptance_policy), quote


def binding(adapter, inputs, paths, *, design=None):
    import tensorflow as tf
    from bayesfilter.inference import HMCCandidateExecutionConfig, PrecomputedMassArtifact
    from bayesfilter.inference.hmc_candidate_set_execution import bind_hmc_candidate_set_execution
    from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
    covariance = tf.constant(inputs['initial_covariance'], tf.float64)
    center = tf.constant(inputs['initial_position'], tf.float64)
    if covariance.shape != (9, 9) or center.shape != (9,):
        raise ValueError('nine-coordinate MacroFinance geometry required')
    tf.debugging.assert_near(covariance, tf.transpose(covariance), atol=1e-12, rtol=1e-12)
    tf.debugging.assert_positive(tf.linalg.eigvalsh(covariance))
    factor = tf.linalg.cholesky(covariance)
    offsets = tf.repeat(tf.constant([[-.15],[-.05],[.05],[.15]], tf.float64), 9, axis=1)
    starts = center + offsets @ tf.transpose(factor)
    p = policy() if design is None else reviewed_design(design)[0]
    config = HMCCandidateExecutionConfig(measurement_num_results=64, verification_num_results=64,
        num_warmup_steps=0, seed=(20261025,1901), acceptance_policy=p,
        target_status_trace_policy='per_chain_step', use_xla=True, chain_mode='batched',
        chunk_max_results=64, transition_profile=HMCTransitionProfile('hmc','uniform_duration'))
    mass = PrecomputedMassArtifact(position=center, factor=factor, covariance=covariance,
        adapter_signature=adapter.adapter_signature(), position_role='reference_center',
        covariance_source='saved P18 covariance hypothesis; no refit or promotion')
    return bind_hmc_candidate_set_execution(adapter=adapter, initial_position=starts,
        target_scope='one-country-affine-phase9-synthetic', target_lineage={
            'dataset_id':inputs['design']['dataset_id'], 'inputs':inputs,
            'prior':'chart Gaussian scale 0.35', 'geometry_role':'supplied hypothesis'},
        config=config, source_paths=paths, scope_id='macrofinance-broad-gpu-diagnostic',
        search_id='20261025-broad', epsilon_domain=EPSILON_DOMAIN, repair_factor=1.25,
        max_repairs_per_family=1, mass_artifact=mass)


def price(bound, output, mark):
    import tensorflow as tf
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCTuningCandidateRecord
    from bayesfilter.inference.hmc_candidate_set_execution import _tensor_payload, _trace_payload
    rows = []
    for index, length in enumerate(L_GRID):
        candidate = HMCTuningCandidateRecord.create(bound.scope, leapfrog_steps=length,
            epsilon=ROOT_EPSILON, creation_ordinal=index+1)
        for repeat in (0,1):
            seed=(20261025,1101+2*index+repeat)
            mark('full_trial_price', dict(L=length,repeat=repeat,completed_calls=len(rows)))
            started=time.monotonic()
            result=bound._run(candidate,bound.initial_active_state,64,seed)
            failures=list(bound.health_failures(bound.initial_active_state,result.samples,result.trace))
            if 'device:GPU:' not in result.samples.device or not result.metadata['use_xla']:
                raise ValueError('pricing did not execute GPU/XLA')
            row=dict(L=length,repeat=repeat,num_results=64,seed=seed,epsilon=ROOT_EPSILON,
                transition_profile='hmc_uniform_duration',discarded_prefix=0,chain_count=4,
                elapsed_seconds=time.monotonic()-started,health_failures=failures,
                native_seconds=result.metadata['sample_chain_call_s'],
                samples_device=result.samples.device,
                mean_acceptance_reporting_only=float(tf.reduce_mean(tf.exp(tf.minimum(result.trace['log_accept_ratio'],0.)))))
            write(output/f'price-L{length}-repeat{repeat}.json',dict(**row,
                samples=_tensor_payload(result.samples),trace=_trace_payload(result.trace)))
            rows.append(row)
            write(output/'price-progress.json',dict(rows=rows))
            if failures:raise ValueError('full-trial pricing health failure; inspect saved trace')
    return dict(rows=rows,qualification_evidence=False)


def tune(adapter, bound, output, remaining, *, design=None, price_rows=None):
    from bayesfilter.inference import tune_hmc_kernel, export_hmc_candidate_retained_runners
    from bayesfilter.testing.inference_validation.modern_bridge_audit import audit_native_evidence
    from scripts.diagnose_modern_hmc_evidence_design import resolution_screen
    prospective_started = time.monotonic()
    if design is not None:
        p, pilot, rungs, _ = reviewed_design(design)
        screen = resolution_screen(p, evidence_rungs=rungs)
    else:
        screen = resolution_screen(bound.config.acceptance_policy)
    write(output/'evidence-design-screen.json',screen)
    if not screen['preferred_center_resolvable']:
        raise ValueError('evidence design cannot resolve its preferred center; require fresh costed design')
    epsilon_by_l = None
    if design is not None:
        if price_rows is None:
            raise ValueError('fresh complete price required before discarded proposals or qualification')
        priced = full_design_price(price_rows, remaining, design)
        write(output/'full-design-price.json', priced)
        if not priced['full_search_funded']:
            raise ValueError('full proposal and qualification design is under-budgeted')
        from bayesfilter.inference.hmc_discarded_epsilon_proposals import propose_discarded_epsilon_candidates
        proposed = propose_discarded_epsilon_candidates(bound, pilot, output_dir=output/'discarded-proposals')
        epsilon_by_l = proposed['epsilon_by_l']
        remaining -= time.monotonic()-prospective_started
    search, quote = controller_config(bound, remaining, design=design, epsilon_by_l=epsilon_by_l)
    write(output/'design.json',dict(policy=bound.config.acceptance_policy.payload(),
        search=search.payload(),work_price=quote,root_epsilon_hypothesis=ROOT_EPSILON))
    run=tune_hmc_kernel(adapter=adapter, initial_position=bound.initial_active_state,
        config=search,candidate_set_adapter=bound.typed_adapter,output_dir=output/'tuning')
    audit=audit_native_evidence(run.result,bound)
    write(output/'native-audit.json',audit)
    paths=export_hmc_candidate_retained_runners(candidate_set_result=run.result,
        retained_binding=bound,output_dir=output/'members')
    if set(paths)!=set(run.result.verified_candidate_ids):
        raise ValueError('verified set changed during export')
    write(output/'members.json',{cid:str(path) for cid,path in paths.items()})
    return dict(completion_status=run.result.completion_status,final_status=run.result.final_status,
        candidate_states=dict(run.result.candidate_states),verified=sorted(paths),
        candidate_count=len(run.result.candidates),repair_count=len(run.result.repair_actions),
        all_verified_members_exported=True,native_audit=audit,no_ranking=True)


def reload(adapter, source, output, mark, *, seed_root=(20261025,2001)):
    import tensorflow as tf
    from bayesfilter.inference import load_hmc_candidate_retained_runners
    from bayesfilter.inference.hmc_candidate_set_execution import _tensor_from_payload
    paths=read(source/'members.json')
    members=load_hmc_candidate_retained_runners(paths.values(),adapter=adapter)
    if set(members)!=set(paths):raise ValueError('reload membership changed')
    rows=[]
    for index,(cid,member) in enumerate(sorted(members.items())):
        saved=read(paths[cid])
        tf.debugging.assert_equal(member.initial_active_state,_tensor_from_payload(saved['verified_endpoint']))
        if member.claim_eligible or member.posterior_convergence_authority:
            raise ValueError('diagnostic acquired unsupported authority')
        seed=(seed_root[0],seed_root[1]+index)
        if seed in member._tuning_seeds():raise ValueError('reload seed reused')
        mark('reload_member',dict(candidate_id=cid,completed_members=len(rows)))
        result=member.run(num_results=8,seed=seed,output_dir=output/f'member-{index}')
        finite=bool(tf.reduce_all(tf.math.is_finite(result['samples'])))
        if not finite or 'device:GPU:' not in result['samples'].device:
            raise ValueError('reload continuation invalid')
        rows.append(dict(candidate_id=cid,seed=seed,endpoint_preserved=True,finite_continuation=finite,
            archive=result['archive_path']))
    return dict(verified=sorted(members),all_verified_members_reloaded=True,rows=rows)


def validated_price_source(source, current_sources, inputs):
    if source is None:
        raise ValueError('tuning requires --price-source')
    manifest=read(source/'manifest.json'); result=read(source/'result.json')
    if result.get('status') != 'completed' or manifest.get('inputs_sha256') != digest(inputs):
        raise ValueError('price incomplete or data scope changed')
    for name, old in manifest['source_sha256'].items():
        if name not in current_sources or current_sources[name] != old:
            raise ValueError('price source changed: '+name)
    return result['execution']['rows']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('price','tune','reload'),required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--wall-seconds',type=float,required=True)
    parser.add_argument('--design',type=Path)
    parser.add_argument('--price-source',type=Path)
    parser.add_argument('--gpu-evidence',type=Path)
    args=parser.parse_args()
    design = read(args.design) if args.design else None
    if args.mode in ('price','tune') and design is None:
        parser.error('fresh price/tune requires --design; historical single-rung policy is retired')
    if design is not None:
        reviewed_design(design)
        if args.gpu_evidence is None: parser.error('fresh design requires --gpu-evidence for changed numerical source')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report=dict(status='running',mode=args.mode,plan=PLAN,posterior_qualification=False,
        external_consumer_admitted=False,default_promoted=False,no_ranking=True)
    from diagnose_modern_hmc_macrofinance import CONSUMER
    paths=list((ROOT/'bayesfilter').rglob('*.py'))+[Path(__file__).resolve(),args.inputs.resolve()]
    paths += [ROOT/'scripts'/name for name in ('diagnose_modern_hmc_macrofinance_public.py',
        'diagnose_modern_hmc_macrofinance.py','modern_hmc_macrofinance_gpu_support.py')]
    paths += ([args.design.resolve()] if args.design else [])
    paths += [CONSUMER/name for name in ('one_country_affine_macro_phase9_hmc.py',
        'one_country_affine_macro_governance.py','one_country_affine_macro_transition_2x2.py',
        'one_country_affine_macro_prior_measurement_contract.py')]
    before={str(p):digest(p) for p in paths}
    manifest=dict(command=[sys.executable,*sys.argv],started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=before,inputs_sha256=digest(args.inputs),plan=PLAN,
        environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH',
            'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS')},
        seed_namespaces=dict(price=[20261025,1101,1112],qualification=[20261025,1901],reload=[20261025,2001,2012]),
        wall_seconds=args.wall_seconds,source_worktree=str(ROOT),data_version='saved P18 synthetic 48-period fixture')
    write(args.output/'manifest.json',manifest)
    def mark(stage,details=None):
        write(args.output/'progress.json',dict(stage=stage,details=details,elapsed_seconds=time.monotonic()-started))
    try:
        if args.mode in ('price','tune'):
            path=args.output/'evidence-design-preflight.json'
            subprocess.run([sys.executable,str(ROOT/'scripts/diagnose_modern_hmc_evidence_design.py'),
                '--output',str(path),'--design',str(args.design.resolve())],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1'),check=True,
                timeout=min(120.,args.wall_seconds),stdout=subprocess.DEVNULL)
            if not read(path)['current_design']['preferred_center_resolvable']:
                raise ValueError('prospective evidence design cannot resolve preferred center')
        if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')!='true' or not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-'):
            raise ValueError('GPU UUID and memory growth required before TensorFlow import')
        sys.path.insert(0,str(CONSUMER))
        from bayesfilter.runtime.gpu_provenance import selected_nvidia_gpu,query_nvidia_smi_gpus
        manifest['physical_gpu']=selected_nvidia_gpu(query_nvidia_smi_gpus())
        manifest['gpu_trust_basis']='trusted_escalated_gpu_execution'
        from diagnose_modern_hmc_macrofinance_public import build
        mark('build_unchanged_measured_adapter')
        adapter,memory=build(read(args.inputs),device='GPU',gpu_evidence=args.gpu_evidence)
        import tensorflow as tf
        import tensorflow_probability as tfp
        manifest.update(memory_policy=memory,tensorflow=tf.__version__,tensorflow_probability=tfp.__version__,
            dtype='float64',tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),use_xla=True)
        write(args.output/'manifest.json',manifest)
        from bayesfilter.runtime.execution_budget import execution_budget,execution_progress
        with execution_budget(deadline=started+args.wall_seconds),execution_progress(mark):
            if args.mode=='reload':
                if args.source is None:raise ValueError('reload source required')
                report['execution']=reload(adapter,args.source,args.output,mark)
            else:
                bound=binding(adapter,read(args.inputs),paths,design=design)
                report['binding_hash']=bound.binding_hash
                report['execution']=(price(bound,args.output,mark) if args.mode=='price' else
                    tune(adapter,bound,args.output,args.wall_seconds-(time.monotonic()-started),
                        design=design,price_rows=validated_price_source(args.price_source, before, args.inputs)))
        report['sources_unchanged']=all(digest(p)==h for p,h in before.items())
        if not report['sources_unchanged']:raise ValueError('numerical sources changed during execution')
        report['allocator_bytes']=tf.config.experimental.get_memory_info('GPU:0')
        from scripts.modern_hmc_campaign_status import completion_status
        report['status']=(completion_status(report['execution']['completion_status'])
            if args.mode=='tune' else 'completed')
    except Exception as exc:
        report.update(status='error',error=repr(exc),traceback=traceback.format_exc())
    report.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
    write(args.output/'result.json',report)
    print(json.dumps({k:report[k] for k in ('status','elapsed_seconds')}),flush=True)
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
