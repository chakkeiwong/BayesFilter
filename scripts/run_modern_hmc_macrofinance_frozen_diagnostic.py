"""Bounded actual-consumer diagnostic; no posterior or public-admission authority.

Composes existing numerical binding, controller, evidence, export and recovery.
The frozen cohort is smaller than its unchanged statistical family allocation.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'),
    os.environ.get('BAYESFILTER_DIAGNOSTIC_CONSUMER_ROOT', '/home/ubuntu/python/MacroFinance')]
PLAN = os.environ.get('BAYESFILTER_DIAGNOSTIC_PLAN',
    'docs/plans/bayesfilter-modern-hmc-diagnostic-extension-2026-10-25.md')
INPUTS = ROOT / 'docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/residual-qualification-20261024-01/macrofinance-inputs.json'
GPU_EVIDENCE = Path(os.environ.get('BAYESFILTER_DIAGNOSTIC_GPU_EVIDENCE',
    str(INPUTS.parent.parent / 'row-isolation-repair-20261025-01/gpu-02/evidence')))


def read(p):
    return json.loads(Path(p).read_text())


def write(p, d):
    p = Path(p)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(d, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pairs(cohort):
    values = tuple((int(l), tuple(float(e) for e in eps)) for l, eps in cohort['epsilon_by_l'])
    if tuple(l for l, _ in values) != (3, 5, 9, 13, 18, 25):
        raise ValueError('frozen cohort must preserve all six L values')
    flat = tuple((l, e) for l, eps in values for e in eps)
    if len(flat) != len(set(flat)) or not flat or any(not math.isfinite(e) or e <= 0 for _, e in flat):
        raise ValueError('invalid frozen nominees')
    if len(flat) != cohort['controller_candidate_cap'] or cohort['statistical_family_cap'] < len(flat):
        raise ValueError('candidate counts differ from frozen cohort')
    if cohort['refinement_rounds'] != 0 or cohort['max_repairs_per_family'] != 0:
        raise ValueError('frozen diagnostic cannot expand its cohort')
    return values


def cohort_price(root_rows, new_rows, cohort, available):
    expected = {(l, e) for l, eps in pairs(cohort) for e in eps}
    # Root L3 contains first compilation; replace it with explicitly warm prices.
    all_rows = [r for r in root_rows if r['L'] != 3] + [r for r in new_rows if not r['cold_call']]
    times = {}
    vetoes = []
    for r in all_rows:
        key = (r['L'], r['epsilon'])
        if key not in expected:
            raise ValueError('price contains an undeclared pair')
        if (r['num_results'], r['chain_count'], r['discarded_prefix'], r['transition_profile']) != (64, 4, 0, 'hmc_uniform_duration'):
            raise ValueError('price differs from frozen execution')
        seconds = r['elapsed_seconds']
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('invalid price')
        times.setdefault(key, []).append(seconds)
        if r['health_failures']:
            vetoes.append(dict(L=key[0], epsilon=key[1], failures=r['health_failures']))
    if set(times) != expected or any(len(v) < 2 for v in times.values()):
        raise ValueError('every frozen pair requires two warm full-horizon prices')
    native = 2 * 128 * sum(max(v) for v in times.values())
    required = 1.5 * native + 120 + 900 + 120
    return dict(schema='bayesfilter.frozen_cohort_price.v1', pairs=[dict(L=l, epsilon=e, full_trial_seconds=max(times[l,e])) for l,e in sorted(expected)],
        native_maximum_seconds=native, operational_margin=1.5, cold_allowance_seconds=120,
        export_reload_seconds=900, closeout_seconds=120, required_seconds=required,
        available_seconds=available, affordable=available >= required, prior_price_vetoes=vetoes,
        statistical_family_cap=cohort['statistical_family_cap'], all_nominees_priced=True,
        full_adaptive_search_funded=False, role='operational estimate; no qualification or runtime guarantee')


def frozen_config(bound, cohort, seconds):
    from bayesfilter.inference import HMCControllerConfig
    from bayesfilter.inference.hmc_acceptance_protocol import replicated_evidence_preflight
    eps = pairs(cohort)
    p = bound.config.acceptance_policy
    if p.max_candidates != cohort['statistical_family_cap'] or p.max_repetitions != 128:
        raise ValueError('statistical policy differs from reviewed frozen design')
    quote = replicated_evidence_preflight(p, evidence_rungs=tuple(cohort['evidence_rungs']),
        candidate_cap=cohort['controller_candidate_cap'], leapfrog_steps=tuple(l for l, es in eps for _ in es),
        transition_profile=bound.config.transition_profile)
    cfg = HMCControllerConfig(primary_l_grid=tuple(l for l, _ in eps), epsilon_by_l=eps,
        total_budget_units=(2*cohort['evidence_rungs'][-1]+3)*cohort['controller_candidate_cap'],
        repair_reserve_units=1, candidate_reserve_units=3, allow_repair_from_free_pool=False,
        max_candidates=cohort['controller_candidate_cap'], evidence_rungs=tuple(cohort['evidence_rungs']),
        refinement_rounds=0, explore_failed_intervals=False,
        max_gradient_work=2*sum(quote['maximum_stage_gradient_work_by_candidate']),
        max_wall_time_seconds=seconds, replicated_acceptance_policy=p)
    return cfg, quote


def frozen_binding(adapter, inputs, paths, design, cohort):
    from scripts.continue_modern_hmc_macrofinance_broad import binding
    from bayesfilter.inference import PrecomputedMassArtifact
    from bayesfilter.inference.hmc_candidate_set_execution import bind_hmc_candidate_set_execution
    base = binding(adapter, inputs, paths, design=design)
    mass = PrecomputedMassArtifact.from_payload(base._spec['layers'][0]['artifact'])
    return bind_hmc_candidate_set_execution(adapter=adapter, initial_position=base.initial_active_state,
        start_coordinates='active', target_scope=base._spec['target_scope'], target_lineage=base._spec['target_lineage'],
        config=replace(base.config, seed=tuple(cohort['qualification_seed'])), source_paths=paths,
        scope_id='macrofinance-frozen-nominee-diagnostic', search_id='20261025-nine-pair',
        epsilon_domain=base.scope.epsilon_domain, repair_factor=base.scope.repair_factor,
        max_repairs_per_family=0, mass_artifact=mass)


def source_paths(campaign):
    isolated = campaign/'isolated-source.json'
    if isolated.exists():
        from scripts.modern_hmc_source_snapshot import validate_snapshot
        paths = validate_snapshot(read(isolated), expected_root=ROOT)
        return [*paths, isolated, campaign/'frozen-cohort.json', campaign/'proposal-design.json'], None
    parent = read(campaign/'price-01/evidence/manifest.json')['source_sha256']
    changed = []
    for name, sha in parent.items():
        observed = digest(name)
        if observed != sha:
            changed.append(dict(path=name, prior_sha256=sha, current_sha256=observed))
    # A source-bound attempt must fail closed when another local task edits a
    # file.  A reviewed reconciliation may authorize a *fresh* binding for a
    # file that is outside the target import path; it never rebinds the old
    # numerical evidence or silently treats the edit as harmless.
    reconciliation_path = campaign/'source-reconciliation.json'
    excluded = set()
    if changed:
        if not reconciliation_path.exists():
            raise ValueError('priced numerical source changed; reviewed reconciliation required')
        reconciliation = read(reconciliation_path)
        excluded = set(reconciliation.get('excluded_paths', ()))
        allowed = tuple(sorted(reconciliation.get('allowed_paths', ())))
        observed = tuple(sorted(item['path'] for item in changed))
        recorded = tuple(sorted(item.get('path') for item in reconciliation.get('changed_paths', ())))
        if (allowed != observed or recorded != observed
                or reconciliation.get('fresh_binding_required') is not True):
            raise ValueError('source reconciliation does not exactly cover current drift')
        by_path = {item['path']: item for item in changed}
        for item in reconciliation.get('changed_paths', ()):
            if (item.get('path') not in by_path
                    or item.get('prior_sha256') != by_path[item['path']]['prior_sha256']):
                raise ValueError('source reconciliation hash does not match current drift')
            # Excluded paths are outside this target's import closure. Their
            # current contents may continue to change under another local
            # study; they are recorded for provenance but never bound.
            if item['path'] not in excluded and item.get('current_sha256') != by_path[item['path']]['current_sha256']:
                raise ValueError('non-excluded source changed after reconciliation')
        if reconciliation.get('old_numerical_evidence_reused') is not False:
            raise ValueError('source drift cannot reuse old numerical evidence')
    else:
        reconciliation = None
    excluded = set() if reconciliation is None else set(reconciliation.get('excluded_paths', ()))
    if reconciliation is not None and excluded != set(reconciliation.get('allowed_paths', ())):
        raise ValueError('source reconciliation exclusion set differs from drift set')
    extra = [Path(__file__).resolve(), campaign/'frozen-cohort.json', campaign/'proposal-design.json',
        reconciliation_path,
        ROOT/'scripts/resume_modern_hmc_macrofinance_public.py', ROOT/'scripts/resume_modern_hmc_bridge.py',
        ROOT/'scripts/resume_modern_hmc_full_model_qualification.py']
    return list(dict.fromkeys([Path(p) for p in parent if p not in excluded] + extra)), reconciliation


def record_controller_outcome(report, result):
    """Retain the primary outcome even when audit or export later fails."""
    report['execution'] = dict(completion_status=result.completion_status,
        final_status=result.final_status, candidate_states=dict(result.candidate_states),
        verified=list(result.verified_candidate_ids), candidate_count=len(result.candidates),
        repair_count=len(result.repair_actions), all_verified_members_exported=False,
        execution_failures=[dict(e) for e in result.accounting_events
                            if e.get('event') == 'execution_failure'])


def worker(args):
    if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') != 'true' or not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-'):
        raise ValueError('GPU UUID and growth required before framework import')
    out = args.output/'evidence'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    manifest = dict(plan=PLAN, command=[sys.executable,*sys.argv], started_utc=datetime.now(timezone.utc).isoformat(),
        environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS')},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        inputs_sha256=digest(INPUTS), data_version='saved 48-period MacroFinance synthetic fixture',
        wall_seconds=args.wall_seconds, gpu_trust_basis='trusted_escalated_gpu_execution')
    report = dict(status='running',plan=PLAN,posterior_qualification=False,external_consumer_admitted=False,default_promoted=False,no_ranking=True)
    write(out/'manifest.json',manifest)
    def mark(stage, details=None):
        write(out/'progress.json',dict(stage=stage,details=details,elapsed_seconds=time.monotonic()-started))
    try:
        paths, reconciliation=source_paths(args.campaign)
        before={str(p):digest(p) for p in paths}
        manifest['source_sha256']=before
        manifest['source_reconciliation']=reconciliation
        if reconciliation is not None:
            manifest['source_reconciliation_observed_sha256'] = {
                path: digest(path) for path in reconciliation.get('excluded_paths', ())
            }
        cohort=read(args.campaign/'frozen-cohort.json')
        manifest['seed_namespaces']=dict(qualification=cohort['qualification_seed'],reload=cohort['reload_seed'])
        from bayesfilter.runtime.gpu_provenance import selected_nvidia_gpu,query_nvidia_smi_gpus
        manifest['physical_gpu']=selected_nvidia_gpu(query_nvidia_smi_gpus())
        write(out/'manifest.json',manifest)
        from scripts.diagnose_modern_hmc_macrofinance_public import build
        adapter,memory=build(read(INPUTS),device='GPU',gpu_evidence=GPU_EVIDENCE)
        isolated = args.campaign/'isolated-source.json'
        if isolated.exists():
            from scripts.modern_hmc_source_snapshot import audit_imports
            manifest['loaded_local_modules'] = audit_imports(read(isolated))
        import tensorflow as tf
        import tensorflow_probability as tfp
        manifest.update(memory_policy=memory,tensorflow=tf.__version__,tensorflow_probability=tfp.__version__,dtype='float64',tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),use_xla=True)
        write(out/'manifest.json',manifest)
        from bayesfilter.runtime.execution_budget import execution_budget,execution_progress
        deadline = started+args.wall_seconds-900
        with execution_budget(deadline=deadline, check=lambda: time.monotonic() < deadline,
                              replace_wall_time_limit=bool(args.source)),execution_progress(mark):
            if args.source:
                from scripts.resume_modern_hmc_macrofinance_public import copy_parent,resume_checkpoint
                copy_parent(args.source,out,INPUTS)
                report['execution']=resume_checkpoint(out/'tuning/tuning_checkpoint.json',adapter,out)
            else:
                from bayesfilter.inference import tune_hmc_kernel,export_hmc_candidate_retained_runners
                from bayesfilter.testing.inference_validation.modern_bridge_audit import audit_native_evidence
                bound=frozen_binding(adapter,read(INPUTS),paths,read(args.campaign/'proposal-design.json'),cohort)
                cfg,quote=frozen_config(bound,cohort,args.wall_seconds-900-(time.monotonic()-started))
                write(out/'design.json',dict(policy=bound.config.acceptance_policy.payload(),search=cfg.payload(),work_price=quote,cohort=cohort))
                run=tune_hmc_kernel(adapter=adapter,initial_position=bound.initial_active_state,config=cfg,candidate_set_adapter=bound.typed_adapter,output_dir=out/'tuning')
                record_controller_outcome(report, run.result)
                # Persist the controller's cause before postprocessing, which
                # has its own independent failure modes. Never export source-
                # invalid evidence or reinterpret it as a resource stop.
                write(out/'controller-outcome.json', report['execution'])
                if run.result.completion_status == 'shared_invalidity':
                    report['status'] = 'shared_invalidity'
                    report['sources_unchanged'] = all(digest(p)==sha for p,sha in before.items())
                    report.update(elapsed_seconds=time.monotonic()-started,
                                  finished_utc=datetime.now(timezone.utc).isoformat())
                    write(out/'result.json', report)
                    return 1
                audit=audit_native_evidence(run.result,bound)
                write(out/'native-audit.json',audit)
                members=export_hmc_candidate_retained_runners(candidate_set_result=run.result,retained_binding=bound,output_dir=out/'members')
                if set(members)!=set(run.result.verified_candidate_ids):
                    raise ValueError('verified membership changed on export')
                write(out/'members.json',{cid:str(path) for cid,path in members.items()})
                report['execution'].update(verified=sorted(members),all_verified_members_exported=True)
            # Report fresh controller outcomes separately from earlier vetoes.
            saved=read(out/'tuning/tuning_checkpoint.json')['result']
            known={(r['L'],r['epsilon']) for r in cohort['prior_price_vetoes']}
            vetoed=[r['candidate_id'] for r in saved['candidates'] if (r['leapfrog_steps'],r['epsilon']) in known]
            write(out/'prior-veto-review.json',dict(prior_vetoed_candidate_ids=vetoed,prior_vetoes=cohort['prior_price_vetoes'],fresh_pass_cannot_erase_prior_veto=True,viable_without_prior_veto=[x for x in report['execution']['verified'] if x not in vetoed],no_posterior_authority=True))
        if report['execution']['completion_status']=='complete':
            from scripts.continue_modern_hmc_macrofinance_broad import reload
            # Existing reload's seed namespace is recorded in frozen-cohort.json.
            (out/'reload').mkdir()
            with execution_budget(deadline=started+args.wall_seconds),execution_progress(mark):
                report['reload']=reload(adapter,out,out/'reload',mark,seed_root=cohort['reload_seed'])
        report['sources_unchanged']=all(digest(p)==sha for p,sha in before.items())
        if isolated.exists():
            report['loaded_local_modules'] = audit_imports(read(isolated))
        if not report['sources_unchanged']:
            raise ValueError('source changed during frozen execution')
        report['allocator_bytes']=tf.config.experimental.get_memory_info('GPU:0')
        from scripts.modern_hmc_campaign_status import completion_status
        report['status']=completion_status(report['execution']['completion_status'])
    except Exception as exc:
        report.update(status='error',error=repr(exc),traceback=traceback.format_exc())
    report.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
    write(out/'result.json',report)
    return 0 if report['status']=='completed' else 1


def supervise(args):
    from scripts.run_modern_hmc_filter_health_gpu import select_shared_gpu,Reassessment,checkpoint_progress,stop_worker,reload_gpu_uuid
    from bayesfilter.runtime.gpu_provenance import query_nvidia_smi_gpus
    args.output.mkdir(exist_ok=False)
    start=time.monotonic();child=None;monitor=Reassessment()
    receipt=dict(status='selecting_device',active_run=True,plan=PLAN,command=[sys.executable,*sys.argv],started_utc=datetime.now(timezone.utc).isoformat(),max_wall_seconds=args.wall_seconds,observations=[])
    write(args.output/'manifest.json',receipt)
    def interrupted(signum,frame):
        raise InterruptedError('signal '+str(signum))
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,interrupted)
    try:
        required=reload_gpu_uuid(args.source) if args.source else None
        while True:
            now=time.monotonic()
            remaining=args.wall_seconds-(now-start)
            if remaining<=1020:raise TimeoutError('allocation exhausted waiting for capacity')
            if monitor.due(now,waiting=True):
                observation=monitor.observe(now,query_nvidia_smi_gpus(),required_uuid=required)
                receipt['observations'].append(observation);write(args.output/'manifest.json',receipt)
                selected=observation['selected_for_next_worker']
                if selected is not None:break
            time.sleep(1)
        allowance=args.wall_seconds-(time.monotonic()-start)-120
        # Refuse to consume a long campaign after queueing removed its full price.
        if not args.source:
            priced=cohort_price(read(args.campaign/'price-01/evidence/result.json')['execution']['rows'],read(args.campaign/'nominee-price-01/result.json')['rows'],read(args.campaign/'frozen-cohort.json'),allowance+120)
            write(args.output/'affordability.json',priced)
            priced['bounded_tranche'] = bool(getattr(args, 'allow_partial_budget', False))
            priced['completion_guaranteed'] = False
            write(args.output/'affordability.json',priced)
            if not priced['affordable'] and not priced['bounded_tranche']:
                raise ValueError('capacity wait left frozen maximum underfunded')
        command=[sys.executable,str(Path(__file__).resolve()),'--campaign',str(args.campaign),'--output',str(args.output),'--wall-seconds',str(allowance),'--worker']
        if args.source:command+=['--source',str(args.source)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=selected['uuid'],TF_FORCE_GPU_ALLOW_GROWTH='true',TF_CPP_MIN_LOG_LEVEL='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='2',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        receipt.update(status='running',selected_gpu=selected,worker_command=command,capacity_wait_seconds=time.monotonic()-start,environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH')})
        with (args.output/'worker.log').open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            receipt['worker_pid']=child.pid;write(args.output/'manifest.json',receipt)
            last=0.
            while child.poll() is None:
                now=time.monotonic()
                if now-start>=args.wall_seconds-5:
                    receipt['status']='wall_budget_exhausted';stop_worker(child);break
                if now-last>=60:
                    progress=checkpoint_progress(args.output)
                    receipt.update(heartbeat_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=now-start,numerical_progress=progress)
                    write(args.output/'manifest.json',receipt);last=now
                if monitor.due(now):
                    receipt['observations'].append(monitor.observe(now,query_nvidia_smi_gpus(),active_uuid=selected['uuid'],required_uuid=required))
                    write(args.output/'manifest.json',receipt)
                time.sleep(1)
        if receipt['status']=='running':
            report=read(args.output/'evidence/result.json')
            receipt['status']=report['status'];receipt['execution']=report.get('execution')
        receipt['exit_code']=child.returncode
    except Exception as exc:
        receipt.update(status='error',error=repr(exc),traceback=traceback.format_exc())
    finally:
        stop_worker(child)
        receipt.update(active_run=False,elapsed_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat())
        write(args.output/'manifest.json',receipt)
        print(json.dumps({k:receipt[k] for k in ('status','elapsed_seconds','active_run')}),flush=True)
    return 0 if receipt['status']=='completed' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wall-seconds',type=float,required=True)
    p.add_argument('--source',type=Path)
    p.add_argument('--allow-partial-budget', action='store_true',
        help='Run a bounded diagnostic tranche with unchanged evidence requirements; requires isolated sources')
    p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    args.campaign=args.campaign.resolve();args.output=args.output.resolve()
    if args.source:args.source=args.source.resolve()
    if args.allow_partial_budget and not (args.campaign/'isolated-source.json').is_file():
        p.error('bounded tranche requires an isolated source manifest')
    if not math.isfinite(args.wall_seconds) or args.wall_seconds<=1020:
        p.error('wall allowance must exceed export/closeout reserves')
    return worker(args) if args.worker else supervise(args)


if __name__=='__main__':raise SystemExit(main())
