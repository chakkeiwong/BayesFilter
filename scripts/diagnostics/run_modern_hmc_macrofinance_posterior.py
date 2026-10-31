"""C3 public retained pricing and experimental posterior checks; no promotion."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

PLAN='docs/plans/bayesfilter-modern-hmc-c3-reference-continuation-2026-10-29.md'
FIXED_PLAN='docs/plans/bayesfilter-modern-hmc-c4-fixed-count-replication-2026-10-30.md'
PAIRS=((3,.4),(5,.4),(9,.4),(13,.4),(18,.4),(25,.2),(25,.4))
NAMES=('S_log_diag_G','S_log_diag_Pi','S_lower_Pi_G','omega_unconstrained',
       'Q_log_cholesky_G','Q_lower_Pi_G','Q_log_cholesky_Pi','R_log_sd_mean','R_log_sd_contrast')
QUANTITIES=(*NAMES,*(f'prior_square_{n}' for n in NAMES),
            *(f'smooth_upper_tail_{n}' for n in NAMES),'state_sd_G','state_sd_Pi','acf1_G','acf1_Pi')


def read(path): return json.loads(Path(path).read_text())
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def validate_reference(reference):
    if reference.get('quantities')!=list(QUANTITIES):
        raise ValueError('reference quantity identity differs')
    screen=reference.get('screen',{})
    if not screen.get('passed') or screen.get('tolerance')!=.02:
        raise ValueError('independent reference precision has not passed')
    import math
    for key in ('mean','sd'):
        values=reference.get(key,[])
        if len(values)!=31 or any(not math.isfinite(v) for v in values):
            raise ValueError('reference vector missing or nonfinite')
    uncertainty=screen.get('uncertainty',[])
    if (len(uncertainty)!=31 or any(not math.isfinite(v) or v<0 for v in uncertainty)
            or any(v<=0 for v in reference['sd'])
            or any(u/s>.02 for u,s in zip(uncertainty,reference['sd']))):
        raise ValueError('reference uncertainty is invalid')
    return dict(reference,uncertainty=uncertainty)


def validate_pairs(members):
    rows=sorted(members.items(),key=lambda item:(item[1].num_leapfrog_steps,item[1].step_size))
    if tuple((m.num_leapfrog_steps,m.step_size) for _,m in rows)!=PAIRS:
        raise ValueError('all seven qualified member identities must be retained')
    return rows


def roots(index, fixed_count_replication=False):
    if fixed_count_replication:
        return dict(price=(20261030,30001+index),warmup=(20261030,1000000+200000*index),
                    retained=(20261030,1100000+200000*index))
    return dict(price=(20261029,30001+index),warmup=(20261029,1000000+200000*index),
                retained=(20261029,1100000+200000*index))


def posterior_quote(price,available):
    import math
    rows=price.get('rows',[])
    if (price.get('status')!='completed' or price.get('stage')!='price'
            or tuple((r['L'],r['epsilon']) for r in rows)!=PAIRS
            or any(r.get('status')!='priced' or r.get('transitions_per_chain')!=1000 for r in rows)):
        raise ValueError('all seven actual retained chunk prices required')
    seconds=[r.get('seconds') for r in rows]
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in seconds):
        raise ValueError('finite positive measured prices required')
    # At most ten discarded and ten retained chunks per member. Public call
    # timing includes evidence validation, so this is deliberately conservative.
    maximum_native=20*sum(seconds)
    load_overhead=max(0.,price['elapsed_seconds']-sum(seconds))
    required=1.5*maximum_native+load_overhead+120
    return dict(required_seconds=required,available_seconds=available,affordable=required<=available,
        maximum_chunks_per_member=20,operational_margin=1.5,measured_load_overhead_seconds=load_overhead,
        closeout_seconds=120,statistical_guarantee=False)


def validate_price_scope(price,manifest,launch,reference_path):
    if price.get('source_unchanged') is not True:
        raise ValueError('pricing source integrity not established')
    if manifest.get('stage')!='price' or manifest.get('git_commit')!=launch['commit']:
        raise ValueError('pricing numerical source scope differs')
    hashes=manifest.get('source_sha256',{})
    for path in (Path(launch['inputs']),Path(reference_path)):
        if hashes.get(str(path.resolve()))!=digest(path):
            raise ValueError('pricing target data or reference scope differs')


def make_transform(phase9,tf):
    programs={}
    def transform(samples):
        samples=tf.convert_to_tensor(samples,tf.float64)
        shape=tuple(samples.shape)
        if len(shape)!=3 or shape[1:]!=(4,9) or shape[0] is None or not 0<shape[0]<=10000:
            raise ValueError('bounded static [draw,4,9] quantities required')
        if shape not in programs:
            @tf.function(input_signature=[tf.TensorSpec(shape,tf.float64)],autograph=False,jit_compile=True)
            def program(values):
                theta=tf.reshape(values,(-1,9))
                chart=phase9._tf_chart_batch_with_derivatives(theta,tf)
                covariance=chart['initial_state_covariance']
                variances=tf.linalg.diag_part(covariance)
                sd=tf.sqrt(variances)
                acf=tf.linalg.diag_part(chart['transition_matrix']@covariance)/variances
                z=theta/tf.constant(.35,tf.float64)
                quantities=tf.concat((theta,z*z,tf.math.sigmoid(2*(z-1)),sd,acf),axis=1)
                quantities=tf.where(chart['support_valid'][:,None],quantities,
                                    tf.constant(float('nan'),tf.float64))
                return tf.reshape(quantities,(shape[0],4,31))
            programs[shape]=program
        return programs[shape](samples)
    return transform


def posterior_config(member,index,fixed_count_replication=False):
    from bayesfilter.inference import (HMCPrecisionTarget,HMCPrecisionPolicy,
        HMCPosteriorAssessmentPolicy,SequentialNeuTraHMCConfig)
    precision=HMCPrecisionPolicy(tuple(HMCPrecisionTarget(n,mcse_sd_ratio_max=.02) for n in QUANTITIES),
        method='lugsail',batch_size=None,min_batches=20,lugsail_r=3,lugsail_c=.5,jit_compile=True)
    assessment=HMCPosteriorAssessmentPolicy(warmup_bulk_ess_min=0.,warmup_tail_ess_min=0.,
        retained_bulk_ess_min=400.,retained_tail_ess_min=400.,warmup_consecutive_checks=1,precision=precision)
    seed=roots(index,fixed_count_replication)
    return SequentialNeuTraHMCConfig(step_size=member.step_size,num_leapfrog_steps=member.num_leapfrog_steps,
        warmup_seed=seed['warmup'],retained_seed=seed['retained'],warmup_chunk_results=1000,
        warmup_min_results=2000,warmup_check_window_results=1000,warmup_max_results=10000,
        warmup_rhat_max=1.05,retained_chunk_results=1000,
        retained_min_results=10000 if fixed_count_replication else 1000,retained_max_results=10000,
        retained_rhat_max=1.01,minimum_chain_count=4,assessment_policy=assessment,jit_compile=True,
        energy_error_log_accept_threshold=-1000.,max_results_per_chain=10000,count_budget_reason=None)


def earliest_internal_prefix(result):
    """Replay C3's internal decision only; reference values never enter."""
    if not result.get('warmup_passed'):
        return None
    for check in result['retained_checks']:
        if check.get('health',{}).get('health_passed') is not True:
            return None
        diagnostic=check.get('modern_rhat')
        if (check['completed_results_per_chain']>=1000 and diagnostic is not None
                and diagnostic.get('passed') is True):
            return check['completed_results_per_chain']
    return None


def fixed_count_comparisons(posterior,ordered,reference,config_factory):
    """Add paired descriptive prefix results without replacing terminal outcomes."""
    import tensorflow as tf
    from bayesfilter.testing.inference_validation.modern_full_model_consumer import reference_screen
    by_id={cid:(i,member) for i,(cid,member) in enumerate(ordered)}
    comparisons=[]
    for row in posterior['rows']:
        item=dict(candidate_id=row['candidate_id'],terminal_delivered=row.get('delivered',False),
                  paired_same_trajectory=True,scientific_ranking=None,coverage_calibrated=False)
        path=Path(row['result_file']) if row.get('result_file') else None
        if path is None or not path.is_file():
            item['status']='assessment_unavailable';comparisons.append(item);continue
        result=read(path);count=earliest_internal_prefix(result)
        item.update(status='assessed',terminal_count=result['retained_results_per_chain'],
                    fixed_count_reached=result['retained_results_per_chain']==10000,
                    earliest_internal_pass_count=count,terminal_reference=row.get('reference'),
                    terminal_hard_vetoes=result['hard_vetoes'])
        if count is not None:
            i,member=by_id[row['candidate_id']]
            values=tf.io.parse_tensor(tf.io.read_file(str(path.parent/'retained_raw.tensor')),tf.float64)
            if values.shape[0]<count:raise ValueError('prefix exceeds saved retained draws')
            item['early_prefix_reference']=reference_screen(values[:count],reference,mcse_multiplier=3.,
                normalized_tolerance=.1,precision_policy=config_factory(member,i).assessment_policy.precision)
        comparisons.append(item)
    return comparisons


def load_continuation_helper():
    path=Path(__file__).resolve().with_name('continuation_helper.py')
    if not path.is_file():
        raise ValueError('frozen posterior continuation helper required')
    spec=importlib.util.spec_from_file_location('c3_frozen_continuation',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def execute_posterior_cohort(*,ordered,output,transform,reference,check_inputs,
                            helper,previous_report=None,config_factory=posterior_config):
    from bayesfilter.testing.inference_validation.modern_full_model_consumer import reference_screen
    candidates=[dict(candidate_id=cid) for cid,_ in ordered]
    prior=previous_report or dict(rows=[dict(candidate_id=cid,status='posterior_not_started',
        result_file=None,tuning_state='verified') for cid,_ in ordered],
        verified_candidate_ids=[cid for cid,_ in ordered])
    return helper.run_cohort(candidates=candidates,members=dict(ordered),previous_report=prior,
        output=output,factory=config_factory,names=QUANTITIES,transform=transform,
        reference_check=lambda values,config:reference_screen(values,reference,mcse_multiplier=3.,
            normalized_tolerance=.1,precision_policy=config.assessment_policy.precision),
        check_inputs=check_inputs)


def worker(args):
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();launch=read(args.campaign/'launch.json')
    reference=validate_reference(read(args.reference))
    sources=dict(launch['source_sha256'])
    reference_manifest=read(args.reference.parent/'manifest.json')
    reference_result=read(args.reference.parent/'result.json')
    if (reference_result.get('status')!='completed' or not reference_result.get('source_unchanged')
            or not reference_result.get('checks',{}).get('passed')
            or reference_manifest['campaign']!=str(args.campaign.resolve())):
        raise ValueError('completed source-checked same-campaign reference required')
    sources.update(reference_manifest['source_sha256'])
    for path in (Path(__file__).resolve(),args.reference.resolve(),args.quantity_fixture.resolve(),
                 args.reference.parent/'manifest.json',args.reference.parent/'result.json'):
        sources[str(path)]=digest(path)
    if args.stage=='posterior':
        if args.price_result is None: raise ValueError('posterior requires measured all-member price result')
        price=read(args.price_result)
        validate_price_scope(price,read(args.price_result.parent/'manifest.json'),launch,args.reference)
        quote=posterior_quote(price,args.wall_seconds-60)
        if not quote['affordable']: raise ValueError('posterior worst-case design exceeds reservation')
        sources[str(args.price_result)]=digest(args.price_result);write(output/'quote.json',quote)
        helper_path=Path(__file__).resolve().with_name('continuation_helper.py')
        sources[str(helper_path)]=digest(helper_path)
    previous_report=None
    if args.previous is not None:
        if args.stage!='posterior': raise ValueError('only posterior chunks can be resumed')
        previous_manifest=read(args.previous/'manifest.json')
        supervisor_manifest=read(args.previous.parent/'manifest.json')
        if supervisor_manifest.get('active_run') is not False:
            raise ValueError('cannot resume an active posterior worker')
        if previous_manifest.get('stage')!='posterior' or previous_manifest.get('git_commit')!=launch['commit']:
            raise ValueError('previous posterior numerical source differs')
        for path,checksum in previous_manifest['source_sha256'].items():
            if digest(path)!=checksum: raise ValueError('previous posterior source drift: '+path)
        if previous_manifest['source_sha256'].get(str(args.reference))!=digest(args.reference):
            raise ValueError('posterior reference changed across continuation')
        helper=load_continuation_helper()
        sources.update(helper.tree_hashes(args.previous/'posterior'))
        previous_report=read(args.previous/'posterior/result.json')
    fixed=bool(getattr(args,'fixed_count_replication',False))
    config_factory=lambda member,index:posterior_config(member,index,fixed)
    report=dict(status='running',stage=args.stage,fixed_count_replication=fixed,rows=[],posterior_qualified=False,release_qualified=False,
                method_ranking=None,tuning_membership_unchanged=True)
    manifest=dict(plan=FIXED_PLAN if fixed else PLAN,fixed_count_replication=fixed,
        command=[sys.executable,*sys.argv],started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=launch['commit'],source_sha256=sources,wall_seconds=args.wall_seconds,
        environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH',
            'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        seeds={str(i):roots(i,fixed) for i in range(7)},stage=args.stage)
    if args.previous is not None and bool(previous_manifest.get('fixed_count_replication',False))!=fixed:
        raise ValueError('posterior stopping mode changed across continuation')
    write(output/'manifest.json',manifest)
    def mark(stage,detail=None):
        write(output/'progress.json',dict(stage=stage,detail=detail,elapsed_seconds=time.monotonic()-started,
            updated_utc=datetime.now(timezone.utc).isoformat()))
    def check():
        for path,checksum in sources.items():
            if digest(path)!=checksum: raise ValueError('source or input changed: '+path)
    try:
        check()
        if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')!='true':
            raise ValueError('pre-import GPU memory growth required')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=launch['gpu_uuid']:
            raise ValueError('preserve physical GPU identity of exported members')
        sys.path[:0]=[launch['consumer'],launch['candidate'],str(Path(launch['candidate'])/'scripts')]
        os.environ['BAYESFILTER_DIAGNOSTIC_CONSUMER_ROOT']=launch['consumer']
        from one_country_affine_macro_source_binding import Phase9SourceBinding
        import one_country_affine_macro_phase9_hmc as phase9
        source=Phase9SourceBinding(Path(launch['candidate']),launch['commit'])
        inputs=read(launch['inputs'])
        if reference['dataset_id']!=inputs['design']['dataset_id']:
            raise ValueError('reference data identity differs')
        mark('build_public_gpu_target')
        adapter=phase9.build_generic_target(inputs['observations'],dataset_id=inputs['design']['dataset_id'],
            source_binding=source,execution_device='GPU',gpu_evidence=Path(launch['gpu_evidence']))
        import tensorflow as tf
        import tensorflow_probability as tfp
        tf.config.experimental.enable_tensor_float_32_execution(False)
        from bayesfilter.runtime.execution_budget import execution_budget,execution_progress,require_execution_budget,ExecutionBudgetExceeded
        from bayesfilter.runtime.gpu_provenance import query_nvidia_smi_gpus,selected_nvidia_gpu
        manifest.update(tensorflow=tf.__version__,tensorflow_probability=tfp.__version__,tf32=False,
            jit_compile=True,dtype='float64',gpu_trust_basis='trusted_escalated_gpu_execution',
            memory_policy=adapter.contract.filter_program.filter_manifest['gpu_policy'],
            physical_gpu=selected_nvidia_gpu(query_nvidia_smi_gpus()),adapter_signature=adapter.adapter_signature())
        write(output/'manifest.json',manifest)
        transform=make_transform(phase9,tf)
        fixture=read(args.reference.parent.parent/'parity-01/parity-points.json')
        expected=read(args.quantity_fixture)
        if expected['points']!=fixture['points'] or expected['quantities']!=list(QUANTITIES):
            raise ValueError('quantity parity fixture mismatch')
        mark('quantity_parity')
        actual=transform(tf.reshape(tf.constant(fixture['points'],tf.float64),(7,4,9)))
        tf.debugging.assert_near(tf.reshape(actual,(28,31)),tf.constant(expected['values'],tf.float64),atol=1.e-9,rtol=1.e-9)
        write(output/'quantity-parity.json',dict(passed=True,points=28,quantities=31,
            max_absolute_error=float(tf.reduce_max(tf.abs(tf.reshape(actual,(28,31))-tf.constant(expected['values'],tf.float64))))))
        if args.stage=='preflight':
            report.update(status='completed',quantity_parity=True)
        else:
            from bayesfilter.inference import load_hmc_candidate_retained_runners,run_hmc_posterior
            from bayesfilter.inference.hmc_posterior_assessment import validate_sequential_seeds
            from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
            mark('reload_all_seven_members')
            paths=read(args.campaign/'qualification-01/evidence/members.json')
            with execution_budget(deadline=started+args.wall_seconds-60),execution_progress(mark):
                members=load_hmc_candidate_retained_runners(paths.values(),adapter=adapter)
                ordered=validate_pairs(members)
                if args.stage=='posterior' and [r['member_hash'] for r in price['rows']]!=[m.member_hash for _,m in ordered]:
                    raise ValueError('priced and posterior member identities differ')
                used=set()
                for i,(_,m) in enumerate(ordered):
                    if m.claim_eligible or m.posterior_convergence_authority: raise ValueError('unsupported promotion')
                    # Public calls separately validate complete tuning inventories.
                    inventory=validate_sequential_seeds(config_factory(m,i),forbidden=used)
                    used.update(seed for _,_,seed in inventory);used.update(roots(i,fixed).values())
                if args.stage=='posterior':
                    posterior=execute_posterior_cohort(ordered=ordered,output=output/'posterior',
                        transform=transform,reference=reference,check_inputs=check,
                        helper=load_continuation_helper(),previous_report=previous_report,config_factory=config_factory)
                    report.update(status=posterior['status'],rows=posterior['rows'],
                        all_declared_checks_passed_count=posterior['delivered_count'])
                    if fixed:
                        comparison=fixed_count_comparisons(posterior,ordered,reference,config_factory)
                        write(output/'fixed-count-comparison.json',dict(rows=comparison,
                            all_seven_preserved=len(comparison)==7,scientific_ranking=None,coverage_calibrated=False))
                for i,(cid,member) in enumerate(ordered if args.stage=='price' else ()):
                    require_execution_budget();check();destination=output/f'member-{i:02d}'
                    row=dict(candidate_id=cid,L=member.num_leapfrog_steps,epsilon=member.step_size,
                             status='running',member_hash=member.member_hash)
                    report['rows'].append(row);write(output/'result.json',report)
                    mark('member_'+args.stage,dict(index=i,**row));begin=time.monotonic()
                    if args.stage=='price':
                        try:
                            result=member.run(num_results=1000,seed=roots(i,fixed)['price'],output_dir=destination)
                        except ValueError:
                            archive=destination/'retained_archive.json'
                            if not archive.exists() or not read(archive).get('health_failures'): raise
                            row.update(status='numerical_health_failed',seconds=time.monotonic()-begin,
                                archive=str(archive),health_failures=read(archive)['health_failures'])
                            write(output/'result.json',report);continue
                        values=transform(result['position_samples'])
                        tf.debugging.assert_all_finite(values,'price quantities')
                        if 'GPU' not in result['samples'].device: raise ValueError('CPU fallback')
                        row.update(status='priced',seconds=time.monotonic()-begin,transitions_per_chain=1000,
                            archive=result['archive_path'],posterior_draws=False,
                            allocator=tf.config.experimental.get_memory_info('GPU:0'))
                    write(output/'result.json',report)
            if args.stage=='price': report['status']='completed'
        check();source.audit();source.check_imports();report['source_unchanged']=True
    except Exception as exc:
        report.update(status='resource_stopped' if type(exc).__name__=='ExecutionBudgetExceeded' else 'error',
                      error=repr(exc),traceback=traceback.format_exc())
    report.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
    write(output/'result.json',report);mark(report['status']);print(json.dumps(report))
    return 0 if report['status']=='completed' else 1


def supervise(args):
    launch=read(args.campaign/'launch.json');sys.path[:0]=[launch['candidate']]
    from scripts.run_modern_hmc_filter_health_gpu import Reassessment,stop_worker
    from bayesfilter.runtime.gpu_provenance import query_nvidia_smi_gpus
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();monitor=Reassessment();process=None
    fixed=bool(getattr(args,'fixed_count_replication',False))
    receipt=dict(status='selecting_device',active_run=True,plan=FIXED_PLAN if fixed else PLAN,
        fixed_count_replication=fixed,stage=args.stage,
        command=[sys.executable,*sys.argv],wall_seconds=args.wall_seconds,observations=[],
        started_utc=datetime.now(timezone.utc).isoformat())
    def interrupt(signum,frame): raise InterruptedError('supervisor signal '+str(signum))
    for signum in (signal.SIGTERM,signal.SIGINT): signal.signal(signum,interrupt)
    try:
        while True:
            now=time.monotonic()
            if now-started>=args.wall_seconds-120: raise TimeoutError('capacity wait exhausted allocation')
            if monitor.due(now,waiting=True):
                row=monitor.observe(now,query_nvidia_smi_gpus(),required_uuid=launch['gpu_uuid'])
                receipt['observations'].append(row);write(args.output/'manifest.json',receipt)
                if row['selected_for_next_worker'] is not None: break
            time.sleep(1)
        command=[sys.executable,str(Path(__file__).resolve()),'--campaign',str(args.campaign),
            '--output',str(args.output/'evidence'),'--reference',str(args.reference),
            '--quantity-fixture',str(args.quantity_fixture),'--stage',args.stage,'--worker',
            '--wall-seconds',str(args.wall_seconds-(time.monotonic()-started)-30)]
        if args.price_result is not None: command.extend(('--price-result',str(args.price_result)))
        if args.previous is not None: command.extend(('--previous',str(args.previous)))
        if fixed:command.append('--fixed-count-replication')
        env=dict(os.environ,**launch['environment'],CUDA_VISIBLE_DEVICES=launch['gpu_uuid'])
        with (args.output/'worker.log').open('x') as log:
            process=subprocess.Popen(command,env=env,cwd='/tmp',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            receipt.update(status='running',worker_pid=process.pid,worker_command=command)
            write(args.output/'manifest.json',receipt);last=0.
            while process.poll() is None:
                now=time.monotonic()
                if now-started>=args.wall_seconds-5:
                    receipt['status']='wall_budget_exhausted';stop_worker(process);break
                if now-last>=60:
                    progress=args.output/'evidence/progress.json'
                    receipt.update(heartbeat_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=now-started,progress=read(progress) if progress.exists() else None)
                    write(args.output/'manifest.json',receipt);last=now
                if monitor.due(now):
                    receipt['observations'].append(monitor.observe(now,query_nvidia_smi_gpus(),active_uuid=launch['gpu_uuid']))
                    write(args.output/'manifest.json',receipt)
                time.sleep(1)
        if receipt['status']=='running': receipt['status']=read(args.output/'evidence/result.json')['status']
        receipt['exit_code']=process.returncode
    except Exception as exc:
        receipt.update(status='error',error=repr(exc),traceback=traceback.format_exc())
    finally:
        stop_worker(process);receipt.update(active_run=False,elapsed_seconds=time.monotonic()-started,
            finished_utc=datetime.now(timezone.utc).isoformat());write(args.output/'manifest.json',receipt)
        write(args.output/'settlement.json',dict(inclusive_charge_seconds=receipt['elapsed_seconds'],
            budget_cap_seconds=args.wall_seconds,unused_reservation_seconds=max(0.,args.wall_seconds-receipt['elapsed_seconds']),
            charge_basis='inclusive supervisor wall once; children not added',status=receipt['status']))
    return 0 if receipt['status']=='completed' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','output','reference','quantity-fixture'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stage',choices=('preflight','price','posterior'),required=True)
    p.add_argument('--wall-seconds',type=float,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--price-result',type=Path)
    p.add_argument('--previous',type=Path)
    p.add_argument('--fixed-count-replication',action='store_true',
                   help='C4 diagnostic: fresh seeds, all seven members, fixed 10000 retained draws')
    args=p.parse_args()
    for name in ('campaign','output','reference','quantity_fixture'): setattr(args,name,getattr(args,name).resolve())
    if args.price_result is not None: args.price_result=args.price_result.resolve()
    if args.previous is not None: args.previous=args.previous.resolve()
    return worker(args) if args.worker else supervise(args)


if __name__=='__main__': raise SystemExit(main())
