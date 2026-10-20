#!/usr/bin/env python3
"""Resumable, budgeted simpler-model warm-start campaign; no q20 execution.

The parent imports no accelerator framework. GPU workers require explicit
device selection and memory growth before importing numerical components.
"""
from __future__ import annotations

import argparse
import fcntl
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
PLAN='docs/plans/bayesfilter-neutra-warm-start-master-2026-09-29.md'
SOURCES=(
    'scripts/run_neutra_warm_start_master.py',
    'scripts/continue_neutra_warm_start_campaign.py',
    'scripts/run_neutra_warm_start_repair_master.py',
    'scripts/run_neutra_causal_repair_master.py',
    'scripts/continue_neutra_causal_repair.py',
    'scripts/run_neutra_gap_closure.py',
    'scripts/run_neutra_geometry_repair.py',
    'scripts/run_neutra_warm_start_repair_campaign.sh',
    'bayesfilter/testing/neutra_warm_start_campaign.py',
    'bayesfilter/testing/neutra_warm_start_policy.py',
    'bayesfilter/testing/neutra_warm_start_closure.py',
    'bayesfilter/testing/neutra_warm_start_qualification.py',
    'bayesfilter/testing/neutra_warm_start_diagnostics.py',
    'bayesfilter/testing/neutra_gap_diagnostics.py',
    'bayesfilter/testing/neutra_geometry_diagnostics.py',
    'bayesfilter/testing/neutra_warm_start_targets_tf.py',
    'bayesfilter/inference/neutra_warm_start_tf.py',
    'bayesfilter/inference/neutra_flow_smc_tf.py',
    'bayesfilter/inference/neutra_transport.py',
    'bayesfilter/inference/neutra_transport_core.py',
    'bayesfilter/inference/neutra_weighted_training.py',
    'bayesfilter/inference/neutra_post_training.py',
    'bayesfilter/inference/hmc_verification.py',
    'bayesfilter/inference/hmc_candidate_set_execution.py',
    'tests/test_neutra_warm_start_pipeline.py',
    'tests/test_neutra_warm_start_master.py',
    'tests/test_neutra_warm_start_queue.py',
    'tests/test_neutra_warm_start_repair.py',
    'tests/test_neutra_causal_repair.py',
    'tests/test_neutra_fit_continuation.py',
    'tests/test_neutra_gap_closure.py')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def configuration():
    cfg={
        'schema':'bayesfilter.neutra.warm_start_master.v1',
        'targets':['gaussian','mixture','warped_mixture','wiggle','funnel'],
        'arms':['rkl','oracle','gabrie','smc','waste_free','aft','craft','gabrie_discovered'],
        'seeds':[11,23,37], 'gpu_process_seconds':7200.,'cpu_core_seconds':7200.,
        'gpu':1,'cpu_threads':2,'batch_size':256,'walkers':64,'walker_steps':4,
        'walker_initial_scale':.2,'widths_2d':[8,16],'widths_funnel':[20,40],
        'learning_rates':[.001,.003],'pilot_updates':256,'gradient_pilot_batches':8,
        'clip_pilot_multiplier':5.,'training_rungs':[256,1024,2048],
        'particles':1024,'mutation_steps':4,'max_smc_stages':128,
        'cess_fraction':.8,'resampling_fraction':.5,
        'mala_step_grid':[.001,.01,.1],
        'flow_particles':256,'flow_stages':8,'aft_inner_updates':32,'craft_passes':16,
        'reference_rows':{'training':8192,'validation':8192,'confirmation':16384},
        'wiggle_quadrature':[[16.,256],[24.,512],[32.,768]],'quadrature_tolerance':.001,
        'mode_starts':64,'mode_max_iterations':200,'mode_score_tolerance':1e-8,
        'mode_merge_distance':1e-6,'mode_curvature_tolerance':1e-7,
        'attempts_per_job':3,'repairs_per_seed':5,
        'repair_multiplier':2,'job_wall_seconds':{'prepare':120.,'checks':120.,'calibrate':240.,'train':240.,'qualify':240.,'price':120.},
        'hmc':{'leapfrogs':[3,9,18],'initial_epsilon':.5,'max_candidates':12,'work_units':48,'job_wall_seconds':220.},
        'plan':PLAN,
        'numerical_status':'explicit pilot hypotheses; none promoted to a production default',
        'authorization':'user requested building, reviewing and executing simpler-model pipeline',
        'q20_authorized_by_this_program':False,
    }
    cfg['provenance']={
        'source_and_canonical_choices':PLAN,
        'reference_rows':'independent low-cost banks; uncertainty recorded; increase before precision claims',
        'walker_initial_scale':'local spread hypothesis; check actual basin occupancy',
        'mode_tolerances':'FP64 reference hypotheses tested against analytic derivatives and repeated known modes',
        'mode_starts':'bounded diagnostic count, no exhaustive-discovery guarantee',
        'mala_step_grid':'log-spaced sensibility hypotheses, nominate by finite movement with acceptance >= 0.5',
        'cess_fraction':'0.8 incremental overlap hypothesis, separately recorded from cumulative ESS',
        'resampling_fraction':'0.5 cumulative ESS hypothesis, source-style threshold',
        'flow_stages_and_training':'small first source-mapped ladder; inadequate fit triggers larger priced rung',
        'wiggle_quadrature':'domain/resolution ladder; .001 relative second-moment/log-Z stability is a reference screen',
        'training_rungs':'work ceilings; continuing improvement at ceiling means undertrained, not converged',
        'cpu_threads':'resource sharing limit for existing concurrent jobs',
        'seeds':'fixed disjoint stream identifiers, not tuned parameters'}
    return cfg


def git_head():
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def source_hashes():
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}


def resource_forecast(state,cfg):
    """Measured phase means, with separate uncertain missing-phase estimates."""
    measured={}
    for phase in ('checks','prepare','calibrate','train','qualify'):
        rows=[r for r in state['attempts'] if r['phase']==phase and r['status']=='complete']
        if rows:
            measured[phase]={key:sum(r[key] for r in rows)/len(rows)
                for key in ('wall_seconds','cpu_core_seconds','gpu_process_seconds')}
            measured[phase]['completed_attempts']=len(rows)
    finished={r['job'] for r in state['attempts'] if r['status'] in ('complete','candidate_failed')}
    pending=[]
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            for target in cfg['targets']:
                if arm=='oracle' and target=='wiggle':continue
                if arm=='gabrie_discovered' and target not in ('gaussian','mixture','warped_mixture'):continue
                job=f'train-{target}-{arm}-s{seed}'
                if job not in finished:pending.append(job)
    estimate={key:len(pending)*measured.get('train',{}).get(key,0.)
        for key in ('wall_seconds','cpu_core_seconds','gpu_process_seconds')}
    return {'measured_phase_averages':measured,'pending_training_jobs':len(pending),
        'pending_training_job_ids':pending,
        'estimated_training_only':estimate,
        'limitation':'simple observed means; later SMC/flow arms, repairs and qualification may cost more; not a completion guarantee'}


def worker(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    faulthandler.register(signal.SIGUSR1,all_threads=False)
    config=json.loads(Path(args.config).read_text())
    if args.warm_parent:
        config={**config,'warm_parent':args.warm_parent,'warm_total_updates':args.warm_total_updates}
    if args.capacity_width is not None:
        config={**config,'capacity_width':args.capacity_width,'warm_total_updates':args.warm_total_updates}
    if args.kernel_repair or (args.phase=='closure_qualify' and args.repair_level>0):
        config={**config,'hmc':{**config['hmc'],'max_candidates':36,'work_units':144,
            'evidence_rungs':[1,2,4],'refinement_rounds':3,'explore_failed_intervals':True,'seed_offset':10000,
            'fixed_grid_max_attempts':config['hmc'].get('retry_fixed_grid_max_attempts',12)}}
    cpu=args.phase in ('prepare','checks','closure_prepare','closure_reference','closure_temporal')
    if cpu:os.environ['CUDA_VISIBLE_DEVICES']='-1'
    else:
        if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH','').lower()!='true':
            raise RuntimeError('GPU worker requires TF_FORCE_GPU_ALLOW_GROWTH=true before import')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(config['gpu']):
            raise RuntimeError('GPU worker visibility differs from explicit campaign assignment')
    os.environ['TF_NUM_INTRAOP_THREADS']=str(config['cpu_threads'])
    os.environ['TF_NUM_INTEROP_THREADS']=str(config['cpu_threads'])
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    if args.cpu_limit:
        limit=max(1,int(args.cpu_limit))
        resource.setrlimit(resource.RLIMIT_CPU,(limit,limit))
    sys.path.insert(0,str(ROOT))
    start=time.monotonic();usage0=resource.getrusage(resource.RUSAGE_SELF)
    children0=resource.getrusage(resource.RUSAGE_CHILDREN)
    manifest={'argv':sys.argv,'git_commit':git_head(),'source_sha256':source_hashes(),
        'effective_config':config,
        'python':sys.executable,'plan':str(ROOT/(config.get('closure',{}).get('plan',PLAN) if args.phase.startswith('closure_') else PLAN)),'phase':args.phase,'target':args.target,
        'arm':args.arm,'seed':args.seed,'cpu_only':cpu,'gpu_devices_intentionally_hidden':cpu,
        'TF_FORCE_GPU_ALLOW_GROWTH':os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH'),
        'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'output':str(output),
        'scientific_promotion':False,'data_version':'analytic_target_specification_and_disjoint_reference_seeds'}
    inputs={}
    for raw in (args.config,args.calibration):
        if raw and Path(raw).is_file():inputs[str(Path(raw).resolve())]=hashlib.sha256(Path(raw).read_bytes()).hexdigest()
    transport_path=config.get('closure',{}).get('teacher_transport')
    if transport_path:
        inputs[str(Path(transport_path).resolve())]=hashlib.sha256(Path(transport_path).read_bytes()).hexdigest()
    for name in ('geometry_map','geometry_assessment'):
        raw=config.get('closure',{}).get(name)
        if raw:inputs[str(Path(raw).resolve())]=hashlib.sha256(Path(raw).read_bytes()).hexdigest()
    if args.training and (Path(args.training)/'checkpoint-candidates.json').is_file():
        listing_path=Path(args.training)/'checkpoint-candidates.json'
        inputs[str(listing_path.resolve())]=hashlib.sha256(listing_path.read_bytes()).hexdigest()
        for row in json.loads(listing_path.read_text())['candidates']:
            for name in ('filename','probe_file','assessment_file'):
                path=Path(args.training)/row[name]
                inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    for raw,names in ((args.prepared,('training.tensor','validation.tensor','confirmation.tensor','discovered_modes.tensor','preparation.json')),
                      (args.training,('rkl-frozen.json','result.json','selected-frozen.json','phase.json','selected-assessment.json','selected-walkers.tensor','teacher-particles.tensor','teacher-log-weights.tensor')),
                      (args.warm_parent,('warm-frozen.json','result.json','teacher-particles.tensor','teacher-log-weights.tensor','teacher.json'))):
        if raw:
            for name in names:
                path=Path(raw)/name
                if path.is_file():inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    if args.warm_parent:
        for pattern in ('warm-*-checkpoint.json','warm-*-frozen.json','walkers-*.tensor','gradient-calibration.json'):
            for path in Path(args.warm_parent).glob(pattern):
                inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        calibration=Path(args.warm_parent).parent/'mutation-calibration.json'
        if calibration.is_file():inputs[str(calibration.resolve())]=hashlib.sha256(calibration.read_bytes()).hexdigest()
    snapshot=ROOT/'frozen-source.json'
    if snapshot.is_file():manifest['executed_source_snapshot']=json.loads(snapshot.read_text())
    manifest['input_sha256']=inputs
    for path in SOURCES:
        saved=output/'source'/path;saved.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/path,saved)
    try:
        if args.phase=='checks':
            command=[sys.executable,'-m','pytest','-q','--disable-warnings',
                'tests/test_neutra_warm_start_pipeline.py','tests/test_neutra_warm_start_master.py',
                'tests/test_neutra_warm_start_queue.py','tests/test_neutra_warm_start_repair.py']
            completed=subprocess.run(command,cwd=ROOT,check=False)
            if completed.returncode:raise RuntimeError('focused mechanics/runner checks failed')
            manifest['status']='complete';return
        import tensorflow as tf
        if cpu:
            manifest['memory_policy']={'mode':'cpu_only_gpu_hidden'}
        else:
            from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
            manifest['memory_policy']=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
            tf.config.experimental.enable_tensor_float_32_execution(True)
            manifest['gpu_devices']=[d.name for d in tf.config.list_logical_devices('GPU')]
            manifest['tf32_enabled']=True
        manifest['tensorflow']=tf.__version__
        write(output/'manifest.json',manifest)
        from bayesfilter.testing import neutra_warm_start_campaign as campaign
        if args.repair:
            original=dict(config)
            factor=config['repair_multiplier']
            parent=Path(args.training)/'result.json'
            parent_result=json.loads(parent.read_text()) if parent.exists() else {}
            collapse=parent_result.get('coverage_lost_during_rkl',False)
            config={**config,'batch_size':config['batch_size']*factor,'walkers':config['walkers']*factor,
                'training_rungs':[x*factor for x in config['training_rungs']],
                'particles':config['particles']*factor,'flow_particles':config['flow_particles']*factor,
                'aft_inner_updates':config['aft_inner_updates']*factor,'craft_passes':config['craft_passes']*factor,
                'repair_parent':args.training,'repair_rkl_collapse':collapse}
            if collapse:config['training_rungs']=original['training_rungs']
            write(output/'repair.json',{'trigger':'RKL lost coverage' if collapse else 'candidate or posterior screen failed',
                'kind':'restore before-RKL map, smaller calibrated learning rate and doubled batch' if collapse else 'fresh budget extension; no method change',
                'parent':args.training,
                'multiplier':factor,'original_config':original,'effective_config':config})
            manifest['effective_config']=config
        if args.phase.startswith('closure_'):
            from bayesfilter.testing import neutra_warm_start_closure as closure
            phase=args.phase.removeprefix('closure_')
            if phase=='temporal':closure.temporal_calibration(output,args.seed)
            elif phase=='geometry':
                from bayesfilter.testing.neutra_geometry_diagnostics import diagnose_geometry
                diagnose_geometry(args.target,args.prepared,output,config,args.seed)
            elif phase=='diagnose':
                from bayesfilter.testing.neutra_gap_diagnostics import diagnose
                diagnose(args.target,args.prepared,args.warm_parent,output,config,args.seed)
            elif phase=='prepare':closure.prepare(args.target,output,config,args.seed,repair=args.repair)
            elif phase=='teacher':closure.teacher(args.target,args.prepared,output,config,args.seed,repair=args.repair_level)
            elif phase=='fit':closure.fit(args.target,args.prepared,args.training,output,config,args.seed,arm=args.arm,repair=args.repair_level)
            elif phase=='refine':closure.refine(args.target,args.prepared,args.training,output,config,args.seed)
            elif phase=='sampler':closure.sampler_check(args.target,args.prepared,args.training,output,config,args.seed,repair=args.repair_level)
            elif phase=='reference':closure.final_reference(args.target,args.prepared,output,config,args.seed)
            elif phase=='qualify':closure.qualify_selected(args.target,args.prepared,args.training,output,config,args.seed)
            else:raise ValueError(f'unknown closure phase {phase}')
        elif args.phase=='prepare':campaign.prepare(args.target,output,config)
        elif args.phase=='calibrate':campaign.calibrate(args.target,args.prepared,output,config)
        elif args.phase=='price':
            if args.flow_composition:campaign.diagnose_flow_composition(args.target,args.calibration,output,config)
            else:campaign.price_flow(args.target,args.calibration,output,config)
        elif args.phase=='train':campaign.train(args.target,args.arm,args.seed,args.prepared,args.calibration,output,config)
        elif args.phase=='qualify':
            from bayesfilter.testing.neutra_warm_start_qualification import qualify
            qualify(args.target,args.seed,args.prepared,args.training,output,config)
        else:raise ValueError(f'unsupported worker phase {args.phase}')
        manifest['status']='complete'
    except BaseException as exc:
        manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        (output/'traceback.txt').write_text(traceback.format_exc())
        if type(exc).__name__=='CandidateFailure':
            manifest['status']='candidate_failed'
            raise SystemExit(20)
        if type(exc).__name__=='ExecutionBudgetExceeded':
            manifest['status']='budget_limited'
            raise SystemExit(21)
        raise
    finally:
        elapsed=time.monotonic()-start
        usage=resource.getrusage(resource.RUSAGE_SELF)
        children=resource.getrusage(resource.RUSAGE_CHILDREN)
        manifest['wall_seconds']=elapsed
        manifest['cpu_core_seconds']=((usage.ru_utime+usage.ru_stime)-(usage0.ru_utime+usage0.ru_stime)+
            (children.ru_utime+children.ru_stime)-(children0.ru_utime+children0.ru_stime))
        manifest['gpu_process_seconds']=0. if cpu else elapsed
        if not cpu and 'tf' in locals():
            try:manifest['allocator_bytes']=tf.config.experimental.get_memory_info('GPU:0')
            except (ValueError,RuntimeError):manifest['allocator_bytes']=None
        write(output/'manifest.json',manifest)


def resource_accounting(state,cfg):
    """Keep descriptive cost estimates separate from safe launch capacity."""
    keys=('gpu_process_seconds','cpu_core_seconds')
    estimated={key:sum(row.get(key,0.) for row in state['attempts']) for key in keys}
    upper=dict(estimated)
    uncertain=[]
    for row in state['attempts']:
        if not row.get('cpu_accounting_estimated'):
            continue
        value=row.get('cpu_core_seconds',0.)
        # An absent bound cannot become permission to spend the estimate's
        # remainder. Reserve the entire allocation until reconciled.
        bound=max(value,row.get('cpu_core_seconds_hard_upper_bound',cfg['cpu_core_seconds']))
        upper['cpu_core_seconds']+=bound-value
        uncertain.append({'job':row['job'],'estimated_cpu_core_seconds':value,
            'cpu_core_seconds_lower_bound':row.get('cpu_core_seconds_lower_bound',0.),
            'cpu_core_seconds_upper_bound':bound,'reason':row.get('accounting','missing final CPU counter')})
    return {'estimated_used':estimated,'conservative_used':upper,
        'estimated_remaining':{key:cfg[key]-estimated[key] for key in keys},
        'remaining_for_launch':{key:cfg[key]-upper[key] for key in keys},
        'cpu_accounting_exact':not uncertain,'uncertain_attempts':uncertain,
        'ceiling_compliance_established':all(upper[key]<=cfg[key] for key in keys)}


def remaining(state,cfg):
    return resource_accounting(state,cfg)['remaining_for_launch']


def invocation_identity(cfg,phase,extra):
    """Ordinary scientific input identity; resource allocations are not data."""
    scientific={k:v for k,v in cfg.items() if k not in
        ('cpu_core_seconds','gpu_process_seconds','job_wall_seconds','authorization')}
    inputs={}
    transport_path=cfg.get('closure',{}).get('teacher_transport')
    if transport_path:
        inputs[str(Path(transport_path).resolve())]=hashlib.sha256(Path(transport_path).read_bytes()).hexdigest()
    for option in ('--prepared','--calibration','--training','--warm-parent'):
        if option not in extra:continue
        path=Path(extra[extra.index(option)+1]).resolve()
        files=([path] if path.is_file() else sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix in ('.json','.tensor'))) if path.exists() else []
        inputs[str(path)]={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    payload={'phase':phase,'config':scientific,'arguments':extra,'inputs':inputs,
             'source':{p:h for p,h in source_hashes().items() if not p.startswith('tests/')}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def run_one(root,state,cfg,job,phase,extra):
    prior=[a for a in state['attempts'] if a['job']==job]
    identity=invocation_identity(cfg,phase,extra)
    state.setdefault('job_identities',{})[job]=identity
    matching=[a for a in prior if a.get('invocation_identity')==identity]
    if any(a['status'] in ('complete','candidate_failed') for a in matching):return True
    if len(matching)>=cfg['attempts_per_job']:
        state['status']='attempt_cap';return False
    rem=remaining(state,cfg)
    if min(rem.values())<=0:
        state['status']='budget_exhausted'
        state['next_phase']='reconcile uncertain costs or allocate additional compute before resuming pending jobs'
        return False
    output=root/'attempts'/f'{job}-r{len(prior)+1}'
    output.mkdir(parents=True,exist_ok=False)
    # Preserve the exact submitted configuration: the shared root config is
    # refreshed between phases and must not be the worker's lasting input.
    launch_config=output/'launch-config.json'
    write(launch_config,cfg)
    cpu=phase in ('prepare','checks','closure_prepare','closure_reference','closure_temporal')
    command=[sys.executable,str(ROOT/'scripts/run_neutra_warm_start_master.py'),'worker',
             '--config',str(launch_config),'--output',str(output),'--phase',phase,*extra]
    command+=['--cpu-limit',str(max(1,math.floor(rem['cpu_core_seconds'])))]
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='-1' if cpu else str(cfg['gpu']),
               TF_FORCE_GPU_ALLOW_GROWTH='true',TF_NUM_INTRAOP_THREADS=str(cfg['cpu_threads']),
               TF_NUM_INTEROP_THREADS=str(cfg['cpu_threads']),TF_CPP_MIN_LOG_LEVEL='2')
    timeout=min(rem['cpu_core_seconds']/cfg['cpu_threads'],cfg['job_wall_seconds'][phase],
                rem['gpu_process_seconds'] if not cpu else math.inf)
    state['active_job']={'job':job,'phase':phase,'output':str(output),'command':command,
                         'started_unix':time.time(),'timeout':timeout,
                         'invocation_identity':identity,
                         'cpu_core_seconds_limit':max(1,math.floor(rem['cpu_core_seconds']))}
    write(root/'state.json',state)
    print(json.dumps({'event':'start','job':job,'remaining':rem}),flush=True)
    started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with (output/'process.log').open('w') as log:
        try:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['active_job']['pid']=process.pid;write(root/'state.json',state)
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();code=124
        except KeyboardInterrupt:
            state['status']='interrupted_worker_may_be_running';raise
    wall=time.monotonic()-started;after=resource.getrusage(resource.RUSAGE_CHILDREN)
    row={'job':job,'phase':phase,'attempt':len(prior)+1,'output':str(output),'exit_code':code,
         'invocation_identity':identity,
         'status':'complete' if code==0 else ('candidate_failed' if code==20 else ('budget_limited' if code in (21,124,-signal.SIGXCPU) else 'failed')),'wall_seconds':wall,
         'gpu_process_seconds':0. if cpu else wall,
         'cpu_core_seconds':(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime)}
    state['attempts'].append(row);state['active_job']=None
    write(root/'state.json',state)
    print(json.dumps({'event':'finish',**row}),flush=True)
    return code in (0,20,21,124,-signal.SIGXCPU) and (phase in ('train','qualify') or phase.startswith('closure_')) or code==0


def completed_path(root,state,job):
    identity=state.get('job_identities',{}).get(job)
    rows=[r for r in state['attempts'] if r['job']==job and r['status']=='complete'
          and (identity is None or r.get('invocation_identity')==identity)]
    return Path(rows[-1]['output']) if rows else None


def qualify_trained(root,state,cfg,job,target,seed,prep,trained):
    extra=['--target',target,'--seed',str(seed),'--prepared',str(prep),'--training',str(trained)]
    old=completed_path(root,state,job)
    old_report=old/'qualification.json' if old else None
    if target in ('mixture','warped_mixture') and old_report and old_report.exists():
        if json.loads(old_report.read_text()).get('diagnostic_revision',1)<2:job+='-diagnostic-v2'
    if not run_one(root,state,cfg,job,'qualify',extra):return False
    output=completed_path(root,state,job)
    path=output/'qualification.json' if output else None
    if path and path.exists() and not json.loads(path.read_text())['qualified']:
        return run_one(root,state,cfg,job+'-search-repair','qualify',extra+['--kernel-repair'])
    return True


def qualified_training(state,trained,target):
    for row in state['attempts']:
        path=Path(row['output'])/'qualification.json'
        if row['phase']!='qualify' or row['status']!='complete' or not path.exists():continue
        result=json.loads(path.read_text())
        if result['training']!=str(trained):continue
        if target in ('mixture','warped_mixture') and result.get('diagnostic_revision',1)<2:continue
        if result['qualified']:return True
    return False


def repair_priority(row):
    target,arm,parent,*_=row
    path=Path(parent)/'result.json'
    result=json.loads(path.read_text()) if path.exists() else {}
    # Repair the isolated failure of RKL first; a failed plain baseline is
    # useful evidence and need not consume scarce repair time ahead of it.
    return (0 if result.get('coverage_lost_during_rkl') else 2 if arm=='rkl' else 1,)


def select_repairs(state,repairs,seed,limit):
    """Share the per-seed allowance across filtered master invocations."""
    used={r['job'] for r in state['attempts'] if r['phase']=='train'
          and r['job'].startswith('repair-') and r['job'].endswith(f'-s{seed}')}
    selected=[];deferred=[]
    for row in repairs:
        target,arm,*_=row
        job=f'repair-{target}-{arm}-s{seed}'
        if job in used or len(used)<limit:
            used.add(job);selected.append(row)
        else:deferred.append(row)
    return selected,deferred


def recover_active(root,state,cfg):
    """Reconcile a killed parent before permitting another worker launch."""
    active=state.get('active_job')
    if not active:return
    pid=active.get('pid');command=Path(f'/proc/{pid}/cmdline')
    observed_cpu=0.
    def observe_cpu():
        nonlocal observed_cpu
        try:
            fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            observed_cpu=max(observed_cpu,(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'))
        except (FileNotFoundError,ProcessLookupError):pass
    def alive():
        try:return str(active['output']).encode() in command.read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError):return False
    if alive():
        print(json.dumps({'event':'recover_live_worker','pid':pid,'job':active['job']}),flush=True)
        while alive() and time.time()-active['started_unix']<active['timeout']:
            observe_cpu()
            time.sleep(1)
        if alive():
            observe_cpu()
            if os.getpgid(pid)==pid:os.killpg(pid,signal.SIGKILL)
            else:os.kill(pid,signal.SIGKILL)
    manifest=Path(active['output'])/'manifest.json'
    data=json.loads(manifest.read_text()) if manifest.exists() else {}
    # Without a final manifest, elapsed time since launch is a conservative
    # charge. Clamping it to the old timeout hides an orphan's possible overrun.
    elapsed=max(0.,time.time()-active.get('started_unix',time.time()))
    status=data.get('status','interrupted')
    if status not in ('complete','candidate_failed','failed','budget_limited'):status='interrupted'
    phase=active.get('phase','prepare' if active['job'].startswith('prepare-') else 'train')
    row={'job':active['job'],'phase':phase,'output':active['output'],
        'invocation_identity':active.get('invocation_identity'),
        'status':status,'recovered':True,'wall_seconds':data.get('wall_seconds',elapsed),
        'gpu_process_seconds':data.get('gpu_process_seconds',0. if phase in ('prepare','checks','closure_prepare','closure_reference','closure_temporal') else elapsed),
        'cpu_core_seconds':max(observed_cpu,data.get('cpu_core_seconds',observed_cpu)),
        'accounting':'finished worker manifest and observed process counters'}
    if 'cpu_core_seconds' not in data or status=='interrupted':
        limit=active.get('cpu_core_seconds_limit')
        if limit is None and '--cpu-limit' in active.get('command',[]):
            command_args=active['command']
            limit=float(command_args[command_args.index('--cpu-limit')+1])
        if limit is None:limit=max(0.,remaining(state,cfg)['cpu_core_seconds'])
        # The /proc observation is a lower bound, not a completed charge.
        # Reserve the whole launch allowance when the final counter is lost.
        row.update(cpu_accounting_estimated=True,cpu_core_seconds_lower_bound=observed_cpu,
            cpu_core_seconds_hard_upper_bound=max(observed_cpu,limit),
            accounting='missing final CPU counter; reserve recorded launch CPU allowance')
    state['attempts'].append(row)
    state['active_job']=None;write(root/'state.json',state)


def summarize(root,state,cfg):
    results=[];qualifications=[]
    for row in state['attempts']:
        qpath=Path(row['output'])/'qualification.json'
        if qpath.exists():qualifications.append(json.loads(qpath.read_text()) | {'result':str(qpath)})
        path=Path(row['output'])/'result.json'
        if path.is_file() and row['phase']=='train':
            data=json.loads(path.read_text());after=data['after_rkl']
            results.append({k:data[k] for k in ('target','arm','seed','finite_candidate',
                'coverage_screen_passed','coverage_lost_during_rkl')} | {
                'heldout_forward_kl':after['heldout_forward_kl_estimate'],'result':str(path)})
    accounting=resource_accounting(state,cfg)
    summary={'state':state['status'],'remaining':accounting['remaining_for_launch'],'results':results,
        'resource_accounting':accounting,
        'cpu_accounting_uncertainty':state.get('cpu_accounting_uncertainty'),
        'forecast':resource_forecast(state,cfg),
        'qualifications':qualifications,'requested_scope':state.get('requested_scope'),
        'deferred_jobs':state.get('deferred_jobs',[]),
        'failed_attempts':[r for r in state['attempts'] if r['status']!='complete'],
        'statistically_supported_method_ranking':False,'posterior_qualification_complete':False,
        'all_executed_qualifications_passed':bool(qualifications) and all(q['qualified'] for q in qualifications),
        'next_phase':state.get('next_phase','resume uncompleted matrix jobs under the remaining budget'),
        'nonclaims':['No q20 result','No default promotion','Descriptive metrics do not rank methods']}
    write(root/'summary.json',summary)
    lines=['# Simpler-model warm-start master: execution status','',
        f"State: `{state['status']}`. Plan: `{PLAN}`.",'',
        f"Conservative remaining GPU process seconds: {summary['remaining']['gpu_process_seconds']:.1f}; "
        f"CPU core seconds: {summary['remaining']['cpu_core_seconds']:.1f}. "
        f"Estimated remaining CPU seconds: {accounting['estimated_remaining']['cpu_core_seconds']:.1f}. "
        f"CPU accounting exact: {accounting['cpu_accounting_exact']}. "
        f"Budget compliance established: {accounting['ceiling_compliance_established']}.",'',
        '| Target | Arm | Seed | Finite | Coverage screen | Coverage lost in RKL | Heldout FKL |',
        '|---|---|---|---|---|---|---|']
    for r in results:
        lines.append(f"| {r['target']} | {r['arm']} | {r['seed']} | {r['finite_candidate']} | "
                     f"{r['coverage_screen_passed']} | {r['coverage_lost_during_rkl']} | {r['heldout_forward_kl']} |")
    lines+=['','| Target | Seed | Training source | Posterior checks | Reason |','|---|---|---|---|---|']
    for q in qualifications:
        lines.append(f"| {q['target']} | {q['seed']} | {Path(q['training']).name} | {q['qualified']} | {q.get('reason',q['tuning_completion'])} |")
    lines+=['','| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |',
        '|---|---|---|---|---|---|',
        '| Preserve candidate evidence | Training screens only | See per-attempt validity and region diagnostics | '
        'Finite training and limited replications | Repair or downstream qualification | Posterior/default readiness |',
        '', '| Inference status | Finding |','|---|---|',
        '| Hard veto screen | Per-attempt finite and coverage results above |',
        '| Statistically supported ranking | None established |',
        '| Descriptive differences | Loss, region error, score residual and runtime |',
        '| Default readiness | Not established |',
        '| Next evidence | Independent precision-qualified frozen-map inference |','']
    (root/'result.md').write_text('\n'.join(lines))


def run(args):
    root=Path(args.output).resolve()
    if not (root/'config.json').is_file():
        root.mkdir(parents=True,exist_ok=True);write(root/'config.json',configuration())
    cfg=json.loads((root/'config.json').read_text())
    defaults=configuration();added={k:v for k,v in defaults.items() if k not in cfg}
    if 'price' not in cfg['job_wall_seconds']:
        cfg['job_wall_seconds']['price']=defaults['job_wall_seconds']['price'];write(root/'config.json',cfg)
    if added:
        write(root/'config-before-runner-completion.json',cfg)
        cfg.update(added);write(root/'config.json',cfg)
        write(root/'runner-migration.json',{'added_fields':added,'budgets_changed':False,
            'reason':'complete checks, interruption recovery, bounded repair and shared HMC phases'})
    state=json.loads((root/'state.json').read_text()) if (root/'state.json').is_file() else {
        'schema':'bayesfilter.neutra.warm_start_master_state.v1','status':'running',
        'attempts':[],'active_job':None,'git_commit':git_head(),'plan':PLAN}
    recover_active(root,state,cfg)
    state['status']='running'
    selected=args.targets.split(',') if args.targets else cfg['targets']
    if any(t not in cfg['targets'] for t in selected):raise ValueError('unknown requested target')
    arms=args.arms.split(',') if args.arms else cfg['arms']
    seeds=[int(s) for s in args.seeds.split(',')] if args.seeds else cfg['seeds']
    if any(a not in cfg['arms'] for a in arms):raise ValueError('unknown requested arm')
    if any(s not in cfg['seeds'] for s in seeds):raise ValueError('seed outside declared confirmation streams')
    state['requested_scope']={'targets':selected,'arms':arms,'seeds':seeds,'through':args.through}
    state.setdefault('deferred_jobs',[])
    try:
        check_hash=hashlib.sha256(json.dumps(source_hashes(),sort_keys=True).encode()).hexdigest()[:12]
        if not run_one(root,state,cfg,'checks-'+check_hash,'checks',[]):
            if state['status'] not in ('budget_exhausted','attempt_cap'):state['status']='checks_require_repair'
            return
        if args.through=='checks':state['status']='checks_complete';return
        for target in selected:
            if not run_one(root,state,cfg,'prepare-'+target,'prepare',['--target',target]):
                if state['status'] not in ('budget_exhausted','attempt_cap'):state['status']='preparation_requires_repair'
                return
        if args.through=='prepare':state['status']='preparation_complete';return
        for target in selected:
            prep=completed_path(root,state,'prepare-'+target)
            if not run_one(root,state,cfg,'calibrate-'+target,'calibrate',['--target',target,'--prepared',str(prep)]):
                if state['status'] not in ('budget_exhausted','attempt_cap'):state['status']='calibration_requires_repair'
                return
        if args.through=='calibrate':state['status']='calibration_complete';return
        if getattr(args,'warm_parent',None) or getattr(args,'capacity_width',None) is not None:
            if len(selected)!=1 or len(arms)!=1 or len(seeds)!=1:
                raise ValueError('targeted warm/capacity experiment requires one target, arm and seed')
            target,arm,seed=selected[0],arms[0],seeds[0]
            if args.warm_total_updates is None or args.warm_total_updates<=0:
                raise ValueError('warm continuation requires a positive lifetime update endpoint')
            extra=[]
            if args.warm_parent:
                if args.capacity_width is not None:raise ValueError('do not change width when restoring a saved optimizer')
                parent=Path(args.warm_parent).resolve()
                if not any(Path(row['output'])==parent and row['status']=='complete' for row in state['attempts']):
                    raise ValueError('warm continuation parent is not a completed attempt in this campaign')
                extra=['--warm-parent',str(parent)]
                job=f'continue-warm-{target}-{arm}-s{seed}-u{args.warm_total_updates}'
            else:
                allowed=cfg['widths_funnel'] if target=='funnel' else cfg['widths_2d']
                if args.capacity_width not in allowed:raise ValueError('capacity width outside reviewed target grid')
                extra=['--capacity-width',str(args.capacity_width)]
                job=f'capacity-{target}-{arm}-w{args.capacity_width}-s{seed}-u{args.warm_total_updates}'
            prep=completed_path(root,state,'prepare-'+target)
            cal=completed_path(root,state,'calibrate-'+target)/'calibration.json'
            if not run_one(root,state,cfg,job,'train',['--target',target,'--arm',arm,'--seed',str(seed),
                '--prepared',str(prep),'--calibration',str(cal),*extra,'--warm-total-updates',str(args.warm_total_updates)]):
                if state['status'] not in ('budget_exhausted','attempt_cap'):state['status']='warm_continuation_requires_repair'
                return
            trained=completed_path(root,state,job)
            result=json.loads((trained/'result.json').read_text()) if trained else None
            if args.through=='qualify' and result and result['finite_candidate'] and result['coverage_screen_passed']:
                if not qualify_trained(root,state,cfg,'qualify-'+job,target,seed,prep,trained):
                    if state['status'] not in ('budget_exhausted','attempt_cap'):state['status']='qualification_harness_requires_repair'
                    return
            state['status']='warm_continuation_attempted'
            state['next_phase']='review warm-duration intervention, RKL coverage, and frozen-map qualification'
            return
        if args.through=='price':
            for target in selected:
                cal=completed_path(root,state,'calibrate-'+target)/'calibration.json'
                extra=['--target',target,'--calibration',str(cal)]
                tag='-composition-v2' if args.flow_composition else '-inner-loop-v3'
                if args.flow_composition:extra+=['--flow-composition']
                if not run_one(root,state,cfg,'price-'+target+tag,'price',extra):
                    if state['status'] not in ('budget_exhausted','attempt_cap'):state['status']='pricing_requires_repair'
                    return
            state['status']='pricing_complete';return
        for seed in seeds:
            repairs=[]
            for arm in arms:
                for target in selected:
                    if arm=='oracle' and target=='wiggle':continue
                    if arm=='gabrie_discovered' and target not in ('gaussian','mixture','warped_mixture'):continue
                    prep=completed_path(root,state,'prepare-'+target)
                    cal=completed_path(root,state,'calibrate-'+target)/'calibration.json'
                    job=f'train-{target}-{arm}-s{seed}'
                    okay=run_one(root,state,cfg,job,'train',['--target',target,'--arm',arm,'--seed',str(seed),
                        '--prepared',str(prep),'--calibration',str(cal)])
                    summarize(root,state,cfg)
                    if not okay:
                        if state['status'] not in ('budget_exhausted','attempt_cap'):state['status']='harness_requires_repair'
                        if state['status']!='budget_exhausted':state['next_phase']=f'repair infrastructure for {job}, then resume'
                        return
                    trained=completed_path(root,state,job)
                    result=json.loads((trained/'result.json').read_text()) if trained else None
                    viable=bool(result and result['finite_candidate'] and result['coverage_screen_passed'])
                    if not viable:
                        parent=trained or Path(next(r['output'] for r in reversed(state['attempts']) if r['job']==job))
                        repairs.append((target,arm,parent,prep,cal))
                    if args.through=='qualify' and viable:
                        if not qualify_trained(root,state,cfg,f'qualify-{target}-{arm}-s{seed}',target,seed,prep,trained):
                            if state['status'] not in ('budget_exhausted','attempt_cap'):state['status']='qualification_harness_requires_repair'
                            return
                        if not qualified_training(state,trained,target):
                            repairs.append((target,arm,trained,prep,cal))
            if args.through in ('repair','qualify'):
                repairs.sort(key=repair_priority)
                selected_repairs,deferred_repairs=select_repairs(state,repairs,seed,cfg['repairs_per_seed'])
                sys.path.insert(0,str(ROOT))
                from bayesfilter.testing.neutra_warm_start_policy import preserve_deferred
                state['deferred_jobs']=preserve_deferred(state['deferred_jobs'],[
                    {'target':t,'arm':a,'seed':seed,'phase':'repair',
                     'reason':'declared cumulative per-seed repair cap; evidence preserved'} for t,a,*_ in deferred_repairs])
                for target,arm,parent,prep,cal in selected_repairs:
                    job=f'repair-{target}-{arm}-s{seed}'
                    if not run_one(root,state,cfg,job,'train',['--target',target,'--arm',arm,'--seed',str(seed),
                        '--prepared',str(prep),'--calibration',str(cal),'--repair','--training',str(parent)]):
                        if state['status'] not in ('budget_exhausted','attempt_cap'):state['status']='repair_harness_requires_repair'
                        return
                    trained=completed_path(root,state,job)
                    if trained:
                        state['deferred_jobs']=preserve_deferred(state['deferred_jobs'],[],resolved=[job])
                    result=json.loads((trained/'result.json').read_text()) if trained else None
                    if args.through=='qualify' and result and result['finite_candidate'] and result['coverage_screen_passed']:
                        if not qualify_trained(root,state,cfg,'qualify-'+job,target,seed,prep,trained):
                            if state['status'] not in ('budget_exhausted','attempt_cap'):state['status']='qualification_harness_requires_repair'
                            return
        state['status']='requested_matrix_attempted' if args.through=='qualify' else 'training_complete_downstream_pending'
        state['next_phase']='review rejected/undertrained candidates and replicated posterior evidence; no automatic q20 promotion'
    finally:
        write(root/'state.json',state);summarize(root,state,cfg)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['plan','run','status','worker'])
    parser.add_argument('--output',required=True)
    parser.add_argument('--config')
    parser.add_argument('--phase',choices=['checks','prepare','calibrate','train','qualify','price',
        'closure_prepare','closure_teacher','closure_fit','closure_refine','closure_sampler','closure_reference','closure_qualify','closure_temporal','closure_diagnose','closure_geometry'])
    parser.add_argument('--target');parser.add_argument('--prepared');parser.add_argument('--calibration')
    parser.add_argument('--arm');parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--training');parser.add_argument('--repair',action='store_true')
    parser.add_argument('--repair-level',type=int,default=0)
    parser.add_argument('--warm-parent');parser.add_argument('--warm-total-updates',type=int)
    parser.add_argument('--capacity-width',type=int)
    parser.add_argument('--flow-composition',action='store_true')
    parser.add_argument('--kernel-repair',action='store_true')
    parser.add_argument('--cpu-limit',type=float)
    parser.add_argument('--targets');parser.add_argument('--arms');parser.add_argument('--seeds')
    parser.add_argument('--through',choices=['checks','prepare','calibrate','price','train','repair','qualify'],default='qualify')
    args=parser.parse_args()
    if args.mode=='plan':
        path=Path(args.output)/'config.json'
        if path.exists():raise FileExistsError(path)
        write(path,configuration());print(path)
    elif args.mode=='worker':worker(args)
    elif args.mode=='run':
        root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
        with (root/'master.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('this campaign already has a live master')
            run(args)
            status=json.loads((root/'state.json').read_text())['status']
            if status.endswith(('_require_repair','_requires_repair')) or status in ('attempt_cap','budget_exhausted'):
                raise SystemExit(2 if status!='budget_exhausted' else 3)
    else:
        root=Path(args.output)
        state=json.loads((root/'state.json').read_text())
        print(json.dumps({'status':state['status'],'active_job':state['active_job'],
            'attempts':len(state['attempts']),
            'resource_accounting':resource_accounting(state,json.loads((root/'config.json').read_text()))},indent=2))


if __name__=='__main__':main()
