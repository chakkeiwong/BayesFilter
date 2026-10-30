#!/usr/bin/env python3
"""Bounded, resumable six-method engineering/pricing master.

The master imports no accelerator framework. Scientific phases remain explicit
matrix entries until implementations, calibration and budgets support them.
"""
from __future__ import annotations

import argparse
import fcntl
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
LIVE=Path('/home/ubuntu/python/BayesFilter')
PLAN='docs/plans/bayesfilter-neutra-six-method-execution-2026-10-03.md'
CAMPAIGN=LIVE/'docs/plans/artifacts/neutra-six-method-2026-10-03/campaign-r1'
SHARED=LIVE/'docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1'
METHODS=('fab','gabrie','ais','smc','aft','craft')
TARGETS=('gaussian','mixture','warped_mixture')
STAGES=('native','posterior_teacher','common_iaf','pre_post_rkl','downstream','random_two','random_three','dimension_transfer','state_space','q20')
LIMITS={'gpu_process_seconds':7200.,'cpu_core_seconds':14400.}
TEST_PATHS=('tests/test_neutra_six_method_controls.py',
            'tests/test_neutra_warm_start_pipeline.py',
            'tests/test_neutra_six_method_master.py')
AUTHOR_FIXTURE='docs/plans/artifacts/neutra-warm-start-master-2026-09-29/author-reference-r2.json'


def read(path): return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def source_snapshot():
    files=list((LIVE/'bayesfilter').rglob('*.py'))
    files.extend(LIVE/name for name in (
        'scripts/run_neutra_six_method_master.py','scripts/run_neutra_six_method_campaign.sh',
        *TEST_PATHS,'tests/conftest.py','pytest.ini',AUTHOR_FIXTURE,PLAN,
        'docs/plans/bayesfilter-neutra-generic-recovery-and-transfer-plan-2026-10-03.md'))
    hashes={str(p.relative_to(LIVE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    destination=CAMPAIGN/('source-'+digest[:16])
    if not destination.exists():
        destination.mkdir()
        for name in hashes:
            dest=destination/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(LIVE/name,dest)
        write(destination/'source.json',{'git_commit':subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=LIVE,text=True).strip(),'sha256':hashes,
            'snapshot_id':digest,'dirty_worktree_preserved':True})
    # An ordinary checksum catches an accidental edit during copying, too.
    for name,expected in hashes.items():
        if hashlib.sha256((destination/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'source snapshot differs from recorded input: {name}')
    return destination


def terminal_matrix_recorded(state):
    """Terminal applicability outcomes count as recorded, not as method passes."""
    latest={(r.get('method'),r.get('target')):r for r in state['attempts'] if r.get('method')}
    return all(latest.get((method,target),{}).get('status')=='complete'
               for method in METHODS for target in TARGETS)


def audit_campaign(campaign, shared_state):
    """Read preserved evidence without importing TensorFlow or running a worker."""
    campaign=Path(campaign)
    state=read(campaign/'state.json')
    errors=[]; limitations=[]; reconciled=[]; sources={}; cells=[]

    def require(condition, message):
        if not condition: errors.append(message)

    for row in state['attempts']:
        label=f"{row['job']}-r{row['attempt']}"
        source=Path(row['source'])
        if str(source) not in sources:
            identity=read(source/'source.json')
            hashes=identity['sha256']
            require(bool(hashes),f'{source.name}: empty source inventory')
            digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
            require(digest==identity['snapshot_id'],f'{source.name}: source index checksum mismatch')
            mismatches=[name for name,expected in hashes.items()
                        if not (source/name).is_file() or
                        hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected]
            require(not mismatches,f'{source.name}: changed or missing files: {mismatches}')
            sources[str(source)]={'snapshot_id':digest,'files_checked':len(hashes),
                                  'mismatches':mismatches,'git_commit':identity['git_commit']}
        for key in LIMITS:
            require(math.isfinite(row[key]) and row[key]>=0,f'{label}: invalid {key}')
        charge='six-method-20261003-'+label
        matches=[r for r in shared_state['attempts'] if r['job']==charge]
        require(len(matches)==1,f'{label}: expected exactly one shared ledger charge')
        if len(matches)==1:
            require(all(math.isclose(row[k],matches[0][k],rel_tol=0,abs_tol=1e-6)
                        for k in LIMITS),f'{label}: shared ledger charge differs')
        manifest_path=Path(row['output'])/'manifest.json'
        if not manifest_path.is_file():
            errors.append(f'{label}: manifest missing');continue
        manifest=read(manifest_path)
        require(manifest.get('source')==str(source/'source.json'),f'{label}: manifest source mismatch')
        require(manifest.get('command')==row['command'],f'{label}: manifest command mismatch')
        delta=row['cpu_core_seconds']-manifest['cpu_core_seconds']
        require(delta>=-1e-6,f'{label}: CPU ledger undercounts the worker manifest')
        reconciled.append({'attempt':label,'ledger_cpu_core_seconds':row['cpu_core_seconds'],
                           'manifest_cpu_core_seconds':manifest['cpu_core_seconds'],
                           'difference_seconds':delta})
        if row['device']=='cpu' and manifest.get('execution_cwd')!=str(source):
            limitations.append(f'{label}: historical tests ran from the live tree; frozen-source coverage not established')

    checks=[r for r in state['attempts'] if r['job']=='check']
    latest_check=checks[-1] if checks else None
    require(latest_check is not None and latest_check['status']=='complete','latest focused check is not complete')
    if latest_check:
        manifest=read(Path(latest_check['output'])/'manifest.json')
        require(manifest.get('execution_cwd')==latest_check['source'],
                'fresh frozen-source CPU check is required')
        require(manifest.get('gpu_devices_intentionally_hidden') is True,
                'CPU check did not record deliberately hidden GPUs')
    preflights=[r for r in state['attempts'] if r['job']=='preflight']
    if not preflights or preflights[-1]['status']!='complete':
        errors.append('GPU/XLA preflight is not complete')
    else:
        preflight=read(Path(preflights[-1]['output'])/'result.json')
        require(preflight.get('status')=='passed' and 'GPU' in preflight.get('device',''),
                'GPU/XLA preflight did not execute on GPU')
    latest={(r.get('method'),r.get('target')):r for r in state['attempts'] if r.get('method')}
    for method in METHODS:
        for target in TARGETS:
            row=latest.get((method,target))
            label=f'{target}/{method}'
            if row is None:
                errors.append(f'{label}: no recorded attempt');continue
            output=Path(row['output']);result_path=output/'result.json'
            if not result_path.is_file():
                errors.append(f'{label}: result missing');continue
            result=read(result_path);manifest=read(output/'manifest.json')
            status=result.get('status')
            require(row['status']=='complete' and manifest['status']=='complete',f'{label}: process incomplete')
            require(result.get('method')==method and result.get('target_name')==target,f'{label}: wrong method or target')
            require(result.get('scientific_promotion') is False,f'{label}: false scientific promotion')
            require(result.get('sampling_quality')=='not_established' and
                    result.get('downstream_hmc')=='not_run',f'{label}: unsupported sampling or HMC claim')
            policy=manifest.get('memory_policy',{})
            require(manifest.get('TF_FORCE_GPU_ALLOW_GROWTH')=='true' and
                    policy.get('all_physical_devices_memory_growth') is True and
                    policy.get('configured_before_logical_device_initialization') is True,
                    f'{label}: GPU memory policy not verified')
            require(manifest.get('jit_compile') is True and manifest.get('dtype')=='float64_reference',
                    f'{label}: unexpected arithmetic/execution scope')
            require(manifest.get('git_commit')==sources[row['source']]['git_commit'],
                    f'{label}: Git identity differs from snapshot')
            probe=None
            if status=='mechanics_control_completed':
                required=('bank.tensor','log-weights.tensor','student-checkpoint.json',
                          'student-frozen.json','post-training-1000.json','native-summary.json',
                          'reference-summary.json','target.json','spec.json','stdout.log')
                missing=[name for name in required if not (output/name).is_file()]
                require(not missing,f'{label}: missing artifacts: {missing}')
                require(result.get('teacher_admitted') is False,f'{label}: unqualified teacher admitted')
                if not missing:
                    frozen=read(output/'student-frozen.json')
                    require(frozen.get('target_signature')==result['target_signature'],f'{label}: student target mismatch')
                    probe=read(output/'post-training-1000.json')
                    require(probe.get('complete') is True and probe.get('finite') is True and
                            probe.get('rows')==1000 and probe.get('valid_rows')==1000,
                            f'{label}: required 1,000-point probe incomplete')
                    require(probe.get('geometry_role')=='explanatory_only_no_calibrated_finite_cutoff',
                            f'{label}: geometry probe misclassified')
            elif status=='prerequisite_failed':
                require(method=='fab' and target in ('mixture','warped_mixture') and
                        result.get('native_training')=='not_run' and bool(result.get('reason')),
                        f'{label}: unexplained or misclassified prerequisite outcome')
            else:
                errors.append(f'{label}: nonterminal result {status}')
            cells.append({'method':method,'target':target,'status':status,'attempt':row['attempt'],
                          'result':str(result_path),'wall_seconds':row['wall_seconds'],
                          'cpu_core_seconds':row['cpu_core_seconds'],
                          'score_residual_median':probe['score_residual_norm']['median'] if probe else None,
                          'artifact_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in output.iterdir() if p.is_file()}})
    matrix_path=campaign/'matrix.json'
    if matrix_path.is_file():
        indexed={(r['method'],r['target']):r for r in read(matrix_path)['rows']
                 if r['phase']=='mechanics_and_pricing'}
        for cell in cells:
            require(indexed.get((cell['method'],cell['target']),{}).get('status')==cell['status'],
                    f"{cell['target']}/{cell['method']}: stale matrix status")
    require(state.get('active') is None,'campaign still has an active worker')
    used={key:sum(r[key] for r in state['attempts']) for key in LIMITS}
    for key,value in used.items():require(value<=state['limits'][key],f'{key}: allocation exceeded')
    return {'schema':'bayesfilter.neutra.six_method_terminal_audit.v1',
            'status':'passed' if not errors else 'failed','errors':errors,'limitations':limitations,
            'source_snapshots':sources,'cells':cells,'attempts_checked':len(state['attempts']),
            'resource_use':used,'accounting_reconciliation':reconciled,
            'accounting_note':'Historical worker manifests exclude shutdown; parent ledgers include it. New manifests record both.',
            'scientific_promotion':False,'source_equivalence':'not_established',
            'posterior_teacher_qualification':'not_established'}


def conservative_shared_remaining():
    cfg,state=read(SHARED/'config.json'),read(SHARED/'state.json')
    used={key:sum(row.get(key,0.) for row in state['attempts']) for key in LIMITS}
    for row in state['attempts']:
        if row.get('cpu_accounting_estimated'):
            used['cpu_core_seconds']+=max(0.,row.get('cpu_core_seconds_hard_upper_bound',cfg['cpu_core_seconds'])-row.get('cpu_core_seconds',0.))
    return {key:cfg[key]-used[key] for key in LIMITS}


def worker(specification):
    spec=read(specification);output=Path(spec['output']);started=time.monotonic()
    os.environ['CUDA_VISIBLE_DEVICES']=str(spec['gpu'])
    if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH')!='true': raise RuntimeError('memory growth must precede imports')
    resource.setrlimit(resource.RLIMIT_CPU,(int(spec['cpu_limit']),int(spec['cpu_limit'])))
    sys.path.insert(0,str(ROOT))
    manifest={'command':[sys.executable,str(Path(__file__).resolve()),'worker','--spec',str(specification)],
        'environment':sys.executable,'plan':PLAN,'source':str(ROOT/'source.json'),
        'git_commit':read(ROOT/'source.json')['git_commit'],'seed':spec.get('seed'),
        'method':spec.get('method'),'target':spec.get('target'),'gpu':spec['gpu'],
        'TF_FORCE_GPU_ALLOW_GROWTH':os.environ['TF_FORCE_GPU_ALLOW_GROWTH'],
        'dtype':'float64_reference','jit_compile':True,'tf32':True,'output':str(output),
        'trust_basis':'trusted_escalated_fixed_campaign_wrapper','status':'running','started_unix':time.time()}
    write(output/'manifest.json',manifest)
    try:
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        manifest['memory_policy']=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(True)
        write(output/'manifest.json',manifest)
        from bayesfilter.testing.neutra_six_method_controls import run_control,serializable
        if spec['action']=='preflight':
            @tf.function(input_signature=[tf.TensorSpec([32,2],tf.float64)],jit_compile=True,autograph=False)
            def kernel(x):return tf.reduce_sum(x*x,1)
            x=tf.ones([32,2],tf.float64);values=kernel(x)
            tf.debugging.assert_equal(values,tf.fill([32],tf.constant(2.,tf.float64)))
            if 'GPU' not in values.device: raise RuntimeError('GPU/XLA preflight executed off GPU')
            result={'status':'passed','device':values.device,'traces':kernel.experimental_get_tracing_count(),
                    'compiled_kernel':'squared_norm','scientific_promotion':False}
        else:
            result=run_control(spec['method'],spec['target'],output,spec['seed'])
        write(output/'result.json',serializable(result))
        manifest.update(status='complete',allocator=tf.config.experimental.get_memory_info('GPU:0'))
        code=0
    except Exception as error:
        traceback.print_exc()
        manifest.update(status='failed',error_type=type(error).__name__,error=str(error))
        code=1
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        manifest.update(wall_seconds=time.monotonic()-started,cpu_core_seconds=usage.ru_utime+usage.ru_stime,finished_unix=time.time())
        manifest['gpu_process_seconds']=manifest['wall_seconds']
        write(output/'manifest.json',manifest)
    return code


class Controller:
    def __init__(self):
        CAMPAIGN.mkdir(parents=True,exist_ok=True)
        self.state=read(CAMPAIGN/'state.json') if (CAMPAIGN/'state.json').exists() else {
            'schema':'bayesfilter.neutra.six_method_engineering.v1','status':'prepared','attempts':[],
            'active':None,'limits':LIMITS,'plan':PLAN,'methods':list(METHODS),'created_unix':time.time()}
        # Read-only/terminal resumes must not manufacture a new execution source
        # or erase the saved phase just because unrelated live files changed.
        self.source=None
        if read(SHARED/'state.json').get('active_job'): raise RuntimeError('another shared campaign has unresolved active work')
        self.recover()
        if not (CAMPAIGN/'state.json').exists():
            self.sync('prepared','check, then preflight, then all six method controls')

    def remaining(self):
        shared=conservative_shared_remaining()
        return {key:min(shared[key],LIMITS[key]-sum(r.get(key,0.) for r in self.state['attempts'])) for key in LIMITS}

    def sync(self,status,next_action):
        self.state.update(status=status,next_action=next_action,remaining=self.remaining(),updated_unix=time.time())
        write(CAMPAIGN/'state.json',self.state)
        write(CAMPAIGN/'next-phase.json',{'phase':status,'next_action':next_action,
            'remaining':self.state['remaining'],'resume_command':'bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_six_method_campaign.sh resume'})
        rows=[]
        for method in METHODS:
            for target in TARGETS:
                attempts=[r for r in self.state['attempts'] if r.get('method')==method and r.get('target')==target]
                last=attempts[-1] if attempts else None
                result=read(Path(last['output'])/'result.json') if last and (Path(last['output'])/'result.json').exists() else {}
                rows.append({'method':method,'target':target,'phase':'mechanics_and_pricing',
                    'status':result.get('status',last['status'] if last else 'not_run'),
                    'attempts':len(attempts),'result':str(Path(last['output'])/'result.json') if last else None})
            for stage in STAGES:
                rows.append({'method':method,'phase':stage,'status':'not_run',
                             'reason':'requires source validity, calibration, evidence and priced allocation'})
        write(CAMPAIGN/'matrix.json',{'methods':list(METHODS),'rows':rows,
            'engineering_complete_is_not_study_complete':True})

    def recover(self):
        row=self.state.get('active')
        if row is None:return
        pid=row.get('pid')
        if pid:
            try:
                os.kill(pid,0)
                if '\nState:\tZ' not in Path(f'/proc/{pid}/status').read_text():
                    raise RuntimeError(f'worker {pid} still active; do not duplicate')
            except (ProcessLookupError,FileNotFoundError):pass
        manifest=Path(row['output'])/'manifest.json'
        saved=read(manifest) if manifest.exists() else {}
        self.finish(row,0 if saved.get('status')=='complete' else 125,saved)

    def finish(self,row,code,manifest):
        wall=manifest.get('wall_seconds',min(time.time()-row['started_unix'],row['wall_limit']))
        cost=manifest.get('cpu_core_seconds',row['cpu_limit'])
        complete={**row,'exit_code':code,'status':'complete' if code==0 else 'failed',
            'wall_seconds':wall,'gpu_process_seconds':wall if row['device']=='gpu' else 0.,
            'cpu_core_seconds':cost,'cpu_accounting_conservative':'cpu_core_seconds' not in manifest}
        self.state['attempts'].append(complete);self.state['active']=None
        shared=read(SHARED/'state.json')
        charge='six-method-20261003-'+row['job']+'-r'+str(row['attempt'])
        if not any(r['job']==charge for r in shared['attempts']):
            shared['attempts'].append({'job':charge,'phase':'six_method_engineering','output':row['output'],
                'status':complete['status'],**{key:complete[key] for key in LIMITS},'wall_seconds':wall})
            write(SHARED/'state.json',shared)
        self.sync('phase_complete' if code==0 else 'repair_required','continue other independent method rows; inspect localized failures')
        print(json.dumps({'job':row['job'],'status':complete['status'],'wall_seconds':wall,'remaining':self.remaining()}),flush=True)

    def execute(self,job,*,device='gpu',method=None,target=None,action='control'):
        if self.source is None:self.source=source_snapshot()
        prior=[r for r in self.state['attempts'] if r['job']==job]
        if prior and prior[-1]['status']=='complete' and prior[-1]['source']==str(self.source):return prior[-1]
        same_source=[r for r in prior if r['source']==str(self.source)]
        if len(same_source)>=2: raise RuntimeError(f'localized retry cap reached for unchanged {job}; inspect the recorded failure')
        available=self.remaining()
        wall_limit=min(600.,available['cpu_core_seconds']/2,
            available['gpu_process_seconds'] if device=='gpu' else 600.)
        if wall_limit<60: raise RuntimeError('initial engineering allocation exhausted')
        attempt=len(prior)+1
        output=CAMPAIGN/f'{job}-r{attempt}';output.mkdir(exist_ok=False)
        cpu_limit=int(min(available['cpu_core_seconds'],wall_limit*2))
        row={'job':job,'attempt':attempt,'output':str(output),'source':str(self.source),'device':device,
             'method':method,'target':target,'wall_limit':wall_limit,'cpu_limit':cpu_limit,'started_unix':time.time()}
        if device=='cpu':
            command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
                *TEST_PATHS,
                '--disable-warnings','--maxfail=3']
            cwd=self.source
        else:
            spec={**row,'action':action,'gpu':1,'seed':20261003}
            write(output/'spec.json',spec)
            command=[sys.executable,str(self.source/'scripts/run_neutra_six_method_master.py'),
                     'worker','--spec',str(output/'spec.json')]
            cwd=self.source
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','TF_FORCE_GPU_ALLOW_GROWTH':'true',
             'XLA_PYTHON_CLIENT_PREALLOCATE':'false','TF_CPP_MIN_LOG_LEVEL':'2',
             'TF_NUM_INTRAOP_THREADS':'2','TF_NUM_INTEROP_THREADS':'1','OMP_NUM_THREADS':'2',
             'OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'}
        env['PYTHONPATH']=str(self.source)
        if device=='cpu':
            # This mechanics suite does not exercise the separate custom-op
            # binary. It must not load an uncaptured binary from the live tree.
            env['BAYESFILTER_PRELOAD_CUSTOM_OP']='0'
            env['WARM_START_AUTHOR_FIXTURE']=str(self.source/AUTHOR_FIXTURE)
        usage0=resource.getrusage(resource.RUSAGE_CHILDREN)
        with (output/'stdout.log').open('w') as log:
            child=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_CPU,(cpu_limit,cpu_limit)))
            row['pid']=child.pid;row['command']=command
            self.state['active']=row;self.sync('running',job)
            try: code=child.wait(timeout=wall_limit)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=124
        usage1=resource.getrusage(resource.RUSAGE_CHILDREN)
        manifest_path=output/'manifest.json'
        manifest=read(manifest_path) if manifest_path.exists() else {}
        if device=='gpu':
            manifest['worker_cpu_core_seconds']=manifest.get('cpu_core_seconds')
        manifest.update(cpu_core_seconds=usage1.ru_utime+usage1.ru_stime-usage0.ru_utime-usage0.ru_stime)
        manifest.update(execution_cwd=str(cwd),process_cpu_accounting='parent_wait_includes_shutdown')
        if device=='cpu':
            manifest.update(command=command,environment=sys.executable,source=str(self.source/'source.json'),
                git_commit=read(self.source/'source.json')['git_commit'],
                plan=PLAN,gpu_devices_intentionally_hidden=True,wall_seconds=time.time()-row['started_unix'],
                status='complete' if code==0 else 'failed')
            write(output/'result.json',{'status':'passed' if code==0 else 'failed','exit_code':code,
                'scope':'CPU independent mechanics; no scientific promotion'})
        write(output/'manifest.json',manifest)
        self.finish(row,code,manifest)
        return self.state['attempts'][-1]

    def run(self):
        if terminal_matrix_recorded(self.state):
            return self.report()
        check=self.execute('check',device='cpu')
        if check['status']!='complete':self.sync('repair_required','CPU mechanics failed; inspect check log');return 1
        preflight=self.execute('preflight',action='preflight')
        if preflight['status']!='complete':self.sync('repair_required','GPU/XLA memory-policy preflight failed');return 1
        for target in TARGETS:
            for method in METHODS:
                attempt=self.execute(f'control-{target}-{method}',method=method,target=target)
                if attempt['status']!='complete':
                    # A process failure may invalidate a shared consumer (as the
                    # original post-training bug did). Inspect/repair it before
                    # repeating it across all dependent controls. Recorded FAB
                    # applicability outcomes still permit independent rows.
                    self.sync('repair_required','inspect infrastructure failure before dependent controls')
                    return 1
        return self.report()

    def report(self):
        records=[]
        for row in self.state['attempts']:
            path=Path(row['output'])/'result.json'
            if row.get('method') and path.exists():
                result=read(path)
                records.append({'method':row['method'],'target':row['target'],'status':result.get('status'),
                    'seconds':row['wall_seconds'],'result':str(path),
                    'native_warm_seconds':result.get('native_warm_seconds'),
                    'student_warm_seconds':result.get('student_warm_seconds'),
                    'probe_seconds':result.get('probe_seconds')})
        write(CAMPAIGN/'pricing.json',{'records':records,'shared_remaining':conservative_shared_remaining(),
            'full_confirmation_trials':636,'simple_confirmation_trials':36,
            'complete_forecast_available':False,
            'missing_costs':['method-specific calibration','qualified teacher replication','production training','fresh public HMC qualification','generalization failures/repairs'],
            'do_not_extrapolate_smoke_updates_into_converged_training':True})
        self.sync('engineering_review_required','audit preserved artifacts before scientific preparation')
        # Each audit is versioned; original worker artifacts remain untouched.
        index=1
        while (CAMPAIGN/f'audit-r{index}').exists():index+=1
        output=CAMPAIGN/f'audit-r{index}';output.mkdir()
        try:
            audit=audit_campaign(CAMPAIGN,read(SHARED/'state.json'))
        except (OSError,ValueError,KeyError,TypeError) as error:
            audit={'status':'failed','errors':[f'{type(error).__name__}: {error}'],
                   'scientific_promotion':False}
        write(output/'result.json',audit)
        self.state['terminal_audit']=str(output/'result.json')
        if audit['status']!='passed':
            self.sync('engineering_audit_failed','repair the recorded artifact/provenance issue; no automatic GPU rerun')
            print(json.dumps({'audit':str(output/'result.json'),'status':'failed','errors':audit['errors']}),flush=True)
            return 1
        self.sync('engineering_complete_scientific_preparation_required',
                  'close source/controller and teacher gaps; price calibrated procedures before scientific phases')
        print(json.dumps({'audit':str(output/'result.json'),'status':'passed','scientific_promotion':False}),flush=True)
        return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('check','preflight','run','resume','status','worker'))
    parser.add_argument('--spec');args=parser.parse_args()
    if args.action=='worker':return worker(args.spec)
    if args.action=='status':
        if not (CAMPAIGN/'state.json').exists():print('Six-method campaign not initialized');return 0
        state=read(CAMPAIGN/'state.json')
        print(json.dumps({k:state.get(k) for k in ('status','next_action','remaining','active')},indent=2))
        return 0
    with (SHARED/'master.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller=Controller()
        if args.action=='check':return 0 if controller.execute('check',device='cpu')['status']=='complete' else 1
        if args.action=='preflight':return 0 if controller.execute('preflight',action='preflight')['status']=='complete' else 1
        return controller.run()


if __name__=='__main__':raise SystemExit(main())
