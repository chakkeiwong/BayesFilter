"""Bounded, replication-only pretraining cloud-law supervisor."""
import datetime,hashlib,json,os,signal,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/home/chakwong/BayesFilter')
BASE=ROOT/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
PHASE=BASE/'pretraining-cloud-law-20261013'
PLAN='docs/plans/controlled-smc-kdm-cloud-law-plan-20261013.md'
RESULT='docs/plans/controlled-smc-kdm-cloud-law-result-20261013.md'
TF='/home/chakwong/anaconda3/envs/tftwogpu/bin/python'
SOURCE='/home/chakwong/anaconda3/envs/tf-gpu2/bin/python'
GPU='GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def main():
    sys.path.insert(0,str(ROOT/'docs/benchmarks'))
    from run_replication_gap_localization import campaign
    balance=campaign();assert not balance['running']
    prior=list(PHASE.glob('attempt*/manifest.json'));assert len(prior)<2
    assert all(read(p)['status']!='RUNNING' for p in prior)
    remaining={r:min(1200-sum(read(p)['resource_wall_seconds'][r] for p in prior),
        balance['remaining_seconds'][r],balance['tranche_remaining_seconds'][r]) for r in ('cpu','gpu')}
    limit=min(600,min(remaining.values()));assert limit>180
    out=PHASE/('attempt%02d'%(len(prior)+1));out.mkdir(parents=True,exist_ok=False)
    paths=[PLAN,'docs/benchmarks/run_kdm_pretraining_cloud_law.py',
        'docs/benchmarks/diagnose_kdm_pretraining_cloud_law.py',
        'docs/benchmarks/diagnose_kdm_fresh_pretraining.py',
        'docs/benchmarks/reference_kdm_source_stages.py','docs/benchmarks/reference_kdm_neural.py',
        'bayesfilter/runtime/gpu_memory_policy.py']
    paths += ['bayesfilter/replication/'+n for n in ('kdm_neural_replay_tf.py','kdm_source_stages_tf.py',
        'kdm_training_controller.py','kdm_sampling_tf.py','kdm_normal_von_mises_tf.py')]
    for name in paths:
        dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    commands=[('cloud_law',[TF,'-I','-B',str(ROOT/'docs/benchmarks/diagnose_kdm_pretraining_cloud_law.py'),
        '--output',str(out/'diagnostic')])]
    manifest=dict(status='RUNNING',supervisor_pid=os.getpid(),commands=commands,plan_file=PLAN,result_file=RESULT,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_dirty=subprocess.check_output(['git','status','--short','--untracked-files=no'],cwd=ROOT,text=True),
        start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_hashes={n:sha(ROOT/n) for n in paths},environments=dict(tf=TF,author=SOURCE),
        seed=202610134,attempt_limit=2,phase_caps=dict(cpu=1200,gpu=1200),per_attempt_limit=limit,
        campaign_remaining_before=remaining,resource_wall_seconds=dict(cpu=0.,gpu=0.),
        device_policy='RTX5080 GPU/XLA FP32 TF32 off memory growth; author CPU CUDA hidden',
        gpu_trust_basis='escalated_local_gpu_access',
        data_version='unit01 saved validation inputs; per-input hashes in diagnostic artifacts',replication='INCOMPLETE')
    write(out/'manifest.json',manifest)
    start=time.monotonic();gpu_start=None;proc=None;decision=dict(status='FAILED')
    try:
        for label,command in commands:
            gpu=label=='cloud_law'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU if gpu else '-1',OMP_NUM_THREADS='1',
                TF_NUM_INTEROP_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_FORCE_GPU_ALLOW_GROWTH='true',
                BAYESFILTER_PRELOAD_CUSTOM_OP='0',LD_LIBRARY_PATH=str(Path(TF if gpu else SOURCE).parents[1]/'lib'))
            if gpu:
                gpu_start=time.monotonic()
                with (out/'nvidia-smi.log').open('w') as log:
                    subprocess.run(['nvidia-smi','--query-gpu=name,uuid,memory.used,memory.total','--format=csv'],
                        stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
            with (out/(label+'.log')).open('w') as log:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                manifest.update(active_stage=label,worker_pid=proc.pid);write(out/'manifest.json',manifest)
                print(json.dumps(dict(stage=label,pid=proc.pid,output=str(out))),flush=True)
                code=proc.wait(timeout=max(1,limit-(time.monotonic()-start)))
                if code:raise RuntimeError(label+' exited '+str(code))
            result=read(out/('diagnostic' if gpu else 'controller')/'result.json')
            if not gpu:assert result['status']=='PASS'
        decision=dict(status=result['status'],cloud_law_result=str(out/'diagnostic/result.json'),
            whole_kdm_replication='INCOMPLETE',scope=result['scope'])
        manifest['status']='COMPLETED'
    except BaseException:
        manifest['status']='FAILED';decision['error']=traceback.format_exc()
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        elapsed=time.monotonic()-start
        costs=dict(cpu=elapsed,gpu=0. if gpu_start is None else time.monotonic()-gpu_start)
        manifest.update(resource_wall_seconds=costs,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(out/'decision.json',decision);write(out/'manifest.json',manifest)
        ledger=BASE/'reference-diagnostics.json';records=read(ledger)
        for r in ('cpu','gpu'):
            identity='kdm-pretraining-cloud-law-'+out.name+'_'+r
            assert not any(x['id']==identity for x in records)
            records.append(dict(id=identity,resource=r,charged_seconds=costs[r],
                summary=str((out/'decision.json').relative_to(ROOT)),summary_sha256=sha(out/'decision.json'),
                manifest=str((out/'manifest.json').relative_to(ROOT)),plan_file=PLAN,result_file=RESULT,
                stage='KDM_PRETRAINING_CLOUD_LAW_DIAGNOSTIC',after_attempt=balance['attempts'],
                next_action='Interpret composed native cloud-law diagnostic; no automatic qualification',
                executions=[dict(path=str(out.relative_to(ROOT)),resource=r,wall_seconds=costs[r],status=manifest['status'])]))
        write(ledger,records)
    print(json.dumps(dict(status=manifest['status'],decision=decision,output=str(out),resource_wall_seconds=costs)),flush=True)
    return 0 if manifest['status']=='COMPLETED' else 1
if __name__=='__main__':raise SystemExit(main())
