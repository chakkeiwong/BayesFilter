"""Resume only native CPU evaluation of the preserved KDM stage-isolation run."""
import datetime, hashlib, json, os, shutil, signal, subprocess, sys, time, traceback
from pathlib import Path
ROOT = Path('/home/chakwong/BayesFilter')
BASE = ROOT / 'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
PHASE = BASE / 'stage-isolation-20261013'
ORIGIN = PHASE / 'attempt03'
PLAN = 'docs/plans/controlled-smc-kdm-stage-isolation-plan-20261013.md'
RESULT = 'docs/plans/controlled-smc-kdm-stage-isolation-result-20261013.md'
EVALUATOR = ROOT / 'docs/benchmarks/reference_kdm_stage_isolation.py'
SOURCEPY = '/home/chakwong/anaconda3/envs/tf-gpu2/bin/python'
CAPS = dict(cpu=16200., gpu=14400.)
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, value):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)
def main():
    sys.path.insert(0, str(ROOT / 'docs/benchmarks'))
    from run_replication_gap_localization import campaign
    balance = campaign()
    assert not balance['running'], 'Original campaign already active'
    history = list(PHASE.glob('attempt*/manifest.json')) + list(PHASE.glob('evaluation-repair*/manifest.json'))
    prior = [read(p) for p in history]
    assert all(x['status'] != 'RUNNING' for x in prior), 'Another stage-isolation process recorded active'
    retries = list(PHASE.glob('evaluation-repair*/manifest.json'))
    assert len(retries) < 2, 'Evaluation-only repair limit reached'
    usage = {r: sum(x['resource_wall_seconds'][r] for x in prior) for r in CAPS}
    available = {r: min(CAPS[r]-usage[r], balance['remaining_seconds'][r], balance['tranche_remaining_seconds'][r]) for r in CAPS}
    timeout = min(1800., available['cpu'])
    assert timeout >= 30, 'Insufficient remaining CPU allocation'
    trained_dir = ORIGIN / 'training/diagnostic'
    trained = read(trained_dir / 'result.json')
    assert trained['status'] == 'PASS' and trained['checks_failed'] == 0 and trained['selected_stages_complete']
    completed = trained['completed_stages']
    assert len(completed) == 1 and completed[0]['stage'] == 'mdpf' and completed[0]['full_protocol']
    best = trained_dir / 'mdpf-best.json'
    assert sha(best) == completed[0]['best_checkpoint_sha256'] == 'b39244cc5c5aada6a24f50be1cd9cb615cb48d5352108e0f66f93af590db5468'
    origin = read(ORIGIN / 'manifest.json')
    assert origin['status'] == 'FAILED' and read(ORIGIN / 'decision.json')['failed_stage'] == 'evaluation'
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert commit == origin['git_commit'], 'Checkout changed since training'
    for name, expected in origin['source_hashes'].items():
        if name.startswith('bayesfilter/') or (name.startswith('docs/benchmarks/') and name != str(EVALUATOR.relative_to(ROOT))):
            assert sha(ROOT / name) == expected, 'Training/reference dependency changed: ' + name
    out = PHASE / ('evaluation-repair%02d' % (len(retries)+1))
    out.mkdir(exist_ok=False)
    files = [Path(__file__).resolve(), EVALUATOR, ROOT / PLAN,
             ROOT/'docs/benchmarks/reference_kdm_source_stages.py',
             ROOT/'docs/benchmarks/reference_kdm_neural.py',
             ROOT/'docs/benchmarks/reference_kdm_neural_exact_replay.py']
    for p in files:
        target=out/'sources'/p.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,target)
    inputs = [best, trained_dir/'result.json', ORIGIN/'manifest.json', ORIGIN/'decision.json']
    input_hashes = {str(p.relative_to(ROOT)):sha(p) for p in inputs}
    source_hashes = {str(p.relative_to(ROOT)):sha(p) for p in files}
    cmd = [SOURCEPY,'-I','-B',str(EVALUATOR),'--mode','evaluate','--trained',str(trained_dir),'--output',str(out/'diagnostic')]
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',LD_LIBRARY_PATH='/home/chakwong/anaconda3/envs/tf-gpu2/lib')
    manifest = dict(status='RUNNING',supervisor_pid=os.getpid(),plan_file=PLAN,result_file=RESULT,
        command=cmd,supervisor_command=[sys.executable,'-I','-B',str(Path(__file__).resolve())],
        start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),git_commit=commit,
        git_dirty=subprocess.check_output(['git','status','--short','--untracked-files=no'],cwd=ROOT,text=True),
        input_hashes=input_hashes,source_hashes=source_hashes,source_training_origin=str(ORIGIN),
        phase_caps=CAPS,phase_usage_before=usage,campaign_remaining_before=balance['remaining_seconds'],
        timeout_seconds=timeout,resource_wall_seconds=dict(cpu=0.,gpu=0.),
        cpu_only=True,gpu_intentionally_hidden=True,environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','LD_LIBRARY_PATH')},
        data_version='Unchanged unit01 1000 validation sequences, hashes in diagnostic inputs',
        seed_policy='Unchanged 2026101310 + 900000 + candidate_index*1000 + draw; eight draws each',
        replication='INCOMPLETE',repair='Native YAML preserves integer dimension keys; no training change')
    write(out/'manifest.json',manifest)
    decision=dict(status='FAILED',replication='INCOMPLETE',scope='evaluation-only recovery of fixed checkpoint intervention')
    proc=None; started=time.monotonic()
    try:
        with (out/'execution.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            manifest['pid']=proc.pid; write(out/'manifest.json',manifest)
            while proc.poll() is None:
                elapsed=time.monotonic()-started
                worker=out/'diagnostic/progress.json'
                write(out/'progress.json',dict(terminal=False,stage='evaluation',pid=proc.pid,wall_seconds=elapsed,cap_seconds=timeout,worker=read(worker) if worker.exists() else {}))
                if elapsed>=timeout: raise TimeoutError('CPU evaluation cap exhausted')
                time.sleep(5)
            manifest['exit_code']=proc.returncode
            if proc.returncode: raise RuntimeError('Evaluator exited '+str(proc.returncode))
        evaluation=read(out/'diagnostic/result.json')
        assert evaluation['status']=='PASS', 'Native evaluation did not pass'
        assert len(evaluation['statistics'])==3 and all(x['n']==8 for x in evaluation['statistics'])
        for name, expected in {**source_hashes,**input_hashes}.items():
            assert sha(ROOT/name)==expected, 'Input/source changed during evaluation: '+name
        decision.update(status='PASS',training_stages=completed,native_statistics=evaluation['statistics'],
            checks=evaluation['checks'],inference='Checkpoint-conditional descriptive evidence only; no population ranking',
            next_action='Review native comparison and stage attribution; KDM completion not inferred automatically')
        manifest['status']='COMPLETED'; manifest['result_sha256']=sha(out/'diagnostic/result.json')
    except BaseException:
        decision.update(error=traceback.format_exc(),next_action='Inspect exact failure before any bounded retry; never repeat completed training')
        manifest['status']='FAILED'
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
        elapsed=time.monotonic()-started
        manifest['resource_wall_seconds']['cpu']=elapsed
        manifest['end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(out/'decision.json',decision); write(out/'manifest.json',manifest)
        ledger=BASE/'reference-diagnostics.json'; records=read(ledger)
        identity='kdm-stage-isolation-'+out.name+'_cpu'
        assert not any(x['id']==identity for x in records), 'Duplicate resource charge'
        records.append(dict(id=identity,resource='cpu',charged_seconds=elapsed,
            summary=str((out/'decision.json').relative_to(ROOT)),summary_sha256=sha(out/'decision.json'),
            manifest=str((out/'manifest.json').relative_to(ROOT)),plan_file=PLAN,result_file=RESULT,
            stage='KDM_STAGE_ISOLATION_EVALUATION_RECOVERY',after_attempt=balance['attempts'],
            next_action=decision['next_action'],executions=[dict(path=str(out.relative_to(ROOT)),resource='cpu',wall_seconds=elapsed,status=manifest['status'])]))
        write(ledger,records)
        write(out/'progress.json',dict(terminal=True,status=manifest['status'],decision=str(out/'decision.json')))
    print(json.dumps(dict(status=manifest['status'],output=str(out),resources=manifest['resource_wall_seconds'])),flush=True)
    return 0 if manifest['status']=='COMPLETED' else 1
if __name__=='__main__': raise SystemExit(main())
