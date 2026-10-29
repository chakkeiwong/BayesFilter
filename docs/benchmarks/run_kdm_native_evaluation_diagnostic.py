"""Bounded native KDM evaluation diagnostic under the replication master plan."""
from __future__ import annotations
import argparse,datetime,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from run_replication_training_pilot_full import ROOT,BASE,TF,SOURCE,GPU,PIN,sha,read,write
from run_replication_gap_localization import campaign
PHASE=BASE/'native-evaluation-20261011'
PILOT=BASE/'repaired-training-20261011/training-pilot-02'
PLAN='docs/plans/controlled-smc-kdm-native-evaluation-20261011.md'
RESULT='docs/plans/controlled-smc-kdm-mechanism-result-20261011.md'
CAP={'gpu':1800.,'cpu':300.}

def main(out,preflight=False):
    if out.parent!=PHASE or out.name not in ('native-evaluation-01','native-evaluation-02'):
        raise ValueError('Use a reviewed versioned output directory')
    prior=sorted(PHASE.glob('native-evaluation-*/manifest.json'))
    if len(prior)>=2 or out.name!='native-evaluation-%02d'%(len(prior)+1) or out.exists():
        raise RuntimeError('Attempt sequence or existing output mismatch')
    spent={r:0. for r in CAP}
    for p in prior:
        m=read(p)
        if m['status']=='RUNNING': raise RuntimeError('A native evaluation is active')
        if m['status']=='COMPLETED': raise RuntimeError('Native diagnostic already completed')
        for r in CAP: spent[r]+=m['resource_wall_seconds'][r]
    pilot=read(PILOT/'manifest.json');decision=read(PILOT/'decision.json')
    audit=read(PILOT/'audit.json');report=read(PILOT/'evaluation/diagnostic/result.json')
    if (pilot['status']!='COMPLETED' or decision['status']!='DEVELOPMENT_PAIR_COMPLETED' or
        pilot['decision_sha256']!=sha(PILOT/'decision.json') or audit['verdict']!='PASS' or
        audit['failures'] or report['decision']['status']!='PASS' or report['source_pin']!=PIN or
        pilot['evaluation_sha256']!=sha(PILOT/'evaluation/diagnostic/result.json')):
        raise RuntimeError('Pilot terminal evidence is not valid')
    for name,digest in pilot['source_hashes'].items():
        if name.startswith('bayesfilter/') and sha(ROOT/name)!=digest:
            raise RuntimeError('Runtime changed after complete training: '+name)
    balance=campaign()
    remaining={r:min(balance['remaining_seconds'][r],balance['tranche_remaining_seconds'][r]) for r in CAP}
    for r in CAP:
        if CAP[r]-spent[r]<=0 or remaining[r]<CAP[r]-spent[r]:
            raise RuntimeError('Insufficient phase or campaign budget: '+r)
    paths=[Path(__file__).resolve(),ROOT/PLAN,
        ROOT/'docs/benchmarks/reference_kdm_native_evaluation.py',
        ROOT/'docs/benchmarks/diagnose_kdm_native_evaluation_tf.py',
        ROOT/'docs/benchmarks/reference_kdm_neural_exact_replay.py',
        ROOT/'docs/benchmarks/run_replication_training_pilot_full.py',
        ROOT/'docs/benchmarks/run_replication_gap_localization.py',
        ROOT/'bayesfilter/runtime/gpu_memory_policy.py']
    paths+=sorted((ROOT/'bayesfilter/replication').glob('kdm*.py'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    if preflight:
        print(json.dumps(dict(status='PREFLIGHT_PASS',prior_phase_seconds=spent,
            campaign_remaining_seconds=remaining,code_files=len(paths),output=str(out))))
        return 0
    out.mkdir(parents=True,exist_ok=False)
    for p in paths:
        target=out/'sources'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
    gpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS='1',TF_NUM_INTEROP_THREADS='1',TF_NUM_INTRAOP_THREADS='1',MPLBACKEND='Agg')
    gpu_env.pop('LD_LIBRARY_PATH',None)
    cpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',LD_LIBRARY_PATH='/home/chakwong/anaconda3/envs/tf-gpu2/lib',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
    manifest=dict(schema='kdm_native_evaluation_diagnostic_v1',status='RUNNING',command=[sys.executable,*sys.argv],
        cwd=str(ROOT),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        plan_file=PLAN,result_file=RESULT,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        supervisor_pid=os.getpid(),source_pin=PIN,source_hashes=hashes,
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in
            (PILOT/'manifest.json',PILOT/'decision.json',PILOT/'audit.json',PILOT/'evaluation/diagnostic/result.json')},
        phase_cap_seconds=CAP,prior_phase_seconds=spent,campaign_remaining_before=remaining,
        resource_wall_seconds={r:0. for r in CAP},steps=[],replication='INCOMPLETE',
        seeds={'evaluation_data':202610081,'filter':[202610082,202610083]},
        cpu_gpu_status='Native TF GPU/XLA escalated RTX5080; source preparation/metric replay deliberately CPU only',
        environment_gpu={k:gpu_env.get(k) for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH',
            'TF_NUM_INTEROP_THREADS','TF_NUM_INTRAOP_THREADS','OMP_NUM_THREADS')},
        environment_cpu={k:cpu_env.get(k) for k in ('CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH',
            'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    def save(): write(out/'manifest.json',manifest)
    def run(label,command,resource,env):
        allowance=min(CAP[resource]-spent[resource],remaining[resource])-manifest['resource_wall_seconds'][resource]
        if allowance<=0: raise RuntimeError('Resource cap exhausted')
        step=dict(label=label,command=command,resource=resource,status='RUNNING',timeout_seconds=allowance,log=label+'.log')
        manifest['steps'].append(step);save();tick=time.monotonic();child=None
        try:
            with (out/step['log']).open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                step['pid']=child.pid;save()
                while True:
                    time_left=allowance-(time.monotonic()-tick)
                    if time_left<=0: raise TimeoutError(label+' PHASE_TIME_LIMIT')
                    try: code=child.wait(timeout=min(30.,time_left));break
                    except subprocess.TimeoutExpired:
                        step['elapsed_seconds']=time.monotonic()-tick;save()
            step.update(status='COMPLETED' if code==0 else 'FAILED',returncode=code)
            if code: raise RuntimeError(label+' exited '+str(code)+'; see '+step['log'])
        except BaseException:
            step['status']='FAILED'
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
        finally:
            step['wall_seconds']=time.monotonic()-tick
            manifest['resource_wall_seconds'][resource]+=step['wall_seconds'];save()
    save();terminal=dict(status='ERROR',replication='INCOMPLETE',scope='NATIVE_ENDPOINT_DIAGNOSTIC')
    try:
        reference=ROOT/'docs/benchmarks/reference_kdm_native_evaluation.py'
        run('prepare',[SOURCE,'-I','-B',str(reference),'--mode','prepare','--output',str(out)],'cpu',cpu_env)
        inputs=read(out/'inputs.json')
        manifest['dataset_sha256']=inputs['dataset_sha256'];manifest['native_inputs_sha256']=sha(out/'inputs.json');save()
        run('nvidia',['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.total','--format=csv'],'gpu',gpu_env)
        run('native',[TF,'-I','-B',str(ROOT/'docs/benchmarks/diagnose_kdm_native_evaluation_tf.py'),'--output',str(out)],'gpu',gpu_env)
        native=read(out/'native-result.json')
        if (native['status']!='PASS' or len(native['records'])!=8 or
            not native['memory_policy']['all_physical_devices_memory_growth'] or
            native['visible_device']!=GPU or not native['jit_compile'] or native['tf32']):
            raise RuntimeError('Native result violates the reviewed evidence contract')
        run('metrics',[SOURCE,'-I','-B',str(reference),'--mode','metrics','--output',str(out)],'cpu',cpu_env)
        metrics=read(out/'metric-audit.json')
        if metrics['status']!='PASS' or len(metrics['records'])!=8 or metrics['failures']:
            raise RuntimeError('Independent source metric audit failed')
        for name,digest in manifest['source_hashes'].items():
            if sha(ROOT/name)!=digest: raise RuntimeError('Source drift: '+name)
        for name,digest in manifest['input_hashes'].items():
            if sha(ROOT/name)!=digest: raise RuntimeError('Input drift: '+name)
        manifest.update(status='COMPLETED',native_result_sha256=sha(out/'native-result.json'),
            metric_audit_sha256=sha(out/'metric-audit.json'),memory_policy=native['memory_policy'])
        terminal=dict(status='NATIVE_ENDPOINT_AND_METRICS_PASS',replication='INCOMPLETE',
            scope='FOUR_CHECKPOINTS_TWO_NATIVE_SEEDS',inference='DESCRIPTIVE_ONLY',
            metrics_max_scaled_error=max(x['max_scaled_error'] for x in metrics['records']),
            records=[{k:x[k] for k in ('variant','checkpoint','seed','nll','rmse')} for x in native['records']],
            next_action='Assess independent multi-training-unit qualification within the existing campaign budget')
    except BaseException:
        manifest['status']='FAILED';(out/'error.txt').write_text(traceback.format_exc())
    finally:
        write(out/'decision.json',terminal);manifest['decision_sha256']=sha(out/'decision.json')
        manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        ledger=BASE/'reference-diagnostics.json';records=read(ledger)
        for resource in CAP:
            record=dict(id=out.name+'_'+resource,resource=resource,charged_seconds=manifest['resource_wall_seconds'][resource],
                summary=str((out/'decision.json').relative_to(ROOT)),summary_sha256=sha(out/'decision.json'),
                manifest=str((out/'manifest.json').relative_to(ROOT)),plan_file=PLAN,result_file=RESULT,
                stage='KDM_NATIVE_ENDPOINT_DIAGNOSTIC',
                next_action='Assess independent multi-training-unit qualification within the existing campaign budget',
                after_attempt=balance['attempts'],
                executions=[dict(path=str(out.relative_to(ROOT)),resource=resource,wall_seconds=manifest['resource_wall_seconds'][resource])])
            if any(x['id']==record['id'] for x in records): raise RuntimeError('Duplicate budget ID')
            records.append(record)
        write(ledger,records)
    print(json.dumps(dict(status=manifest['status'],decision=terminal['status'],resource_wall_seconds=manifest['resource_wall_seconds'],output=str(out))))
    return 0 if manifest['status']=='COMPLETED' else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    raise SystemExit(main(a.output,a.preflight))
