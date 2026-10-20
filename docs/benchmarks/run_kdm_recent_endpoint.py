"""Bounded recent-checkpoint native endpoint diagnostic; no training."""
import argparse,datetime,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/home/chakwong/BayesFilter')
sys.path.insert(0,str(ROOT/'docs/benchmarks'))
from run_replication_gap_localization import campaign
from reference_kdm_native_evaluation import BASE,PIN,read,sha,write
TF='/home/chakwong/anaconda3/envs/tftwogpu/bin/python'
SOURCE='/home/chakwong/anaconda3/envs/tf-gpu2/bin/python'
GPU='GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a'
PHASE=BASE/'recent-endpoint-20261013'
PLAN='docs/plans/controlled-smc-kdm-recent-endpoint-plan-20261013.md'
RESULT='docs/plans/controlled-smc-kdm-recent-endpoint-result-20261013.md'
CAP={'gpu':900.,'cpu':300.}

def main(preflight=False):
    balance=campaign();assert not balance['running']
    prior=sorted(PHASE.glob('attempt*/manifest.json'));assert len(prior)<2
    assert all(read(p)['status']!='RUNNING' for p in prior)
    assert not any(read(p)['status']=='COMPLETED' for p in prior)
    spent={r:sum(read(p)['resource_wall_seconds'][r] for p in prior) for r in CAP}
    remaining={r:min(balance['remaining_seconds'][r],balance['tranche_remaining_seconds'][r]) for r in CAP}
    assert all(CAP[r]>spent[r] and remaining[r]>=CAP[r]-spent[r] for r in CAP)
    out=PHASE/('attempt%02d'%(len(prior)+1));assert not out.exists()
    paths=[Path(__file__).resolve(),ROOT/PLAN,
        ROOT/'docs/plans/controlled-smc-kdm-native-evaluation-20261011.md']
    paths += [ROOT/'docs/benchmarks'/name for name in (
        'reference_kdm_recent_endpoint.py','reference_kdm_native_evaluation.py',
        'diagnose_kdm_native_evaluation_tf.py','reference_kdm_neural_exact_replay.py',
        'run_replication_gap_localization.py','run_replication_training_pilot_full.py')]
    paths += [ROOT/'bayesfilter/runtime/gpu_memory_policy.py']
    paths += sorted((ROOT/'bayesfilter/replication').glob('kdm*.py'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    if preflight:
        print(json.dumps(dict(status='PREFLIGHT_PASS',out=str(out),phase_spent=spent,
            campaign_remaining=remaining,files=len(paths))));return 0
    out.mkdir(parents=True,exist_ok=False)
    for p in paths:
        dest=out/'sources'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    gpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS='1',TF_NUM_INTEROP_THREADS='1',
        TF_NUM_INTRAOP_THREADS='1',MPLBACKEND='Agg',BAYESFILTER_PRELOAD_CUSTOM_OP='0')
    gpu_env.pop('LD_LIBRARY_PATH',None)
    cpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',LD_LIBRARY_PATH='/home/chakwong/anaconda3/envs/tf-gpu2/lib',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
    manifest=dict(schema='kdm_recent_endpoint_v1',status='RUNNING',command=[sys.executable,*sys.argv],
        cwd=str(ROOT),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        plan_file=PLAN,result_file=RESULT,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        supervisor_pid=os.getpid(),source_pin=PIN,source_hashes=hashes,input_hashes={},
        phase_cap_seconds=CAP,prior_phase_seconds=spent,campaign_remaining_before=remaining,
        resource_wall_seconds={r:0. for r in CAP},steps=[],completed_units=[],
        interpreters={'gpu':TF,'cpu':SOURCE},gpu_trust_basis='escalated_local_gpu_access',
        environment_gpu={k:gpu_env[k] for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','OMP_NUM_THREADS')},
        environment_cpu={k:cpu_env[k] for k in ('CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','OMP_NUM_THREADS')},
        seed_policy={'filter_seeds':[202610082,202610083],'data':'unchanged recorded unit evaluation data'},
        data_version='fixed calibration units01/02 data; per-input SHA256',
        cpu_gpu_status='author reference CPU intentionally hidden; native TF GPU/XLA FP32 TF32 off',
        replication='INCOMPLETE')
    def save():write(out/'manifest.json',manifest)
    save()
    def run(label,command,resource):
        allowance=CAP[resource]-spent[resource]-manifest['resource_wall_seconds'][resource]
        if allowance<=0:raise RuntimeError('Resource cap exhausted')
        step=dict(label=label,command=command,resource=resource,status='RUNNING',timeout_seconds=allowance,log=label+'.log')
        manifest['steps'].append(step);save();tick=time.monotonic();child=None
        print(json.dumps(dict(stage=label,resource=resource,output=str(out))),flush=True)
        try:
            with (out/step['log']).open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=gpu_env if resource=='gpu' else cpu_env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                step['pid']=child.pid;save()
                while True:
                    left=allowance-(time.monotonic()-tick)
                    if left<=0:raise TimeoutError(label+' PHASE_TIME_LIMIT')
                    try:code=child.wait(timeout=min(30.,left));break
                    except subprocess.TimeoutExpired:
                        step['elapsed_seconds']=time.monotonic()-tick;save()
            step.update(status='COMPLETED' if code==0 else 'FAILED',returncode=code)
            if code:raise RuntimeError(label+' exited '+str(code))
        except BaseException:
            step['status']='FAILED'
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
        finally:
            step['wall_seconds']=time.monotonic()-tick
            manifest['resource_wall_seconds'][resource]+=step['wall_seconds'];save()
    decision=dict(status='FAIL',replication='INCOMPLETE',next_action='Localize failed endpoint check')
    try:
        run('nvidia',['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.total','--format=csv'],'gpu')
        rows=[];checks=0;metric_rows=[]
        for unit in (1,2):
            dest=out/('unit%02d'%unit);dest.mkdir()
            run('unit%d-prepare'%unit,[SOURCE,'-I','-B',str(ROOT/'docs/benchmarks/reference_kdm_recent_endpoint.py'),
                '--unit',str(unit),'--output',str(dest)],'cpu')
            inputs=read(dest/'inputs.json');manifest['input_hashes'].update(inputs['provenance'])
            manifest['input_hashes'][str(dest/'inputs.json')]=sha(dest/'inputs.json');save()
            run('unit%d-native'%unit,[TF,'-I','-B',str(ROOT/'docs/benchmarks/diagnose_kdm_native_evaluation_tf.py'),
                '--output',str(dest)],'gpu')
            native=read(dest/'native-result.json');n=5 if unit==1 else 4
            assert native['status']=='PASS' and len(native['records'])==2*n and len(native['metric_replays'])==n
            assert native['memory_policy']['all_physical_devices_memory_growth']
            assert native['jit_compile'] and not native['tf32'] and native['visible_device']==GPU
            run('unit%d-metrics'%unit,[SOURCE,'-I','-B',str(ROOT/'docs/benchmarks/reference_kdm_native_evaluation.py'),
                '--mode','metrics','--output',str(dest)],'cpu')
            audit=read(dest/'metric-audit.json')
            assert audit['status']=='PASS' and not audit['failures'] and len(audit['records'])==2*n and audit['source_pin']==PIN
            rows.extend(dict(unit=unit,**row) for row in native['records'])
            metric_rows.extend(dict(unit=unit,**row) for row in audit['records'])
            checks+=len(native['checks'])
            manifest['completed_units'].append(dict(unit=unit,native_result_sha256=sha(dest/'native-result.json'),
                metric_audit_sha256=sha(dest/'metric-audit.json'),memory_policy=native['memory_policy'],
                allocator=native['allocator']));save()
        for name,digest in hashes.items():assert sha(ROOT/name)==digest,'Source drift '+name
        for name,digest in manifest['input_hashes'].items():assert sha(name)==digest,'Input drift '+name
        decision=dict(status='PASS',replication='INCOMPLETE',scope='NINE_RECENT_CHECKPOINTS_TWO_SEEDS',
            inference='DESCRIPTIVE_ONLY',records=rows,checks=checks,metric_checks=metric_rows,
            next_action='Interpret recent native endpoint coverage; unit02 initial-weight intervention remains separate')
        manifest['status']='COMPLETED'
    except BaseException:
        manifest['status']='FAILED';(out/'error.txt').write_text(traceback.format_exc())
    finally:
        write(out/'decision.json',decision);manifest['decision_sha256']=sha(out/'decision.json')
        manifest['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        ledger=BASE/'reference-diagnostics.json';entries=read(ledger)
        for r in CAP:
            identity='kdm-recent-endpoint-'+out.name+'_'+r
            assert not any(x['id']==identity for x in entries)
            entries.append(dict(id=identity,resource=r,charged_seconds=manifest['resource_wall_seconds'][r],
                summary=str((out/'decision.json').relative_to(ROOT)),summary_sha256=sha(out/'decision.json'),
                manifest=str((out/'manifest.json').relative_to(ROOT)),plan_file=PLAN,result_file=RESULT,
                stage='KDM_RECENT_ENDPOINT_DIAGNOSTIC',next_action=decision['next_action'],after_attempt=balance['attempts'],
                executions=[dict(path=str(out.relative_to(ROOT)),resource=r,
                    wall_seconds=manifest['resource_wall_seconds'][r],status=manifest['status'])]))
        write(ledger,entries)
    print(json.dumps(dict(status=manifest['status'],decision=decision['status'],output=str(out),
        resource_wall_seconds=manifest['resource_wall_seconds'])),flush=True)
    return 0 if manifest['status']=='COMPLETED' else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    raise SystemExit(main(parser.parse_args().preflight))
