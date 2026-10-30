"""Evaluation-only recovery of unit02 initialization; never runs training."""
import argparse,datetime,json,math,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/home/chakwong/BayesFilter')
sys.path.insert(0,str(ROOT/'docs/benchmarks'))
import run_kdm_unit02_initialization as original
from reference_kdm_native_evaluation import BASE,PIN,read,sha,write
from run_replication_gap_localization import campaign
PLAN='docs/plans/controlled-smc-kdm-unit02-evaluation-repair-plan-20261014.md'
RESULT='docs/plans/controlled-smc-kdm-unit02-evaluation-repair-result-20261014.md'
PARENT=BASE/'unit02-initialization-20261013'
PRIOR=PARENT/'attempt01'
CAP={'gpu':900.,'cpu':300.}
CHANGED={'docs/benchmarks/diagnose_kdm_native_evaluation_tf.py','docs/benchmarks/reference_kdm_recent_endpoint.py'}
def main(preflight=False):
    previous=read(PRIOR/'manifest.json')
    assert previous['status']=='FAILED' and previous['current_step']=='endpoint-native'
    for name in ('control','training','validation','terminal-audit','endpoint-prepare'):
        assert next(x for x in previous['steps'] if x['label']==name)['status']=='COMPLETED',name
    failed=read(PRIOR/'endpoint/native-result.json')
    assert failed['status']=='FAIL' and not failed['records']
    assert failed['checks'][-1]==dict(name='recent checkpoint count',passed=False)
    for name,digest in previous['source_hashes'].items():
        snapshot=PRIOR/'sources'/name
        assert sha(snapshot)==digest,name+' old snapshot'
        if name not in CHANGED:assert sha(ROOT/name)==digest,name+' unexpected drift'
    original.verify_inputs()
    training=PRIOR/'training/diagnostic'
    c=read(PRIOR/'control/diagnostic/result.json');t=read(training/'result.json')
    v=read(PRIOR/'validation/diagnostic/result.json');a=read(PRIOR/'terminal-audit.json')
    tm=read(PRIOR/'training/manifest.json')
    assert c['status']=='PASS' and c['checks_failed']==0 and c['control_max_batch_loss_difference']==0.
    assert t['status']=='PASS' and t['selected_stages_complete'] and t['checks_failed']==0
    assert t['unit']==2 and t['selected_stages']==['mdpf'] and t['seed_base']==1200000000
    assert tm['result_sha256']==sha(training/'result.json')
    assert len(t['completed_stages'])==1 and t['completed_stages'][0]['full_protocol']
    best=Path(t['completed_stages'][0]['best_checkpoint'])
    assert sha(best)==t['completed_stages'][0]['best_checkpoint_sha256']==previous['training_best_checkpoint_sha256']
    assert v['status']=='PASS' and len(v['statistics'])==3 and all(x['n']==8 for x in v['statistics'])
    assert a['verdict']=='PASS' and not a['failures']
    regression=PARENT/'scope-repair-preflight-20261014.json'
    assert read(regression)['status']=='PASS' and len(read(regression)['checks'])==9
    balance=campaign();assert not balance.get('running')
    remaining={r:min(balance['remaining_seconds'][r],balance['tranche_remaining_seconds'][r]) for r in CAP}
    assert all(remaining[r]>=CAP[r] and previous['resource_wall_seconds'][r]+CAP[r]<=original.CAP[r] for r in CAP)
    out=PARENT/'attempt02';assert not out.exists()
    files=set(previous['source_hashes'])|{PLAN,str(Path(__file__).relative_to(ROOT))}
    hashes={name:sha(ROOT/name) for name in sorted(files)}
    reused=[PRIOR/'manifest.json',PRIOR/'control/diagnostic/result.json',training/'result.json',
        PRIOR/'training/manifest.json',best,PRIOR/'validation/diagnostic/result.json',
        PRIOR/'terminal-audit.json',PRIOR/'endpoint/inputs.json',regression]
    reused_hashes={str(p):sha(p) for p in reused}
    if preflight:
        print(json.dumps(dict(status='PREFLIGHT_PASS',output=str(out),reused_training_epochs=t['completed_stages'][0]['epochs'],
            best_checkpoint_sha256=sha(best),source_controller_checks=a['checks'],remaining=remaining,cap_seconds=CAP)),flush=True)
        return 0
    out.mkdir(exist_ok=False)
    for name in hashes:
        dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    gpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES=original.GPU,TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS='1',TF_NUM_INTEROP_THREADS='1',
        TF_NUM_INTRAOP_THREADS='1',BAYESFILTER_PRELOAD_CUSTOM_OP='0')
    gpu_env.pop('LD_LIBRARY_PATH',None)
    cpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',LD_LIBRARY_PATH='/home/chakwong/anaconda3/envs/tf-gpu2/lib',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
    manifest=dict(schema='kdm_unit02_evaluation_recovery_v1',status='RUNNING',
        command=[sys.executable,*sys.argv],cwd=str(ROOT),git_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        supervisor_pid=os.getpid(),plan_file=PLAN,result_file=RESULT,source_pin=PIN,source_hashes=hashes,
        reused_input_hashes=reused_hashes,reused_attempt=str(PRIOR),phase_cap_seconds=CAP,
        prior_phase_seconds=previous['resource_wall_seconds'],campaign_remaining_before=remaining,
        resource_wall_seconds={r:0. for r in CAP},steps=[],input_checks=previous['input_checks'],
        interpreters={'gpu':original.TF,'cpu':original.SOURCE},environment_gpu={k:gpu_env[k] for k in
        ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','OMP_NUM_THREADS','TF_NUM_INTEROP_THREADS','TF_NUM_INTRAOP_THREADS')},
        environment_cpu={k:cpu_env[k] for k in ('CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','OMP_NUM_THREADS')},
        gpu_trust_basis='escalated_local_gpu_access',cpu_gpu_status='TF GPU/XLA FP32 TF32 off; author CPU intentionally hidden',
        seed_policy=previous['seed_policy'],data_version=previous['data_version'],replication='INCOMPLETE')
    def save():write(out/'manifest.json',manifest)
    save()
    original.PLAN=PLAN;original.RESULT=RESULT
    original.refresh_documents(out,manifest)
    endpoint=out/'endpoint';endpoint.mkdir()
    def run(label,command,resource):
        left=CAP[resource]-manifest['resource_wall_seconds'][resource]
        assert left>1
        step=dict(label=label,command=command,resource=resource,status='RUNNING',timeout_seconds=left,log=label+'.log',
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        manifest['steps'].append(step);manifest['current_step']=label;save()
        print(json.dumps(dict(stage=label,output=str(out))),flush=True)
        tick=time.monotonic();child=None
        try:
            with (out/step['log']).open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=gpu_env if resource=='gpu' else cpu_env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                step['pid']=child.pid;save()
                while True:
                    allowance=left-(time.monotonic()-tick)
                    if allowance<=0:raise TimeoutError(label+' exceeded resource cap')
                    try:code=child.wait(timeout=min(30.,allowance));break
                    except subprocess.TimeoutExpired:
                        step['elapsed_seconds']=time.monotonic()-tick
                        if (endpoint/'progress.json').exists():step['worker_progress']=read(endpoint/'progress.json')
                        save()
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
    decision=dict(status='FAIL',replication='INCOMPLETE',next_action='Inspect evaluation-only recovery failure; preserve completed training')
    try:
        run('endpoint-prepare',[original.SOURCE,'-I','-B',str(original.stage_path('reference_kdm_recent_endpoint.py')),
            '--unit','2','--intervention-training',str(training),'--output',str(endpoint)],'cpu')
        inputs=read(endpoint/'inputs.json')
        assert inputs.pop('evaluation_scope')=='unit02_mdpf_initialization'
        assert inputs==read(PRIOR/'endpoint/inputs.json'),'Scientific evaluation inputs changed'
        manifest['unchanged_scientific_inputs']=True;manifest['endpoint_inputs_sha256']=sha(endpoint/'inputs.json');save()
        run('endpoint-native',[original.TF,'-I','-B',str(original.stage_path('diagnose_kdm_native_evaluation_tf.py')),
            '--output',str(endpoint)],'gpu')
        e=read(endpoint/'native-result.json')
        assert e['status']=='PASS' and all(x['passed'] for x in e['checks'])
        assert len(e['records'])==6 and len(e['metric_replays'])==3
        assert e['memory_policy']['all_physical_devices_memory_growth'] and e['jit_compile'] and not e['tf32']
        manifest['memory_policy']=e['memory_policy']
        run('endpoint-metrics',[original.SOURCE,'-I','-B',str(original.stage_path('reference_kdm_native_evaluation.py')),
            '--mode','metrics','--output',str(endpoint)],'cpu')
        m=read(endpoint/'metric-audit.json')
        assert m['status']=='PASS' and not m['failures'] and len(m['records'])==6
        baseline=read(BASE/'recent-endpoint-20261013/attempt01/unit02/native-result.json')
        saved={(x['checkpoint'],x['seed']):x for x in baseline['records'] if x['variant']=='mdpf'}
        rows={(x['checkpoint'],x['seed']):x for x in e['records']}
        expected={(name,seed) for name in ('source_trained','tf_trained','author_start_tf_trained') for seed in (202610082,202610083)}
        assert set(rows)==expected and len(rows)==len(e['records']) and len(saved)==4
        differences=[]
        for key,value in saved.items():
            for metric in ('nll','rmse'):
                diff=abs(rows[key][metric]-value[metric]);assert diff<=1e-5
                differences.append(diff)
        for name,digest in hashes.items():assert sha(ROOT/name)==digest,'source drift '+name
        for name,digest in reused_hashes.items():assert sha(Path(name))==digest,'reused evidence drift '+name
        scores={x['candidate']:x['mean'] for x in v['statistics']}
        nll={name:sum(x['nll'] for x in e['records'] if x['checkpoint']==name)/2.
            for name in ('source_trained','tf_trained','author_start_tf_trained')}
        rmse={name:sum(x['rmse'] for x in e['records'] if x['checkpoint']==name)/2. for name in nll}
        old=scores['old_tf_best']-scores['author_best'];new=scores['author_start_tf_best']-scores['author_best']
        oldend=nll['tf_trained']-nll['source_trained'];newend=nll['author_start_tf_trained']-nll['source_trained']
        decision=dict(status='PASS',replication='INCOMPLETE',scope='UNIT02_INITIALIZATION_INTERVENTION',
            inference='DESCRIPTIVE_UNIT_CONDITIONAL',changed_initial_parameters=previous['input_checks']['changed_parameter_count'],
            control_checks=c['checks_total'],control_max_loss_difference=c['control_max_batch_loss_difference'],
            training_checks=t['checks_total'],training_stages=t['completed_stages'],terminal_audit_checks=a['checks'],
            validation=v['statistics'],validation_old_gap=old,validation_new_gap=new,native_endpoint_nll=nll,
            native_endpoint_rmse=rmse,endpoint_old_gap=oldend,endpoint_new_gap=newend,
            explanation_supported=(old>0 and new<old and oldend>0 and newend<oldend),
            endpoint_checks=len(e['checks']),author_metric_replays=len(m['records']),
            baseline_max_absolute_difference=max(differences),training_repeated=False,
            next_action='Consolidate source-conformance coverage and localize residual valid training differences; no population ladder implied')
        manifest['status']='COMPLETED';manifest['current_step']='complete'
    except BaseException:
        manifest['status']='FAILED';(out/'error.txt').write_text(traceback.format_exc())
    finally:
        write(out/'decision.json',decision);manifest['decision_sha256']=sha(out/'decision.json')
        manifest['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        ledger=BASE/'reference-diagnostics.json';entries=read(ledger)
        for resource in CAP:
            identity='kdm-unit02-initialization-'+out.name+'_'+resource
            assert not any(x['id']==identity for x in entries)
            entries.append(dict(id=identity,resource=resource,charged_seconds=manifest['resource_wall_seconds'][resource],
                summary=str((out/'decision.json').relative_to(ROOT)),summary_sha256=sha(out/'decision.json'),
                manifest=str((out/'manifest.json').relative_to(ROOT)),plan_file=PLAN,result_file=RESULT,
                stage='KDM_UNIT02_INITIALIZATION_EVALUATION_REPAIR',next_action=decision['next_action'],after_attempt=balance['attempts'],
                executions=[dict(path=str(out.relative_to(ROOT)),resource=resource,
                    wall_seconds=manifest['resource_wall_seconds'][resource],status=manifest['status'])]))
        write(ledger,entries)
        original.refresh_documents(out,manifest,decision)
    print(json.dumps(dict(status=manifest['status'],output=str(out),resource_wall_seconds=manifest['resource_wall_seconds'])),flush=True)
    return 0 if manifest['status']=='COMPLETED' else 1
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    raise SystemExit(main(parser.parse_args().preflight))
