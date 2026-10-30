"""Bounded supervisor for the unit02 KDM MDPF initialization intervention."""
import argparse, datetime, hashlib, json, math, os, shutil, signal, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path('/home/chakwong/BayesFilter')
sys.path.insert(0,str(ROOT/'docs/benchmarks'))
from run_replication_gap_localization import campaign
from reference_kdm_native_evaluation import BASE,PIN,read,sha,write
TF='/home/chakwong/anaconda3/envs/tftwogpu/bin/python'
SOURCE='/home/chakwong/anaconda3/envs/tf-gpu2/bin/python'
GPU='GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a'
PHASE=BASE/'unit02-initialization-20261013'
PLAN='docs/plans/controlled-smc-kdm-unit02-initialization-plan-20261013.md'
RESULT='docs/plans/controlled-smc-kdm-unit02-initialization-result-20261013.md'
CAP={'gpu':21600.,'cpu':3600.}
PRIOR=BASE/'stage-isolation-20261013/attempt03'
OLD=BASE/'paired-calibration-20261011/calibration-01/unit02/training/diagnostic'
AUTHOR_START=PRIOR/'source-audit/diagnostic/unit02-author-start.json'
VERIFICATION=PRIOR/'verification/diagnostic/result.json'
PREFIXES=('weight_model.','observation_encoder.','particle_encoder_for_weights_model.')
def stage_path(name):return ROOT/'docs/benchmarks'/name
def verify_inputs():
    assert sha(AUTHOR_START) and sha(VERIFICATION)
    prior_manifest=read(PRIOR/'verification/manifest.json')
    assert prior_manifest['status']=='COMPLETED' and sha(VERIFICATION)==prior_manifest['result_sha256']
    proof=read(VERIFICATION);assert proof['status']=='PASS' and proof['mode']=='verify' and proof['stage_transfers_checked']
    for name,digest in prior_manifest['hashes'].items():
        if name.startswith('bayesfilter/replication/'):assert sha(ROOT/name)==digest,name
    audit=read(PRIOR/'source-audit/diagnostic/result.json')
    assert audit['status']=='PASS' and audit['checks']>=484 and audit['source_commit']==PIN
    for name,digest in audit['input_hashes'].items():assert sha(ROOT/name)==digest,name
    author=read(AUTHOR_START);old=read(OLD/'mdpf-starting-state.json')
    assert set(author)==set(old) and set(author['parameters'])==set(old['parameters'])
    assert all(author[k]==old[k] for k in author if k!='parameters')
    changed=[k for k in author['parameters'] if author['parameters'][k]!=old['parameters'][k]]
    assert changed and all(k.startswith(PREFIXES) for k in changed)
    assert author['step']==old['step']==0.
    original=read(OLD/'result.json')
    assert original['status']=='PASS' and original['source_training_protocol_complete']
    aud=read(BASE/'paired-calibration-20261011/calibration-01/unit02/audit.json')
    assert aud['verdict']=='PASS' and not aud['failures']
    source=read(BASE/'kdm-source-variability-20261008-01/unit02-full/diagnostic/result.json')
    assert source['status']=='PASS' and source['source_training_protocol_complete']
    for info in source['datasets'].values():assert sha(info['path'])==info['sha256']
    for info in source['fixtures'].values():assert sha(info['path'])==info['sha256']
    for name,digest in audit['author_hashes'].items():
        assert sha(ROOT/'.localresources/code/younis-mdpf-neurips-2023'/name)==digest
    return dict(changed_parameter_count=len(changed),author_start_sha256=sha(AUTHOR_START),
        original_start_sha256=sha(OLD/'mdpf-starting-state.json'),
        verification_sha256=sha(VERIFICATION),source_audit_sha256=sha(PRIOR/'source-audit/diagnostic/result.json'),
        original_audit_sha256=sha(BASE/'paired-calibration-20261011/calibration-01/unit02/audit.json'))

def refresh_documents(out,manifest,decision=None):
    """Keep the master and concise checkpoint tied to this attempt's manifest."""
    status=manifest['status'];stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    link=str(out.relative_to(ROOT));terminal=status!='RUNNING'
    detail=('Control replay, source-prescribed MDPF training, native validation and T150 evaluation are executing.'
            if not terminal else ('All declared stages completed; interpretation remains unit-conditional.'
                                 if status=='COMPLETED' else 'A stage failed; preserve its logs and inspect the first failure.'))
    if decision and decision.get('status')=='PASS':
        detail+=(f" Validation NLL gap: {decision['validation_old_gap']:.6f} -> {decision['validation_new_gap']:.6f} nat."
                 f" T150 NLL gap: {decision['endpoint_old_gap']:.6f} -> {decision['endpoint_new_gap']:.6f} nat.")
    block=(f"Updated {stamp}. Unit02 initialization intervention: **{status}**.\n\n{detail}\n\n"
           f"Run: {link}; supervisor PID {manifest['supervisor_pid']}. "
           f"GPU seconds charged: {manifest['resource_wall_seconds']['gpu']:.3f}; "
           f"CPU-reference seconds charged: {manifest['resource_wall_seconds']['cpu']:.3f}. "
           f"Phase limits remain 21600 GPU /3600 CPU seconds across at most two attempts.\n\n"
           f"Plan: [unit02 initialization]({Path(PLAN).name}). "
           f"Result: [unit02 result]({Path(RESULT).name}). Whole KDM replication remains incomplete.\n")
    master=ROOT/'docs/plans/controlled-smc-kdm-replication-master-program-20260928.md'
    text=master.read_text();start='<!-- UNIT02_INITIALIZATION_BEGIN -->';end='<!-- UNIT02_INITIALIZATION_END -->'
    if start in text and end in text:
        first=text.index(start)+len(start);last=text.index(end,first)
        text=text[:first]+'\n'+block+text[last:];master.write_text(text)
    remaining={r:manifest['campaign_remaining_before'][r]-manifest['resource_wall_seconds'][r] for r in CAP}
    checkpoint=(f"# Controlled-SMC / KDM active checkpoint\n\nUpdated {stamp}.\n"
        f"Checkout {ROOT}, branch surrogate-hmc, HEAD {manifest['git_commit']}. Preserve unrelated edits.\n\n"
        "Question: does replacing only unit02's initial measurement weights explain its saved MDPF gap?\n"
        "Replication only; cSMC bounded fidelity accepted; whole KDM incomplete; Task3 and population-quality certification excluded.\n\n"
        f"Current stage: {status}. {detail}\nRun: {link}\nSupervisor PID: {manifest['supervisor_pid']}.\n"
        "Exact next action: inspect this run's manifest and child progress; if active, monitor without launching a duplicate. "
        "After training, finish the already planned native evaluations. If terminal, inspect decision.json and result note before any next experiment.\n\n"
        "Checked inputs: saved stage verification matches current numerical runtime; source audit 484 checks reused; "
        "18 changed parameter arrays belong only to measurement modules; seed base 1200000000 and original MDPF stage index 1. "
        "Isolated-stage terminal audit passed 124520 checks on completed unit01 before this launch.\n"
        "Earlier evidence: unit01 initialization substitution removed 95.26% validation and 92.68% native endpoint NLL gaps. "
        "Saved pretraining boundaries, composed cloud law and nine-model recent native endpoint passed; no unlocalized translation defect found.\n\n"
        f"Remaining after completed stages: CPU {remaining['cpu']:.6f} /GPU {remaining['gpu']:.6f} seconds; "
        "an active child also consumes elapsed uncharged time. No new allocation assumed.\n\n"
        f"Master: docs/plans/controlled-smc-kdm-replication-master-program-20260928.md\nPlan: {PLAN}\nResult: {RESULT}\n"
        "Previous result: docs/plans/controlled-smc-kdm-recent-endpoint-result-20261013.md\n")
    (ROOT/'docs/plans/controlled-smc-kdm-checkpoint.md').write_text(checkpoint)
    if not terminal:return
    lines=["# Unit02 MDPF initialization result","",f"Updated {stamp}. Status: {status}.", "",detail,"",
           f"Evidence: {link}/manifest.json, decision.json, training and evaluation logs.","",
           "The intervention changed only initial measurement weights. Source data, source stopping, optimizer and TF random schedule stayed fixed.",""]
    if decision and decision.get('status')=='PASS':
        lines+=["| Evaluation NLL gap (TF minus author) | Original TF | Author-start TF |",
                "|---|---:|---:|",
                f"| Native validation, mean of eight draws | {decision['validation_old_gap']:.8f} | {decision['validation_new_gap']:.8f} |",
                f"| Native T150, mean of two seeds | {decision['endpoint_old_gap']:.8f} | {decision['endpoint_new_gap']:.8f} |","",
                f"Initialization explanation supported on both fixed-unit NLL evaluations: {decision['explanation_supported']}.",
                f"Control max loss difference: {decision['control_max_loss_difference']}; training checks: {decision['training_checks']}; "
                f"actual author-controller/coverage audit: {decision['terminal_audit_checks']} checks.",""]
    lines+=["| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Limits |",
            "|---|---|---|---|---|---|",
            f"| {status} diagnostic | Controlled attribution, descriptive NLL contrasts | See decision and audit | One fixed training unit | {decision['next_action'] if decision else 'Inspect logs'} | No whole-KDM completion |","",
            "| Inference item | Finding |","|---|---|",
            f"| Hard veto screen | {'All declared validity checks passed' if status=='COMPLETED' else 'Failed/incomplete; inspect first failed stage'} |",
            "| Statistically supported ranking | None; draws are conditional on fixed trained checkpoints |",
            "| Descriptive differences | Validation and endpoint means, per-draw values, RMSE and histories are preserved |",
            "| Default readiness | No algorithm/default changed or promoted |",
            "| Next evidence needed | Root-cause interpretation of remaining gaps; population studies are separate from replication |","",
            "Post-run red-team: a favorable result can still be specific to this initialization and TF random schedule. "
            "A mismatch in any non-measurement parameter or a failed replay would invalidate attribution. "
            "The weakest evidence is generalization beyond these fixed development units.","",
            f"Charged resources: {manifest['resource_wall_seconds']['gpu']:.6f} GPU seconds and "
            f"{manifest['resource_wall_seconds']['cpu']:.6f} CPU-reference seconds. Prior results remain preserved.",""]
    (ROOT/RESULT).write_text('\n'.join(lines))

def main(preflight=False):
    balance=campaign();assert not balance.get('running')
    prior=sorted(PHASE.glob('attempt*/manifest.json'));assert len(prior)<2
    assert all(read(p)['status']!='RUNNING' for p in prior)
    assert not any(read(p)['status']=='COMPLETED' for p in prior)
    spent={r:sum(read(p)['resource_wall_seconds'][r] for p in prior) for r in CAP}
    remaining={r:min(balance['remaining_seconds'][r],balance['tranche_remaining_seconds'][r]) for r in CAP}
    assert all(CAP[r]>spent[r] and remaining[r]>=CAP[r]-spent[r] for r in CAP)
    input_checks=verify_inputs()
    out=PHASE/('attempt%02d'%(len(prior)+1));assert not out.exists()
    paths=[Path(__file__).resolve(),ROOT/PLAN,stage_path('diagnose_kdm_stage_isolation_tf.py'),
        stage_path('reference_kdm_stage_isolation.py'),stage_path('reference_kdm_recent_endpoint.py'),
        stage_path('reference_kdm_native_evaluation.py'),stage_path('diagnose_kdm_native_evaluation_tf.py'),
        stage_path('reference_kdm_training_terminal_audit.py'),stage_path('diagnose_kdm_source_epoch_tf.py'),
        stage_path('run_replication_gap_localization.py'),ROOT/'bayesfilter/runtime/gpu_memory_policy.py']
    paths+=sorted((ROOT/'bayesfilter/replication').glob('kdm*.py'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    if preflight:
        print(json.dumps(dict(status='PREFLIGHT_PASS',output=str(out),remaining=remaining,spent=spent,
            sources=len(paths),input_checks=input_checks)),flush=True);return 0
    out.mkdir(parents=True,exist_ok=False)
    for p in paths:
        dest=out/'sources'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    gpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS='1',TF_NUM_INTEROP_THREADS='1',
        TF_NUM_INTRAOP_THREADS='1',BAYESFILTER_PRELOAD_CUSTOM_OP='0')
    gpu_env.pop('LD_LIBRARY_PATH',None)
    cpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',LD_LIBRARY_PATH='/home/chakwong/anaconda3/envs/tf-gpu2/lib',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
    manifest=dict(schema='kdm_unit02_initialization_v1',status='RUNNING',command=[sys.executable,*sys.argv],
        cwd=str(ROOT),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        plan_file=PLAN,result_file=RESULT,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        supervisor_pid=os.getpid(),source_pin=PIN,source_hashes=hashes,input_checks=input_checks,
        phase_cap_seconds=CAP,prior_phase_seconds=spent,campaign_remaining_before=remaining,
        resource_wall_seconds={r:0. for r in CAP},steps=[],interpreters={'gpu':TF,'cpu':SOURCE},
        gpu_trust_basis='escalated_local_gpu_access',cpu_gpu_status='TF GPU/XLA FP32 TF32 off; author CPU intentionally hidden',
        environment_gpu={k:gpu_env[k] for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','OMP_NUM_THREADS')},
        environment_cpu={k:cpu_env[k] for k in ('CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','OMP_NUM_THREADS')},
        seed_policy={'shuffle_base':1200000000,'global_stage_index':1,'endpoint_filter_seeds':[202610082,202610083]},
        data_version='fixed unit02 source datasets and native T150 evaluation data, hashed in source artifacts',
        replication='INCOMPLETE')
    def save():write(out/'manifest.json',manifest)
    save()
    refresh_documents(out,manifest)
    def run(label,command,resource,limit=None):
        allowance=min(CAP[resource]-spent[resource]-manifest['resource_wall_seconds'][resource],
                      remaining[resource]-manifest['resource_wall_seconds'][resource])
        if limit is not None:allowance=min(allowance,limit)
        if allowance<=30:raise RuntimeError('Resource cap exhausted before '+label)
        step=dict(label=label,command=command,resource=resource,status='RUNNING',timeout_seconds=allowance,
            log=label+'.log',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        manifest['steps'].append(step);manifest['current_step']=label;save();tick=time.monotonic();child=None
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
                        worker=out/label/'diagnostic/progress.json'
                        if not worker.exists():worker=out/label/'progress.json'
                        step['elapsed_seconds']=time.monotonic()-tick
                        if worker.exists():step['worker_progress']=read(worker)
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
    decision=dict(status='FAIL',replication='INCOMPLETE',next_action='Inspect the first failed stage')
    try:
        nvidia=out/'nvidia';nvidia.mkdir()
        run('nvidia',['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.total','--format=csv'],'gpu',60)
        control=out/'control/diagnostic'
        run('control',[TF,'-I','-B',str(stage_path('diagnose_kdm_stage_isolation_tf.py')),
            '--unit','2','--mode','control','--verification',str(VERIFICATION),'--output',str(control)],'gpu',900)
        c=read(control/'result.json')
        assert c['status']=='PASS' and c['checks_failed']==0 and c['seed_base']==1200000000
        assert c['control_max_batch_loss_difference']<=1e-5
        training=out/'training/diagnostic'
        run('training',[TF,'-I','-B',str(stage_path('diagnose_kdm_stage_isolation_tf.py')),
            '--unit','2','--mode','train','--verification',str(VERIFICATION),
            '--author-start',str(AUTHOR_START),'--output',str(training)],'gpu',20500)
        t=read(training/'result.json')
        assert t['status']=='PASS' and t['checks_failed']==0 and t['selected_stages_complete']
        assert t['unit']==2 and t['seed_base']==1200000000 and t['selected_stages']==['mdpf']
        assert len(t['completed_stages'])==1 and t['completed_stages'][0]['full_protocol']
        manifest['training_best_checkpoint_sha256']=t['completed_stages'][0]['best_checkpoint_sha256'];save()
        native=out/'validation/diagnostic'
        run('validation',[SOURCE,'-I','-B',str(stage_path('reference_kdm_stage_isolation.py')),
            '--unit','2','--mode','evaluate','--trained',str(training),'--output',str(native)],'cpu',1800)
        v=read(native/'result.json')
        assert v['status']=='PASS' and len(v['statistics'])==3 and all(x['n']==8 for x in v['statistics'])
        terminal=out/'terminal-audit.json'
        # Reuse the actual source-controller replay. An isolated start is explicit.
        # The audit expects a small manifest linking the reference and frozen sources.
        run_dir=out/'training'
        reference=BASE/'kdm-source-variability-20261008-01/unit02-full/diagnostic/result.json'
        write(run_dir/'manifest.json',dict(mode='tf',epoch_mode='full',status='COMPLETED',returncode=0,
            reference=str(reference),reference_sha256=sha(reference),hashes=hashes,
            selected_stages=['mdpf'],starting_state_sha256=sha(AUTHOR_START),
            result_sha256=sha(training/'result.json')))
        run('terminal-audit',[SOURCE,'-I','-B',str(stage_path('reference_kdm_training_terminal_audit.py')),
            '--run',str(run_dir),'--isolated-start',str(AUTHOR_START),'--output',str(terminal)],'cpu',900)
        a=read(terminal);assert a['verdict']=='PASS' and not a['failures']
        endpoint=out/'endpoint';endpoint.mkdir()
        run('endpoint-prepare',[SOURCE,'-I','-B',str(stage_path('reference_kdm_recent_endpoint.py')),
            '--unit','2','--intervention-training',str(training),'--output',str(endpoint)],'cpu',600)
        inputs=read(endpoint/'inputs.json')
        assert len(inputs['arms'])==3 and all(x['variant']=='mdpf' for x in inputs['arms'])
        manifest['endpoint_inputs_sha256']=sha(endpoint/'inputs.json');save()
        run('endpoint-native',[TF,'-I','-B',str(stage_path('diagnose_kdm_native_evaluation_tf.py')),
            '--output',str(endpoint)],'gpu',1200)
        e=read(endpoint/'native-result.json')
        assert e['status']=='PASS' and len(e['records'])==6 and len(e['metric_replays'])==3
        assert e['memory_policy']['all_physical_devices_memory_growth'] and e['jit_compile'] and not e['tf32']
        run('endpoint-metrics',[SOURCE,'-I','-B',str(stage_path('reference_kdm_native_evaluation.py')),
            '--mode','metrics','--output',str(endpoint)],'cpu',600)
        m=read(endpoint/'metric-audit.json')
        assert m['status']=='PASS' and not m['failures'] and len(m['records'])==6
        baseline=read(BASE/'recent-endpoint-20261013/attempt01/unit02/native-result.json')
        baseline_rows={(row['checkpoint'],row['seed']):row for row in baseline['records'] if row['variant']=='mdpf'}
        for row in e['records']:
            key=(row['checkpoint'],row['seed'])
            if key in baseline_rows:
                saved=baseline_rows[key]
                assert math.isclose(row['nll'],saved['nll'],rel_tol=0.,abs_tol=1e-5)
                assert math.isclose(row['rmse'],saved['rmse'],rel_tol=0.,abs_tol=1e-5)
        for name,digest in hashes.items():assert sha(ROOT/name)==digest,'Source drift '+name
        scores={x['candidate']:x['mean'] for x in v['statistics']}
        nll={name:sum(x['nll'] for x in e['records'] if x['checkpoint']==name)/2.
             for name in ('source_trained','tf_trained','author_start_tf_trained')}
        rmse={name:sum(x['rmse'] for x in e['records'] if x['checkpoint']==name)/2.
              for name in nll}
        old_gap=scores['old_tf_best']-scores['author_best']
        new_gap=scores['author_start_tf_best']-scores['author_best']
        old_end=nll['tf_trained']-nll['source_trained']
        new_end=nll['author_start_tf_trained']-nll['source_trained']
        decision=dict(status='PASS',replication='INCOMPLETE',scope='UNIT02_INITIALIZATION_INTERVENTION',
            inference='DESCRIPTIVE_UNIT_CONDITIONAL',changed_initial_parameters=input_checks['changed_parameter_count'],
            control_checks=c['checks_total'],control_max_loss_difference=c['control_max_batch_loss_difference'],
            training_checks=t['checks_total'],training_stages=t['completed_stages'],
            terminal_audit_checks=a['checks'],validation=v['statistics'],validation_old_gap=old_gap,
            validation_new_gap=new_gap,native_endpoint_nll=nll,native_endpoint_rmse=rmse,
            endpoint_old_gap=old_end,endpoint_new_gap=new_end,
            explanation_supported=(old_gap>0 and new_gap<old_gap and old_end>0 and new_end<old_end),
            next_action='Interpret unit02 controlled intervention; localize remaining valid gap or close source-conformance scope')
        manifest['status']='COMPLETED'
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
                stage='KDM_UNIT02_INITIALIZATION_DIAGNOSTIC',next_action=decision['next_action'],
                after_attempt=balance['attempts'],executions=[dict(path=str(out.relative_to(ROOT)),
                    resource=resource,wall_seconds=manifest['resource_wall_seconds'][resource],status=manifest['status'])]))
        write(ledger,entries)
        refresh_documents(out,manifest,decision)
    print(json.dumps(dict(status=manifest['status'],decision=decision['status'],output=str(out),
        resource_wall_seconds=manifest['resource_wall_seconds'])),flush=True)
    return 0 if manifest['status']=='COMPLETED' else 1
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    raise SystemExit(main(parser.parse_args().preflight))
