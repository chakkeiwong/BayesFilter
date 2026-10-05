"""Bounded KSC mixture comparison; shared analytical particle executor."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
PLAN='docs/plans/sqmc-ksc-sv-comparison-20260928.md'
CAMPAIGN=ROOT/'docs/plans/artifacts/sqmc-ksc-sv-20260928'
DEADLINE=datetime.fromisoformat('2026-09-30T16:15:40.010888+00:00').timestamp()
PRIOR=21377.32999273797
COMMIT_HOOK_RESERVE=300.  # Conservative whole wall allowance; GPU use not measured.
CAL=(211001,211002)
VAL=(212001,)
FINAL=tuple(range(213001,213009))
FILTER=tuple(range(214001,214009))
ROUTES=('iid_dual_cap','previous_inverse_cdf','repaired_permutation','repaired_permutation_ablation')


def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def sources():
    paths=list((ROOT/'bayesfilter').rglob('*.py'))+list((ROOT/'experiments/dpf_implementation/tf_tfp').rglob('*.py'))+[Path(__file__),ROOT/PLAN,
        ROOT/'docs/benchmarks/run_sqmc_expanded_comparison.py',
        ROOT/'docs/benchmarks/run_sqmc_expanded_repair.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def require_elapsed_budget():
    if time.time() >= DEADLINE:
        raise RuntimeError('Authorized elapsed window expired; an owner renewal is required before numerical execution')


def configure():
    require_elapsed_budget()
    import run_sqmc_expanded_comparison as base
    tf,meta=base.configure_gpu()
    meta['cpu_gpu_status']='GPU analytical particle kernel; CPU converged mixture quadrature reference'
    return tf,meta


def check(out):
    out.mkdir(parents=True,exist_ok=False)
    tf,meta=configure()
    from bayesfilter.highdim.sqmc_ksc_tf import KSCSpec
    from bayesfilter.highdim.sqmc_campaign_tf import value_and_score,random_inputs
    import run_sqmc_expanded_comparison as base
    spec=KSCSpec();theta=spec.default_theta()
    obs=tf.constant([[-1.],[-.5]],tf.float64)
    controls=dict(base.CONTROLS[0],reset_epsilon=25.6)
    before=sources()
    result=dict(gpu=meta,source_sha256=before,program='FP64 KSC diagnostic; UNTUNED',rows=[],scale_rows=[])
    for route in ROUTES:
        inputs=random_inputs(route,215001,16,1,2,tf.float64)
        graph=value_and_score(spec,route,controls,theta,obs,215001,16,jit_compile=False,inputs=inputs)
        compiled=value_and_score(spec,route,controls,theta,obs,215001,16,jit_compile=True,inputs=inputs)
        if not bool(graph[2]) or not bool(compiled[2]):
            raise RuntimeError('tiny KSC parity fixture is numerically invalid')
        delta=max(abs(float(graph[0]-compiled[0])),float(tf.reduce_max(tf.abs(graph[1]-compiled[1]))))
        fd=[]
        for i in range(2):
            step=tf.one_hot(i,2,dtype=tf.float64)*1e-5
            vp=value_and_score(spec,route,controls,theta+step,obs,215001,16,jit_compile=True,inputs=inputs)
            vm=value_and_score(spec,route,controls,theta-step,obs,215001,16,jit_compile=True,inputs=inputs)
            if not bool(vp[2]) or not bool(vm[2]):raise RuntimeError('FD perturbation invalid')
            fd.append(float((vp[0]-vm[0])/2e-5))
        fd_error=max(abs(a-b) for a,b in zip(fd,compiled[1].numpy().tolist()))
        row=dict(route=route,value=float(compiled[0]),score=compiled[1].numpy().tolist(),
                 graph_xla_max_error=delta,finite_difference=fd,finite_difference_max_error=fd_error)
        result['rows'].append(row);dump(out/'checks.json',result);print(json.dumps(row),flush=True)
        if delta>1e-7 or fd_error>2e-4:raise RuntimeError('KSC graph/XLA or finite-program derivative check failed')
    for route in ROUTES:
        value,score,valid=value_and_score(spec,route,controls,theta,obs,215002,1008,jit_compile=True)
        row=dict(route=route,particle_count=1008,valid=bool(valid),
                 value=float(value) if bool(valid) else None,score=score.numpy().tolist() if bool(valid) else None)
        result['scale_rows'].append(row);dump(out/'checks.json',result);print(json.dumps(row),flush=True)
        if not bool(valid):raise RuntimeError('KSC N=1008 bounded fixture is numerically invalid; inspect before campaign')
    if before!=sources():raise RuntimeError('source changed during bounded checks')
    result['status']='pass';dump(out/'checks.json',result)


def worker(out,horizon,route):
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    before=sources()
    meta=dict(command=[sys.executable,*sys.argv],plan=PLAN,git_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=before,
        started_utc=datetime.now(timezone.utc).isoformat(),seed_partitions=dict(calibration=CAL,validation=VAL,final=FINAL,filter=FILTER))
    result=dict(scope=f'ksc_T{horizon}',route=route,horizon=horizon,particle_count=1008,
                program='fp64_gpu_xla_ksc_seven_mixture',production_deviations='FP64/TF32 off diagnostic; no production/default/HMC claim',
                tuning_status='pending',status='running',parameter_names=['gamma_raw','log_beta'],rows=[])
    dump(out/'manifest.json',meta);dump(out/'result.json',result)
    tf=None
    ReferenceConvergenceError=()  # Replaced after the optional TensorFlow import.
    try:
        tf,gpu=configure();meta.update(gpu)
        from bayesfilter.highdim.sqmc_ksc_tf import KSCSpec,checked_reference,gaussian_approximation_reference,enumeration_reference,ReferenceConvergenceError
        from bayesfilter.highdim.sqmc_campaign_tf import random_inputs,evaluate_diagnostic
        from bayesfilter.highdim.sqmc_campaign_tuning import tune_campaign,evaluate_untouched
        import run_sqmc_expanded_comparison as base
        import run_sqmc_expanded_repair as repair
        base.CAL=CAL
        spec=KSCSpec();theta=spec.default_theta()
        data={}
        for seed in (*CAL,*VAL,*FINAL):
            observations=spec.simulate(theta,horizon,seed,jit_compile=True)
            values=observations.numpy().tolist()
            reference=checked_reference(tuple(theta.numpy().tolist()),tuple(x[0] for x in values))
            with tf.device('/CPU:0'):
                gaussian=gaussian_approximation_reference(theta,observations)
                first=enumeration_reference(1)(theta,observations[:1])
            data[str(seed)]=dict(observations=values,reference=reference,gaussian_value=float(gaussian[0]),
                                 gaussian_score=gaussian[1].numpy().tolist(),first_score=first[1].numpy().tolist())
            dump(out/'data.json',data)
        designs={}
        for seed in (*CAL,*VAL,*FILTER):
            designs[str(seed)]={name:hashlib.sha256(tf.io.serialize_tensor(t).numpy()).hexdigest()
                for name,t in zip(('initial','noise','uniforms'),random_inputs(route,seed,1008,1,horizon,tf.float64))}
        dump(out/'designs.json',designs)
        meta.update(data_sha256=hashlib.sha256((out/'data.json').read_bytes()).hexdigest(),design_sha256=hashlib.sha256((out/'designs.json').read_bytes()).hexdigest())
        dump(out/'manifest.json',meta)
        def progress(stage,controls,row):
            with (out/'evaluations.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,controls=controls,row=row),allow_nan=False)+'\n')
            print(json.dumps(dict(stage=stage,route=route,horizon=horizon,seed=row['seed'],valid=row['valid'],error=row.get('score_l2_error'))),flush=True)
        controls=repair.prepare_controls(spec,route,theta,horizon,1008,out)
        if controls:
            artifact,tuning=tune_campaign(spec,route,controls,theta,horizon,1008,CAL,VAL,FINAL,jit_compile=True,on_result=progress)
        else:artifact,tuning=None,dict(decision='transport_calibration_failed')
        dump(out/'tuning.json',tuning)
        result['selected_controls']=tuning.get('selected_controls')
        result['tuning_path']=str(out/'tuning.json')
        if artifact is None:
            result.update(status='candidate_tuning_failed',tuning_status='no_validated_controls',tuning_decision=tuning['decision'])
        else:
            result['tuning_status']='exact_scope_frozen_controls'
            # A dedicated safety sensitivity diagnostic uses calibration data
            # only. It cannot select controls by final accuracy or promote a default.
            sensitivity=[]
            selected=result['selected_controls']
            calibration_observations=spec.simulate(theta,horizon,CAL[0],jit_compile=True)
            baseline=evaluate_diagnostic(spec,route,selected,calibration_observations,theta,CAL[0],1008,jit_compile=True)
            if not baseline['valid']:raise RuntimeError('Frozen calibration changed validity on deterministic replay')
            for parameter,factors in (('reset_ridge',(.1,10.)),('correction_lm_damping',(.1,10.)),('correction_trust_radius',(.5,2.))):
                for factor in factors:
                    changed=dict(selected);changed[parameter]*=factor
                    row=evaluate_diagnostic(spec,route,changed,calibration_observations,theta,CAL[0],1008,jit_compile=True)
                    change=max(abs(a-b) for a,b in zip(row['score'],baseline['score'])) if row['valid'] else None
                    sensitivity.append(dict(parameter=parameter,factor=factor,row=row,
                        score_change_max_abs=change,baseline=baseline,
                        disposition='explanatory_only; no control selected; no non-harm or default claim'))
                    dump(out/'safety_sensitivity.json',sensitivity)
                    progress('safety_sensitivity',changed,row)
            for seed,filter_seed in zip(FINAL,FILTER):
                row=evaluate_untouched(spec,route,artifact,theta,horizon,1008,seed,jit_compile=True,filter_seed=filter_seed)
                row.pop('scope',None)
                ref=data[str(seed)]
                oracle=ref['reference']['score']
                if max(abs(a-b) for a,b in zip(row['oracle_score'],oracle))>1e-10:raise RuntimeError('saved reference differs')
                row['heuristic_errors']={name:math.sqrt(sum((a-b)**2 for a,b in zip(v,oracle))) for name,v in
                    [('zero_score',[0.,0.]),('first_only',ref['first_score']),('gaussian_kalman',ref['gaussian_score'])]}
                row['heuristic_dominance']={name:('observed_loss' if row['score_l2_error']>e else 'no_observed_loss')
                    for name,e in row['heuristic_errors'].items()} if row['valid'] else {'status':'invalid_candidate'}
                result['rows'].append(row);dump(out/'result.json',result);progress('final',result['selected_controls'],row)
            result['status']='complete' if all(r['valid'] for r in result['rows']) else 'complete_with_invalid_cells'
        if before!=sources():raise RuntimeError('source changed during worker')
        meta['source_unchanged']=True
    except ReferenceConvergenceError as error:
        result.update(status='reference_convergence_veto',error=str(error),reference_failure=error.record)
        traceback.print_exc();raise
    except BaseException as error:
        result.update(status='infrastructure_or_harness_failure',error=repr(error));traceback.print_exc();raise
    finally:
        meta.update(status=result['status'],wall_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        if tf is not None:
            try:meta['gpu_allocator_bytes']=tf.config.experimental.get_memory_info('GPU:0')
            except (ValueError,RuntimeError) as error:meta['allocator_error']=str(error)
        dump(out/'result.json',result);dump(out/'manifest.json',meta)


def supervise(out,phase):
    out=out.resolve()
    if CAMPAIGN.resolve() not in out.parents:
        raise ValueError('Output must be a versioned child of the KSC campaign directory')
    out.mkdir(parents=True,exist_ok=True)
    # All versioned output roots share one ledger; changing directories cannot
    # restart the aggregate compute or per-unit infrastructure retry budget.
    budget_path=CAMPAIGN/'budget.json'
    ledger=json.loads(budget_path.read_text()) if budget_path.exists() else dict(
        prior_gpu_seconds=PRIOR,commit_hook_conservative_seconds=COMMIT_HOOK_RESERVE,cap_seconds=43200.,
        deadline_utc=datetime.fromtimestamp(DEADLINE,timezone.utc).isoformat(),attempts=[])
    if ledger['deadline_utc']!=datetime.fromtimestamp(DEADLINE,timezone.utc).isoformat():
        raise RuntimeError('Reconcile an owner-authorized deadline renewal in the ledger before launch')
    if any(a.get('status')=='running' for a in ledger['attempts']):
        raise RuntimeError('Reconcile the interrupted attempt and its actual charge before launching')
    def remaining_budget():
        used=ledger['prior_gpu_seconds']+ledger['commit_hook_conservative_seconds']+sum(a['wall_seconds'] for a in ledger['attempts'])
        ledger.update(charged_seconds=used,remaining_gpu_seconds=ledger['cap_seconds']-used)
        return min(ledger['cap_seconds']-used,DEADLINE-time.time())
    if remaining_budget()<=15:
        ledger['status']='budget_or_deadline_exhausted';dump(budget_path,ledger);return
    before=sources()
    signature=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest()
    passed_checks=[]
    for a in ledger['attempts']:
        if a['unit']=='check' and a['returncode']==0:
            check_path=Path(a['output'])/'checks.json'
            record=json.loads(check_path.read_text())
            if record.get('status')=='pass' and record.get('source_sha256')==before:
                passed_checks.append(check_path)
    if phase=='campaign' and not passed_checks:
        raise RuntimeError('Campaign requires passing bounded GPU checks for the current source files')
    snapshot=out/f'source-{signature[:12]}'
    if not snapshot.exists():
        snapshot.mkdir()
        for name in before:
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
        dump(snapshot/'sha256.json',before)
    # This query and all children require escalated GPU permissions.
    inventory=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
    tasks=[('check',None,None)] if phase=='check' else [(f'T{h}_{r}',h,r) for h in (10,20,50,120) for r in ROUTES]
    for name,horizon,route in tasks:
        prior_attempts=[a for a in ledger['attempts'] if a['unit']==name]
        if any(a['returncode']==0 and a.get('source_signature')==signature for a in prior_attempts):continue
        if phase=='campaign' and any(a['returncode']==0 for a in prior_attempts):
            raise RuntimeError('Completed scope has different sources; do not reuse exposed final data for a repair')
        if any(a.get('failure_class')=='reference_convergence_veto' for a in prior_attempts):
            raise RuntimeError('Reference continuation veto requires scientific review before further execution')
        if len(prior_attempts)>=3:raise RuntimeError('Unit infrastructure retry cap exhausted')
        remaining=remaining_budget();dump(budget_path,ledger)
        if remaining<=15:
            ledger['status']='budget_or_deadline_exhausted';dump(budget_path,ledger);return
        if sources()!=before:raise RuntimeError('Source changed during supervisor execution')
        target=out/f'{name}-attempt-{len(prior_attempts)+1:02d}'
        if target.exists():raise RuntimeError('Preserve existing output and select a fresh versioned directory')
        (out/f'{target.name}-nvidia.log').write_text(inventory)
        if 'GPU-68251639-fe82-8f81-3ccc-2953c32e805b' in inventory:
            raise RuntimeError('Chosen GPU has an existing compute process; ownership unresolved')
        cmd=[sys.executable,str(Path(__file__).resolve()),'--output',str(target),'--mode','check-worker' if phase=='check' else 'worker']
        if horizon is not None:cmd+=['--horizon',str(horizon),'--route',route]
        started=time.monotonic()
        entry=dict(unit=name,command=cmd,wall_seconds=0.,returncode=None,output=str(target),
                   source_signature=signature,source_snapshot=str(snapshot),status='running',
                   started_utc=datetime.now(timezone.utc).isoformat())
        ledger['attempts'].append(entry);dump(budget_path,ledger)
        with (out/f'{target.name}.log').open('x') as log:
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            entry['pid']=process.pid;dump(budget_path,ledger)
            # Leave room for termination and accounting inside both limits.
            try:code=process.wait(timeout=min(remaining-12.,900. if phase=='check' else remaining-12.))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10.)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                code=124
        entry.update(wall_seconds=time.monotonic()-started,returncode=code,status='finished')
        result_path=target/('checks.json' if phase=='check' else 'result.json')
        if result_path.exists():entry['result_status']=json.loads(result_path.read_text()).get('status','incomplete')
        if code:entry['failure_class']='reference_convergence_veto' if entry.get('result_status')=='reference_convergence_veto' else 'infrastructure_or_harness_failure'
        remaining_budget();dump(budget_path,ledger);print(json.dumps(entry),flush=True)
        if code:
            ledger['status']='stopped_for_failure_review';dump(budget_path,ledger);return
        # Recheck ownership between units; an unrelated job may have started.
        if remaining_budget()>15:
            inventory=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
    ledger['status']=phase+'_complete';dump(budget_path,ledger)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('check','campaign','check-worker','worker'),required=True)
    parser.add_argument('--horizon',type=int)
    parser.add_argument('--route',choices=ROUTES)
    args=parser.parse_args()
    if args.mode in ('check','campaign'):supervise(args.output,args.mode)
    elif args.mode=='check-worker':check(args.output)
    else:worker(args.output,args.horizon,args.route)


if __name__=='__main__':main()
