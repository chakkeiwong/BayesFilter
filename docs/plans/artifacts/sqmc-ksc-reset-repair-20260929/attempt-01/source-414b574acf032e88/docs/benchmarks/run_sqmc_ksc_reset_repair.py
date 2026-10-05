#!/usr/bin/env python3
"""Bounded diagnostic runner; no admission, default, or HMC authority.

TF/TFP kernels only. CPU-only preparation calibrates the protection without
looking at KSC accuracy. GPU worker lifetime (including compilation) is charged.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CAMPAIGN = ROOT/'docs/plans/artifacts/sqmc-ksc-reset-repair-20260929'
ATTEMPT = CAMPAIGN/'attempt-01'
PLAN = 'docs/plans/sqmc-ksc-reset-repair-plan-20260929.md'
RESULT = 'docs/benchmarks/sqmc-ksc-reset-repair-results-20260929.md'
PRIOR = ROOT/'docs/plans/artifacts/sqmc-ksc-discrepancy-20260929/budget.json'
ROUTES = ('iid_dual_cap','previous_inverse_cdf','repaired_permutation','repaired_permutation_ablation')
BASE = dict(flow_substeps=8, reset_epsilon=102.4, reset_sinkhorn_steps=24,
            reset_balance_steps=12, correction_steps=1, correction_strength=.12,
            pairwise_steps=1, pairwise_strength=.03)
ARMS = ('baseline','design','protection','combined','steps4','steps8','epsilon25')

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n');tmp.replace(path)
def sources():
    paths=list((ROOT/'bayesfilter/highdim').glob('*.py'))+[Path(__file__), ROOT/PLAN,
          ROOT/'docs/benchmarks/ksc_reset_repair_diagnostic.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def latest(unit):
    paths=sorted(ATTEMPT.glob(unit+'-attempt-*/result.json'))
    good=[p for p in paths if read(p).get('status')=='finished']
    if not good: raise RuntimeError('missing completed unit '+unit)
    return read(good[-1])
def configuration(arm, radius):
    controls=dict(BASE)
    if arm in ('protection','combined','steps4','steps8','epsilon25','reversed'):
        controls['coordinate_cap_identity_radius']=radius
    if arm=='steps4': controls['correction_steps']=4
    if arm in ('steps8','epsilon25'): controls['correction_steps']=8
    if arm=='epsilon25': controls['reset_epsilon']=25.6
    design='repeated_axes' if arm in ('baseline','protection') else 'normal_quantiles'
    if arm=='reversed': design='normal_quantiles_reversed'
    return controls,design

def prepare():
    if CAMPAIGN.exists(): raise RuntimeError('refuse existing campaign')
    prior=read(PRIOR)
    if dt.datetime.now(dt.timezone.utc)>=dt.datetime.fromisoformat(prior['deadline_utc']):
        raise RuntimeError('elapsed budget exhausted')
    ATTEMPT.mkdir(parents=True)
    dump(CAMPAIGN/'budget.json',dict(schema='sqmc_ksc_reset_repair_budget_v1',
        prior_ledger=str(PRIOR.relative_to(ROOT)),prior_sha256=sha(PRIOR),
        prior_charged_seconds=prior['charged_seconds'],cap_seconds=43200.,
        charged_seconds=prior['charged_seconds'],remaining_gpu_seconds=43200.-prior['charged_seconds'],
        deadline_utc=prior['deadline_utc'],attempts=[]))
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    from bayesfilter.highdim.sqmc_campaign_tf import reset_design
    from bayesfilter.highdim.higher_moment_contract_e import identity_core_standardized_cap
    q=reset_design(1008,1,tf.float64,'normal_quantiles')[:,0]
    raw={'normal':q, 'moderate_skew':tf.exp(.35*q),
         'mixture':q+tf.where(q>0.,tf.constant(.7,tf.float64),tf.constant(-.3,tf.float64))}
    fixtures={k:(x-tf.reduce_mean(x))/tf.math.reduce_std(x) for k,x in raw.items()}
    rows=[]
    for radius in (2.,4.,8.):
        for name,x in fixtures.items():
            y,dy=identity_core_standardized_cap(x,radius=radius,tail_fraction=.98)
            rows.append(dict(radius=radius,fixture=name,max_standardized=float(tf.reduce_max(tf.abs(x))),
              max_displacement=float(tf.reduce_max(tf.abs(y-x))),max_derivative_deviation=float(tf.reduce_max(tf.abs(dy-1.))),
              exact_identity=bool(tf.reduce_all(y==x)&tf.reduce_all(dy==1.))))
    acceptable=[r for r in (2.,4.,8.) if all(x['exact_identity'] for x in rows if x['radius']==r)]
    if not acceptable: raise RuntimeError('no non-harming protection radius')
    radius=min(acceptable)
    stress=tf.constant([-1e6,-20.,-8.1,8.1,20.,1e6],tf.float64)
    y,dy=identity_core_standardized_cap(stress,radius=radius,tail_fraction=.98)
    valid=bool(tf.reduce_all(tf.math.is_finite(y)) & tf.reduce_all(tf.abs(y)<=radius*1.98)
               & tf.reduce_all(dy>=0.) & tf.reduce_all(dy<=1.))
    if not valid: raise RuntimeError('protection stress veto')
    dump(ATTEMPT/'protection-calibration.json',dict(cpu_only=True,gpu_intentionally_hidden=True,
      command=[sys.executable,*sys.argv],radius=radius,curve=rows,stress_input=stress.numpy().tolist(),
      stress_output=y.numpy().tolist(),stress_derivative=dy.numpy().tolist(),stress_valid=valid,
      rationale='smallest tested identity region covering declared healthy fixtures; independent of KSC accuracy',
      limitation='bounded before affine covariance restore; no final support bound'))

def worker(unit, out):
    out=Path(out); started=time.monotonic();before=sources()
    manifest=dict(command=[sys.executable,*sys.argv],started_utc=now(),git_commit=subprocess.check_output(
      ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=before,plan=PLAN,result=RESULT,
      environment=sys.executable,unit=unit,program='diagnostic FP64 GPU/XLA shared analytical Contract-E extension',
      tuning='experimental offline selection only; no canonical admission artifact',
      status='running',n=1008,chunk=1008)
    result=dict(status='running',unit=unit,rows=[],fd_checks=[])
    def save(): dump(out/'result.json',result)
    dump(out/'manifest.json',manifest);save()
    try:
        from run_sqmc_ksc_discrepancy import configure
        tf,hardware=configure();manifest.update(hardware);dump(out/'manifest.json',manifest)
        from bayesfilter.highdim.sqmc_ksc_tf import KSCSpec
        from bayesfilter.highdim.sqmc_campaign_tf import value_and_score,random_inputs
        spec=KSCSpec();theta=spec.default_theta()
        radius=read(ATTEMPT/'protection-calibration.json')['radius']
        if unit=='reference':
            from bayesfilter.highdim.sqmc_ksc_gaussian_sum_reference_tf import gaussian_sum_reference
            from bayesfilter.highdim.sqmc_ksc_tf import checked_reference,enumeration_reference,gaussian_approximation_reference
            cases=[]
            for seed in (241001,241002,242001,243001,243002):
                obs=spec.simulate(theta,120,seed)
                cases.append(dict(data_seed=seed,observations=obs.numpy().tolist()))
                for horizon in (2,10,120):
                    data=obs[:horizon];levels=[]
                    for nodes,bound in ((801,40.),(1201,40.),(1201,48.)):
                        ref=gaussian_sum_reference(nodes,bound,True)(theta,data)
                        levels.append(dict(nodes=nodes,bound=bound,value=float(ref['value']),score=ref['score'].numpy().tolist()))
                    grid=checked_reference((1.5,0.),tuple(x[0] for x in data.numpy().tolist()))
                    final=levels[-1]
                    err=max(abs(r['value']-final['value']) for r in levels+[grid])
                    err=max(err,*(abs(a-b) for r in levels+[grid] for a,b in zip(r['score'],final['score'])))
                    if not math.isfinite(err) or err>1e-7: raise RuntimeError('reference continuation veto')
                    with tf.device('/CPU:0'):
                        gv,gs=gaussian_approximation_reference(theta,data)
                        fv,fs=enumeration_reference(1)(theta,data[:1])
                    result['rows'].append(dict(data_seed=seed,horizon=horizon,**final,levels=levels,
                      independent_grid=grid,max_agreement_error=err,
                      heuristics=dict(gaussian=dict(value=float(gv),score=gs.numpy().tolist()),
                        first_only=dict(value=float(fv),score=fs.numpy().tolist()),zero=dict(value=0.,score=[0.,0.]))))
                    save()
            dump(ATTEMPT/'data.json',cases)
        else:
            refs={(r['data_seed'],r['horizon']):r for r in latest('reference')['rows']}
            cases={r['data_seed']:r for r in read(ATTEMPT/'data.json')}
            manifest['data_sha256']=sha(ATTEMPT/'data.json');dump(out/'manifest.json',manifest)
            parts=unit.split('__');phase=parts[0];route=parts[1]
            def evaluate(arm,horizon,data_seed,design_seed):
                controls,design=configuration(arm,radius)
                obs=tf.constant(cases[data_seed]['observations'][:horizon],tf.float64)
                inputs=random_inputs(route,design_seed,1008,1,horizon,tf.float64)
                start=time.monotonic()
                value,score,valid=value_and_score(spec,route,controls,theta,obs,design_seed,1008,
                  inputs=inputs,reset_design_kind=design)
                v=float(value);s=score.numpy().tolist();ref=refs[data_seed,horizon]
                ok=bool(valid) and math.isfinite(v) and all(math.isfinite(x) for x in s)
                row=dict(route=route,arm=arm,horizon=horizon,data_seed=data_seed,design_seed=design_seed,
                  n=1008,controls=controls,reset_design=design,valid=ok,
                  failure_class=None if ok else 'candidate_numerical_invalidity',wall_seconds=time.monotonic()-start,
                  input_sha256=[hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest() for x in inputs],
                  value=v if math.isfinite(v) else None,score=[x if math.isfinite(x) else None for x in s],
                  reference_value=ref['value'],reference_score=ref['score'])
                if ok:
                    errors=[a-b for a,b in zip(s,ref['score'])]
                    row.update(value_error=v-ref['value'],absolute_value_error=abs(v-ref['value']),
                      score_error=errors,absolute_score_error=list(map(abs,errors)),score_l2_error=math.sqrt(sum(x*x for x in errors)))
                result['rows'].append(row);save()
                print(json.dumps({k:row[k] for k in ('arm','horizon','data_seed','design_seed','valid','wall_seconds')}),flush=True)
                return row,inputs,obs
            if phase=='checks':
                from ksc_reset_repair_diagnostic import trace_kernel
                from run_sqmc_ksc_discrepancy import fd_classify
                for horizon,arm in ((2,'combined'),(10,'epsilon25')):
                    row,inputs,obs=evaluate(arm,horizon,241001,251001)
                    if not row['valid']: raise RuntimeError('GPU candidate validity veto')
                    controls,design=configuration(arm,radius)
                    fn=trace_kernel(spec,route,controls,1008,horizon,design,stages=False)
                    value,score,anc=fn(theta,tf.constant([1.,0.],tf.float64),*inputs,obs)
                    if abs(float(value)-row['value'])>1e-8 or abs(float(score)-row['score'][0])>1e-7:
                        raise RuntimeError('call-chain trace parity veto')
                    for coordinate in range(2):
                        ladder=[]
                        for step in (1e-4,1e-5,1e-6,1e-7,1e-8):
                            delta=tf.one_hot(coordinate,2,dtype=tf.float64)*step
                            vp,_,ap=fn(theta+delta,tf.one_hot(coordinate,2,dtype=tf.float64),*inputs,obs)
                            vm,_,am=fn(theta-delta,tf.one_hot(coordinate,2,dtype=tf.float64),*inputs,obs)
                            changes=bool(tf.reduce_any(ap!=anc)|tf.reduce_any(am!=anc))
                            ladder.append(dict(step=step,centered=(float(vp)-float(vm))/(2*step),branch_changes=changes))
                            verdict=fd_classify(row['score'][coordinate],ladder)
                            if step<=1e-6 and verdict['status']=='pass': break
                        result['fd_checks'].append(dict(arm=arm,horizon=horizon,coordinate=coordinate,
                          analytical=row['score'][coordinate],ladder=ladder,**verdict));save()
                        if verdict['status']!='pass': raise RuntimeError('finite-program derivative veto')
            elif phase=='stage':
                from ksc_reset_repair_diagnostic import trace_kernel
                for arm in ('baseline','design','protection','combined','reversed'):
                    row,inputs,obs=evaluate(arm,10,241001,251001)
                    controls,design=configuration(arm,radius)
                    fn=trace_kernel(spec,route,controls,1008,10,design,stages=True)
                    for coordinate in range(2):
                        raw=fn(theta,tf.one_hot(coordinate,2,dtype=tf.float64),*inputs,obs)
                        record={k:v.numpy().tolist() for k,v in raw.items()}
                        if abs(record['value']-row['value'])>1e-8 or abs(record['score']-row['score'][coordinate])>1e-7:
                            raise RuntimeError('stage trace parity veto')
                        dump(out/f'{arm}-coordinate-{coordinate}.json',record)
            else:
                horizon=int(parts[2])
                if phase=='calibration':
                    for arm in ARMS:
                        for data_seed in (241001,241002):
                            for design_seed in (251001,251002): evaluate(arm,horizon,data_seed,design_seed)
                    chosen=select_arm(result['rows'])
                    result['selection']=dict(arm=chosen,selection_rule='minimum calibration score L2 subject to validity and value non-harm',
                      frozen_utc=now(),data_seeds=[241001,241002],design_seeds=[251001,251002]);save()
                else:
                    chosen=latest(f'calibration__{route}__{horizon}')['selection']['arm']
                    datasets=(242001,) if phase=='validation' else (243001,243002)
                    seeds=(252001,252002) if phase=='validation' else tuple(range(253001,253005))
                    if phase=='extension': seeds=tuple(range(253005,253009))
                    for arm in dict.fromkeys(('baseline',chosen)):
                        for data_seed in datasets:
                            for design_seed in seeds: evaluate(arm,horizon,data_seed,design_seed)
                    result['frozen_selection']=chosen
        if before!=sources(): raise RuntimeError('source changed while numerical worker was active')
        result['status']='finished';save()
    except Exception as exc:
        result.update(status='failed',failure_type=type(exc).__name__,failure=str(exc));save();raise
    finally:
        manifest.update(finished_utc=now(),wall_seconds=time.monotonic()-started,status=result['status'])
        if 'tf' in locals():
            try: manifest['allocator']=tf.config.experimental.get_memory_info('GPU:0')
            except Exception as exc: manifest['allocator_error']=str(exc)
        dump(out/'manifest.json',manifest)

def select_arm(rows):
    groups={arm:[r for r in rows if r['arm']==arm] for arm in ARMS}
    baseline=groups['baseline']
    if not baseline or not all(r['valid'] for r in baseline): raise RuntimeError('invalid calibration baseline')
    b=sum(r['absolute_value_error'] for r in baseline)/len(baseline)
    eligible=[]
    for arm,records in groups.items():
        if len(records)!=len(baseline) or not all(r['valid'] for r in records): continue
        v=sum(r['absolute_value_error'] for r in records)/len(records)
        if v<=b+max(.1*b,.01):
            eligible.append((sum(r['score_l2_error'] for r in records)/len(records),arm))
    return min(eligible)[1]

def supervise(phase):
    ledger=read(CAMPAIGN/'budget.json')
    if any(x['status']=='running' for x in ledger['attempts']): raise RuntimeError('unreconciled running attempt')
    if phase=='reference': units=['reference']
    elif phase=='stage': units=['stage__previous_inverse_cdf']
    elif phase=='checks': units=['checks__'+r for r in ROUTES]
    else: units=[f'{phase}__{r}__{h}' for r in ROUTES for h in (10,120)]
    if phase not in ('reference','checks'):
        for route in ROUTES: latest('checks__'+route)
    for unit in units:
        attempts=[a for a in ledger['attempts'] if a['unit']==unit]
        if any(a['status']=='finished' for a in attempts): continue
        if len(attempts)>=3: raise RuntimeError('two infrastructure retries exhausted')
        available=min(ledger['remaining_gpu_seconds'],(dt.datetime.fromisoformat(ledger['deadline_utc'])-dt.datetime.now(dt.timezone.utc)).total_seconds())-30.
        limits={'reference':600.,'checks':700.,'stage':900.,'calibration':1000.,'validation':600.,'evaluation':900.,'extension':900.}
        timeout=min(available,limits[phase])
        if timeout<60: raise RuntimeError('compute or elapsed budget exhausted')
        out=ATTEMPT/f'{unit}-attempt-{len(attempts)+1:02d}';out.mkdir()
        hashes=sources()
        source_id=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
        snapshot=ATTEMPT/('source-'+source_id)
        if not snapshot.exists():
            snapshot.mkdir()
            for path in hashes:
                dest=snapshot/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,dest)
        dump(out/'source-sha256.json',dict(snapshot=str(snapshot.relative_to(ROOT)),files=hashes))
        command=[sys.executable,str(Path(__file__).resolve()),'--mode','worker','--unit',unit,'--output',str(out)]
        entry=dict(unit=unit,output=str(out.relative_to(ROOT)),command=command,started_utc=now(),status='running',timeout_seconds=timeout)
        ledger['attempts'].append(entry);dump(CAMPAIGN/'budget.json',ledger)
        started=time.monotonic()
        with (out/'worker.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            entry['pid']=proc.pid;dump(CAMPAIGN/'budget.json',ledger)
            try: code=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code=124
        elapsed=time.monotonic()-started
        entry.update(finished_utc=now(),wall_seconds=elapsed,returncode=code,status='finished' if code==0 else 'failed')
        ledger['charged_seconds']+=elapsed;ledger['remaining_gpu_seconds']=ledger['cap_seconds']-ledger['charged_seconds']
        dump(CAMPAIGN/'budget.json',ledger)
        print(json.dumps(entry|dict(remaining_gpu_seconds=ledger['remaining_gpu_seconds'])),flush=True)
        if code: raise RuntimeError('worker failed; inspect preserved evidence before bounded retry')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prepare','supervise','worker'],required=True)
    parser.add_argument('--phase',choices=['reference','checks','stage','calibration','validation','evaluation','extension'])
    parser.add_argument('--unit');parser.add_argument('--output');args=parser.parse_args()
    if args.mode=='prepare': prepare()
    elif args.mode=='supervise': supervise(args.phase)
    else: worker(args.unit,args.output)
