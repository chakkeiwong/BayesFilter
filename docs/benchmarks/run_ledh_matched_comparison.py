#!/usr/bin/env python3
"""Diagnostic matched comparison. All filtering calls shared TF authorities.

This runner compares frozen implementations; it cannot promote defaults. CPU is
only used for explicitly hidden-GPU smoke checks. It never installs packages.
"""
from pathlib import Path
import argparse
import csv
import dataclasses
import datetime
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).parent))
import run_ledh_covariance_proposal as candidate
from run_ledh_nonlinear_master import BASE,arm_settings
PLAN=ROOT/'docs/plans/ledh-matched-comparison-20261008.md'
OUT=ROOT/'docs/plans/artifacts/ledh-matched-comparison-20261008-01'
OLD=ROOT/'docs/plans/artifacts/ledh-zhao-horizons-20261006-01'
PROTECTED=ROOT/'docs/plans/artifacts/ledh-sir-no-oracle-tuning-20261006-01/terminal_evidence.json'
MODELS=('lgssm','ksc','predator_prey','sir_d18')
SEEDS=tuple(range(261006101,261006105))
N=1008
T=50
DATA=26100611
ROUTE='iid_dual_cap'
read=lambda p:json.loads(Path(p).read_text())
write=candidate.write


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_sha(tf,x): return hashlib.sha256(bytes(tf.io.serialize_tensor(x).numpy())).hexdigest()
def specs(name): return candidate.spec_for(name,'p44')

def old_controls(name,covariance_only=False):
    controls,design=arm_settings('guarded_pairwise',BASE)
    controls['importance_weight_policy']='ancestor' if name in ('lgssm','ksc') else 'marginal_mixture'
    if covariance_only: controls.update(correction_steps=0,pairwise_steps=0)
    return controls,design


def fixed_inputs(tf,spec,seed,n=N,h=T):
    from bayesfilter.highdim.sqmc_campaign_tf import random_inputs
    shared=random_inputs(ROUTE,seed,n,spec.dimension,h,tf.float64,jit_compile=True)
    # Extra categorical branch/ancestor variates have no old-IID counterpart.
    extra=tf.random.stateless_uniform([h,n,2],[seed,991],dtype=tf.float64)
    return shared,(shared[0],shared[1],extra)


def data_and_reference(tf,spec,name,dataset_file=None,data_seed=DATA):
    theta=spec.default_theta(tf.float64)
    if dataset_file is not None:
        data=read(dataset_file)
        if (data['target_id']!=spec.target_id or data['data_seed']!=data_seed
                or data['dtype']!='float64' or len(data['observations'])!=T):
            raise ValueError('independent dataset scope mismatch')
        obs=tf.constant(data['observations'],tf.float64)
        if tensor_sha(tf,obs)!=data['observation_sha256']:
            raise ValueError('corrupted independent observations')
        if name in ('lgssm','ksc'):
            value,score=spec.reference_value_and_score(theta,obs)
            ref=dict(value=value,score=score,value_se=0.,score_se=[0.]*spec.parameter_count,
                     kind='exact Kalman' if name=='lgssm' else 'independently checked refined KSC grid')
        else:
            ref=dict(value=None,score=None,kind='independent references assembled after evaluation')
    elif name in ('predator_prey','sir_d18'):
        path=OLD/'inputs'/f'{name}-T50'/'dataset.json'
        data=read(path); obs=tf.constant(data['observations'],tf.float64)
        if data['target_id']!=spec.target_id or data['data_seed']!=DATA: raise ValueError('target/data mismatch')
        digest=tensor_sha(tf,obs)
        if digest!=data['observation_sha256']: raise ValueError('corrupted observations')
        summaries=read(OLD/'026-bootstrap-references/summary.json')
        refs=[r for r in summaries if r['model']==name and r['horizon']==T]
        for r in refs:
            if r['observation_sha256']!=digest or r['target_id']!=spec.target_id or r['theta']!=theta.numpy().tolist():
                raise ValueError('reference scope mismatch')
        authority=next(r for r in refs if r['particles']==131072)
        ref=dict(value=authority['log_likelihood'],score=authority['score'],
                 value_se=authority['jackknife_mcse_log_likelihood'],score_se=authority['jackknife_mcse_score'],
                 kind='saved N131072 four-replication bootstrap/Fisher; finite-particle bias not bounded',
                 source=str(OLD/'026-bootstrap-references/summary.json'),source_sha256=sha(OLD/'026-bootstrap-references/summary.json'))
        tab=list(csv.DictReader((OLD/'committed-evidence-03/report-009/values-and-scores.csv').open()))
        zhao=[r for r in tab if r['model']==name and r['horizon']=='50' and r['method'].startswith('zhao_cui')]
        ref.update(bootstrap_ladder=refs,zhao_cui_comparators=zhao)
    else:
        obs=spec.simulate(theta,T,DATA,jit_compile=True)
        data=dict(observations=obs,data_seed=DATA,target_id=spec.target_id,dtype='float64',
                  observation_sha256=tensor_sha(tf,obs),generator='original protected replay simulator GPU/XLA')
        value,score=spec.reference_value_and_score(theta,obs)
        ref=dict(value=value,score=score,value_se=0.,score_se=[0.]*spec.parameter_count,
                 kind='exact Kalman' if name=='lgssm' else 'independently checked refined KSC grid')
    return theta,obs,data,candidate.clean(ref)


def replay_check(name,seed,row):
    expected=None
    if seed==261006201:
        rows=read(PROTECTED)['protected_regressions']['rows']
        expected=next((r for r in rows if r['horizon']==T and r['model']==('ksc_sv' if name=='ksc' else name)),None)
        source=str(PROTECTED)
    elif name in ('predator_prey','sir_d18'):
        folder='011-ledh-predator_prey-T50' if name=='predator_prey' else '024-ledh-sir_d18-T50'
        source=str(OLD/folder/'rows.json')
        rows=read(source)
        expected=next((r for r in rows if r.get('design_seed')==seed and r.get('importance_weight_policy')=='marginal_mixture'),None)
    if expected is None:return None
    value=expected.get('value',expected.get('log_likelihood'))
    vg=abs(row['value']-value)
    sg=max(abs(a-b) for a,b in zip(row['score'],expected['score']))
    valid=vg<=1e-8*(1+abs(value)) and sg<=1e-6*(1+max(map(abs,expected['score'])))
    return dict(source=source,source_sha256=sha(source),value_gap=vg,max_score_gap=sg,passed=valid)


def worker(args):
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();tf,memory=candidate.prepare(args)
    spec=specs(args.model)
    dataset_file=getattr(args,'dataset_file',None)
    data_seed=getattr(args,'data_seed',DATA)
    independent=dataset_file is not None
    if independent != bool(getattr(args,'independent',False)):
        raise ValueError('independent mode requires an explicit dataset file')
    theta,obs,data,reference=data_and_reference(tf,spec,args.model,dataset_file,data_seed)
    from bayesfilter.highdim.sqmc_campaign_tf import value_and_score
    from bayesfilter.highdim.covariance_proposal_tf import make_filter,TRACE_FIELDS
    ctl=candidate.controls(args)
    tuning=read(args.tuning)
    if not tuning['valid'] or tuning['scope']!=candidate.scope(args,spec): raise ValueError('invalid tuning scope')
    if data_seed in (tuning['calibration_data_seed'],tuning['validation_data_seed']): raise ValueError('tuning data leakage')
    beta=tf.constant(tuning['beta'],tf.float64)
    program=make_filter(spec,N,T,ctl,tf.float64)
    metadata=dict(model=args.model,target_id=spec.target_id,theta=theta,particles=N,horizon=T,
       data=data,reference=reference,tuning=tuning,controls_old=old_controls(args.model),controls_new=ctl,
       git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       command=sys.argv,environment=sys.executable,device=args.device,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
       memory_policy=memory,tf_version=tf.__version__,dtype='float64',jit_compile=True,tf32=False,
       plan=getattr(args,'plan_file',str(PLAN)),result_file=getattr(args,'result_file','docs/benchmarks/ledh-matched-comparison-results-20261008.md'),
       tuning_source=str(args.tuning),tuning_sha256=sha(args.tuning),independent_dataset=independent,
       source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_paths()})
    write(folder/'manifest.json',metadata)
    rows=[]
    def keep(row):
        rows.append(candidate.clean(row));write(folder/'rows.json',rows)
        print(json.dumps({k:rows[-1][k] for k in ('model','method','seed','value','score','valid','wall_seconds')}),flush=True)
    if args.model!='sir_d18' and not independent:
        tick=time.monotonic();seed=261006201;old_input,_=fixed_inputs(tf,spec,seed)
        oc,design=old_controls(args.model)
        val,score,valid=value_and_score(spec,ROUTE,oc,theta,obs,seed,N,inputs=old_input,reset_design_kind=design)
        row=candidate.clean(dict(model=args.model,method='old_replay',seed=seed,value=val,score=score,valid=valid,wall_seconds=time.monotonic()-tick))
        row['replay']=replay_check(args.model,seed,row);keep(row)
        if not row['valid'] or (row['replay'] is not None and not row['replay']['passed']):raise ValueError('old protected replay failed')
    for seed in getattr(args,'design_seeds',SEEDS):
        old_input,new_input=fixed_inputs(tf,spec,seed)
        if old_input[0] is not new_input[0] or old_input[1] is not new_input[1]: raise AssertionError('unpaired inputs')
        hashes=dict(initial=tensor_sha(tf,old_input[0]),process=tensor_sha(tf,old_input[1]),
                    old_uniforms=tensor_sha(tf,old_input[2]),candidate_categorical=tensor_sha(tf,new_input[2]))
        for method in ('old','old_covariance_only','new'):
            tick=time.monotonic()
            if method!='new':
                oc,design=old_controls(args.model,method=='old_covariance_only')
                diagnostics={}
                val,score,valid=value_and_score(spec,ROUTE,oc,theta,obs,seed,N,inputs=old_input,
                                                reset_design_kind=design,diagnostics=diagnostics)
                row=candidate.clean(dict(model=args.model,method=method,seed=seed,value=val,score=score,
                    valid=valid,wall_seconds=time.monotonic()-tick,input_sha256=hashes,diagnostics=diagnostics))
                if method=='old':
                    check=None if independent else replay_check(args.model,seed,row);row['replay']=check
                    keep(row)
                    if check and not check['passed']:raise ValueError('nonlinear old replay failed')
                    continue
            else:
                outputs=[program(theta,tf.one_hot(k,spec.parameter_count,dtype=tf.float64),obs,*new_input,beta) for k in range(spec.parameter_count)]
                values=[float(x['value']) for x in outputs];scores=[float(x['score']) for x in outputs]
                valid=all(bool(x['valid']) and int(x['steps_completed'])==T for x in outputs)
                valid=valid and all(math.isfinite(v) for v in values+scores) and max(values)-min(values)<=1e-10*(1+abs(values[0]))
                row=dict(model=args.model,method=method,seed=seed,value=values[0],score=scores,valid=valid,
                         wall_seconds=time.monotonic()-tick,input_sha256=hashes,beta=tuning['beta'],
                         steps_completed=[int(x['steps_completed']) for x in outputs],trace_fields=TRACE_FIELDS,
                         trace=outputs[0]['trace'],directional_value_spread=max(values)-min(values))
            keep(row)
    write(folder/'completion.json',dict(complete=True,wall_seconds=time.monotonic()-started,rows=len(rows)))


def source_paths():
    return [Path(__file__),Path(candidate.__file__),PLAN,*sorted((ROOT/'bayesfilter/highdim').glob('covariance_proposal*_tf.py')),
            *[ROOT/'bayesfilter/highdim'/p for p in ('sqmc_campaign_tf.py','sqmc_lgssm_tf.py','sqmc_ksc_tf.py','sqmc_nonlinear_tf.py',
               'ledh_canonical_score_tf.py','ledh_canonical_score_stages_tf.py','ledh_marginal_weights_tf.py')]]


def run(args):
    OUT.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',TF_FORCE_GPU_ALLOW_GROWTH='true',TF_CPP_MIN_LOG_LEVEL='2',
             TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    attempts=[]
    def launch(label,command):
        used=sum(x['wall_seconds'] for x in attempts)
        if len(attempts)>=24 or used>=14400:raise RuntimeError('campaign budget exhausted')
        limit=min(1800,14400-used);start=time.monotonic();log=OUT/f'{label}.log'
        entry=dict(label=label,command=command,log=str(log),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   timeout=limit,status='running',wall_seconds=0.)
        attempts.append(entry);write(OUT/'attempts.json',attempts)
        with log.open('w') as stream:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            entry['pid']=proc.pid;write(OUT/'attempts.json',attempts)
            try:code=proc.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code=124
        entry.update(status='complete',exit_code=code,wall_seconds=time.monotonic()-start)
        write(OUT/'attempts.json',attempts);print(json.dumps(entry),flush=True)
        return code
    write(OUT/'campaign.json',dict(plan=str(PLAN),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         command=sys.argv,budget_seconds=14400,max_attempts=24,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_paths()},
         environment=sys.executable,cuda_visible_devices='1',models=MODELS,seeds=SEEDS,data_seed=DATA,particles=N,horizon=T,
         result_file='docs/benchmarks/ledh-matched-comparison-results-20261008.md'))
    for model in MODELS:
        tune=OUT/f'{model}-tuning'
        options=['--device','gpu','--dtype','float64','--no-tf32','--particles',str(N),'--horizon',str(T),
                 '--model',model,'--lgssm-family','p44','--flow-steps','16','--reset-steps','40','--reset-epsilon','1',
                 '--correction-steps','0']
        code=launch(model+'-calibration',[sys.executable,'-B',str(Path(candidate.__file__)),'calibrate',*options,
                     '--seed','261008100','--data-seed','26100820','--output',str(tune)])
        if code:continue
        launch(model+'-comparison',[sys.executable,'-B',str(Path(__file__)),'worker',*options,
               '--tuning',str(tune/'tuning.json'),'--output',str(OUT/f'{model}-comparison')])
    report(args)


def stats(values):
    mean=statistics.mean(values);se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None
    return dict(n=len(values),mean=mean,se=se)


def report(args):
    summary=[];all_rows=[]
    for model in MODELS:
        folder=OUT/f'{model}-comparison'
        if not (folder/'manifest.json').exists() or not (folder/'rows.json').exists():continue
        manifest=read(folder/'manifest.json');ref=manifest['reference'];rows=read(folder/'rows.json')
        for row in rows:
            if row['valid']:
                row['absolute_value_error']=abs(row['value']-ref['value'])
                row['score_error']=[a-b for a,b in zip(row['score'],ref['score'])]
                row['score_l2_error']=math.sqrt(sum(x*x for x in row['score_error']))
        all_rows.extend(rows)
        arms={}
        for method in ('old','old_covariance_only','new'):
            group=[r for r in rows if r['method']==method and r['valid']]
            if group:arms[method]=dict(value=stats([r['value'] for r in group]),
                score=[stats([r['score'][j] for r in group]) for j in range(len(ref['score']))],
                absolute_value_error=stats([r['absolute_value_error'] for r in group]),
                score_l2_error=stats([r['score_l2_error'] for r in group]))
        paired={}
        for metric in ('absolute_value_error','score_l2_error'):
            old={r['seed']:r for r in rows if r['method']=='old' and r['valid']}
            new={r['seed']:r for r in rows if r['method']=='new' and r['valid']}
            delta=[new[s][metric]-old[s][metric] for s in SEEDS if s in old and s in new]
            if delta:
                value=stats(delta)
                value['paired_differences']=delta
                value['t95_interval']=([value['mean']-3.182446305284*value['se'],value['mean']+3.182446305284*value['se']] if len(delta)==4 else None)
                paired[metric]=value
        summary.append(dict(model=model,reference=ref,arms=arms,paired=paired,beta=manifest['tuning']['beta'],
                            invalid_rows=[dict(method=r['method'],seed=r['seed']) for r in rows if not r['valid']],
                            complete=(folder/'completion.json').exists()))
    write(OUT/'comparison.json',dict(models=summary,rows=all_rows,
          default_promoted=False,scope='one fixed dataset per model, four paired designs; frozen implementation diagnostic',
          uncertainty='t intervals use df=3, assume approximately Gaussian paired differences and condition on numerical references; no population generalization'))
    lines=['# Matched T=50 comparison','',
           'N=1008, FP64 GPU/XLA, TF32 disabled; original data seed 26100611. Four paired designs. '
           'Old and new receive identical initial/process tensors. The candidate also uses its required independent categorical variates.','',
           '| Model | Method | Mean log likelihood | Mean absolute likelihood error | Mean score L2 error |','|---|---|---:|---:|---:|']
    for result in summary:
        for method,arm in result['arms'].items():
            lines.append(f"| {result['model']} | {method} | {arm['value']['mean']:.9f} | {arm['absolute_value_error']['mean']:.9f} | {arm['score_l2_error']['mean']:.9f} |")
    lines+=['','All individual values, score coordinates, reference uncertainties, input hashes, validity checks and paired differences are in `comparison.json`.']
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([dict(model=x['model'],complete=x['complete'],invalid=x['invalid_rows'],paired=x['paired']) for x in summary]),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','worker','report'])
    p.add_argument('--model',choices=MODELS,default='lgssm');p.add_argument('--output');p.add_argument('--tuning')
    p.add_argument('--device',choices=['gpu','cpu'],default='gpu');p.add_argument('--dtype',default='float64')
    p.add_argument('--no-tf32',action='store_true');p.add_argument('--lgssm-family',default='p44')
    p.add_argument('--particles',type=int,default=N);p.add_argument('--horizon',type=int,default=T)
    p.add_argument('--flow-steps',type=int,default=16);p.add_argument('--reset-steps',type=int,default=40)
    p.add_argument('--reset-epsilon',type=float,default=1.);p.add_argument('--correction-steps',type=int,default=0)
    p.add_argument('--dataset-file',type=Path)
    p.add_argument('--data-seed',type=int,default=DATA)
    p.add_argument('--design-seeds',type=int,nargs='+',default=SEEDS)
    p.add_argument('--independent',action='store_true',help='Checked new dataset; historical replay does not apply')
    p.add_argument('--plan-file',default=str(PLAN))
    p.add_argument('--result-file',default='docs/benchmarks/ledh-matched-comparison-results-20261008.md')
    args=p.parse_args()
    if args.particles!=N or args.horizon!=T or args.dtype!='float64':p.error('matched scope is fixed')
    return globals()[args.command](args) or 0

if __name__=='__main__':raise SystemExit(main())
